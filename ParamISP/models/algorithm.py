#!/usr/bin/env python
import argparse

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import make_grid
import kornia.losses
import numpy as np

import layers.bayer
import layers.color
import data.utils
import data.modules
import utils.metrics
import utils.io
import utils.convert
import models.utils.arg as arg
import models.utils.cli as cli
from models.utils.image_logger import ImageWriter


class CommonArgs(
    arg.ProcessArgs,
    arg.RunsPath,
    arg.BayerPattern,
    arg.NumWorkers,
    arg.Profiler,
    arg.Mode,
):
    inverse: bool

    @classmethod
    def add_to(cls, parser: argparse.ArgumentParser):
        arg.RunsPath.add_to(parser)
        parser.add_argument("--inverse", action="store_true", help="Inverse the tone curve")
        arg.BayerPattern.add_to(parser)
        arg.NumWorkers.add_to(parser)
        arg.Profiler.add_to(parser)


class TestArgs(
    CommonArgs,
    arg.CameraModels,
):
    @classmethod
    def add_to(cls, parser: argparse.ArgumentParser):
        arg.CameraModels.add_to(parser)


class PredictArgs(
    CommonArgs,
    arg.CameraModel,
    arg.SampleIndices,
):
    @classmethod
    def add_to(cls, parser: argparse.ArgumentParser):
        arg.CameraModel.add_to(parser)
        arg.SampleIndices.add_to(parser)


def parse_args():
    parser = arg.Parser(description=AlgorithmOnlyModel.__doc__,
                        namespaces=[CommonArgs, TestArgs, PredictArgs])
    CommonArgs.add_to(parser)
    subparsers = arg.Mode.add_subparsers(parser)
    TestArgs.add_to(subparsers.add_parser("test"))
    PredictArgs.add_to(subparsers.add_parser("predict"))
    return parser.parse_args()


def main():
    cli.init_env()
    args = CommonArgs(parse_args())

    model = AlgorithmOnlyModel(args)

    if args.mode == "test":
        args = TestArgs(args)

        def get_datamodule(camera_model): return data.modules.CameraTestingData(
            camera_model, crop_type="5", bayer_pattern=args.bayer_pattern,
            batch_size=5, crop_size=1024, num_workers=args.num_workers)

        cli.test(args, model, get_datamodule)

    elif args.mode == "predict":
        args = PredictArgs(args)

        if len(args.sample_indices) < args.num_workers:
            args.num_workers = len(args.sample_indices)

        datamodule = data.modules.CameraTestingData(
            args.camera_model, crop_size=1920, crop_type="center", bayer_pattern=args.bayer_pattern,
            select_index=args.sample_indices, num_workers=args.num_workers)

        cli.predict(args, model, datamodule)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


class AlgorithmOnlyModel(pl.LightningModule):
    """ Use traditional ISP algorithms only. """

    def __init__(self, args: arg.Namespace):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.set_attributes()

    def set_attributes(self):
        self.demosaic = layers.bayer.Demosaic()

    def tensorboard_add_images(self, images: dict, step: int | None = None, dataformats: str = "CHW"):
        tensorboard: SummaryWriter = self.logger.experiment  # type: ignore
        for name, image in images.items():
            tensorboard.add_image(name, image.clip(0, 1), global_step=step, dataformats=dataformats)

    def csv_log_metrics(self, metrics: dict, step: int | None = None):
        assert self.logger is not None
        self.logger.log_metrics(metrics, step=step)

    def log_on_training(self, metrics: dict, batch_size: int):
        self.log_dict({
            f"train/{key}/step": value for key, value in metrics.items()
        }, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        self.log_dict({
            **{f"train/{key}/epoch": value for key, value in metrics.items()},
            "step": float(self.current_epoch),
        }, on_step=False, on_epoch=True, batch_size=batch_size)

    def log_on_validation(self, metrics: dict, batch_size: int):
        self.log_dict({
            **{f"val/{key}": value for key, value in metrics.items()},
            "step": float(self.current_epoch),
        }, on_step=False, on_epoch=True, batch_size=batch_size)

    def log_on_test(self, metrics: dict, batch: data.utils.ImageDataBatch, batch_idx: int):
        inst_infos = zip(batch["camera_name"], batch["index"], batch["inst_id"], batch["window_top_left"])
        inst_infos = "/".join(f"{camera_name}:[{index}]{inst_id}:{top}:{left}"
                              for camera_name, index, inst_id, (top, left) in inst_infos)

        self.csv_log_metrics({
            **metrics,
            "target": inst_infos,
        }, batch_idx)

        self.log_dict(metrics, on_step=False, on_epoch=True, batch_size=batch["raw"].size(0))

    def log_images_on_validation(self, images: dict, batch: data.utils.ImageDataBatch, raw: bool = False):
        if raw:
            images = {key: self.visualize_raw(value, batch["bayer_pattern"], batch["white_balance"], batch["color_matrix"])
                      for key, value in images.items()}

        batch_size = batch["raw"].size(0)
        if batch_size > 4:
            images = {key: value[:4] for key, value in images.items()}
            batch_size = 4

        self.tensorboard_add_images({
            key: make_grid(value, nrow=batch_size, value_range=(0, 1))
            for key, value in images.items()
        }, self.current_epoch)

    def log_image_on_predict(self, image: torch.Tensor, name: str):
        imwriter: ImageWriter = self.logger.experiment  # type: ignore
        imwriter.log_image(image, name)

    @classmethod
    def sub_state_dict(cls, state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
        if not prefix.endswith("."):
            prefix += "."
        return {k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}

    @classmethod
    def crop(cls, x: torch.Tensor) -> torch.Tensor:
        return x[..., 2:-2, 2:-2]

    def compute_metrics(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        x = self.crop(x)
        y = self.crop(y)
        return {
            "L1": F.l1_loss(x, y),
            "psnr": utils.metrics.psnr(x, y, 1.0),
            "ssim": utils.metrics.ssim(x, y),
            "ms-ssim": utils.metrics.ms_ssim(x.clip(min=1e-9), y.clip(min=1e-9)),
        }

    def visualize_raw(self, x: torch.Tensor, bayer_pattern: torch.Tensor,
                      white_balance: torch.Tensor, color_matrix: torch.Tensor):
        x = layers.bayer.apply_white_balance(x, bayer_pattern, white_balance)
        x = self.demosaic(x, bayer_pattern)
        x = layers.color.apply_color_matrix(x.clip(0, 1), color_matrix).clip(0, 1)
        x = layers.color.linear_to_srgb(x)
        return x

    def forward(self, batch: data.utils.ImageDataBatch, inverse: bool = False):
        if inverse:
            x = batch["rgb"]
            # x = layers.color.srgb_to_linear(x)
            x = x.clip(min=1e-12).pow(2.2)
            x = 0.5 - torch.sin(torch.arcsin(1 - 2*x)/3)
            x = layers.color.apply_color_matrix(x, batch["color_matrix"].inverse())
            x = layers.bayer.mosaic(x, batch["bayer_pattern"])
            x = layers.bayer.apply_white_balance(x, batch["bayer_pattern"], 1/batch["white_balance"])
        else:
            x = batch["raw"]
            x = layers.bayer.apply_white_balance(x, batch["bayer_pattern"], batch["white_balance"]).clip(0, 1)
            x = self.demosaic(x, batch["bayer_pattern"])
            x = layers.color.apply_color_matrix(x, batch["color_matrix"])
            x = 3*x**2 - 2*x**3
            x = x.clip(min=1e-12).pow(1/2.2)
            # x = layers.color.linear_to_srgb(x)

        return x

    def test_step(self, batch: data.utils.ImageDataBatch, batch_idx: int):
        args = TestArgs(self.args)

        x = self(batch, inverse=args.inverse)
        y = batch["raw" if args.inverse else "rgb"]

        metrics = self.compute_metrics(x, y)

        self.log_on_test(metrics, batch, batch_idx)

        return metrics

    def predict_step(self, batch: data.utils.ImageDataBatch, batch_idx: int):
        args = PredictArgs(self.args)

        x: torch.Tensor = self(batch, inverse=args.inverse)
        y = batch["raw" if args.inverse else "rgb"]
        assert x.size(0) == 1
        index   = batch["index"][0]
        inst_id = batch["inst_id"][0]

        if args.inverse:
            vx = self.visualize_raw(x, batch["bayer_pattern"], batch["white_balance"], batch["color_matrix"])
            vy = self.visualize_raw(y, batch["bayer_pattern"], batch["white_balance"], batch["color_matrix"])
            self.log_image_on_predict(x,  f"{index:04d}_{inst_id}_raw.tiff")
            self.log_image_on_predict(vx, f"{index:04d}_{inst_id}_raw_vis.png")
            self.log_image_on_predict(torch.abs(vx - vy), f"{index:04d}_{inst_id}_raw_diff.png")
        else:
            self.log_image_on_predict(x, f"{index:04d}_{inst_id}_rgb.png")
            self.log_image_on_predict(torch.abs(x - y), f"{index:04d}_{inst_id}_rgb_diff.png")


if __name__ == "__main__":
    main()
