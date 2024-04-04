from collections import OrderedDict
from typing import Callable
import os
import shutil
import argparse
from datetime import datetime
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import kornia.color
import kornia.filters
import kornia.losses
from kornia.geometry.transform import rescale
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from tqdm.auto import tqdm

from pytorch_lightning.strategies.ddp import DDPStrategy

import sys
sys.path.append('./')

import utils.metrics
import utils.io
import utils.env
from utils.path import projroot
import utils.misc
import data.utils
import data.modules
import layers.bayer
import layers.color
import layers.misc
import layers.nn.v2
import models.utils.arg as arg
import models.utils.cli as cli
from models.algorithm import AlgorithmOnlyModel
import models.paramisp


def parse_args():
    parser = argparse.ArgumentParser(description="InvISP")
    parser.add_argument("-o", "--runs-name", metavar="NAME", required=True, help="Name of the output runs folder")  # noqa: E501
    parser.add_argument("-O", "--runs-root", metavar="PATH", default=(projroot/"runs").as_posix(), help="Parent path of the runs directory")  # noqa: E501
    parser.add_argument("--bayer-pattern", metavar="CCCC", choices=["RGGB", "GRBG", "GBRG", "BGGR"], default="RGGB", help="Bayer pattern to use")  # noqa: E501
    parser.add_argument("-j", "--num-workers", metavar="N", type=int, default=4, help="Number of workers for data loading")  # noqa: E501

    subparsers = parser.add_subparsers(dest="mode", required=True, description="Train or Test")  # noqa: E501

    parser_train = subparsers.add_parser("train", help="Train InvISP")
    parser_train.add_argument("--camera", metavar="[NAME,...]", type=lambda s: s.split(","), default=",".join(data.modules.EVERY_CAMERA_MODEL), help="Name of the dataset to use")  # noqa: E501
    parser_train.add_argument("--crop-size", metavar="N", type=int, default=448, help="Crop size of patches")  # noqa: E501
    parser_train.add_argument("-b", "--batch-size", metavar="N", type=int, default=2, help="Batch size for training")  # noqa: E501
    parser_train.add_argument("-l", "--num-samples", metavar="N", type=int, default=1024, help="Sampling size for each training epoch")  # noqa: E501
    parser_train.add_argument("--max-epochs", metavar="N", type=int, default=-1, help="Maximum number of training epochs")  # noqa: E501
    parser_train.add_argument("--lr", metavar="F", type=float, default=1e-4, help="Learning rate")  # noqa: E501
    parser_train.add_argument("--wd", metavar="F", type=float, default=1e-12, dest="weight_decay", help="Weight decay")  # noqa: E501
    parser_train.add_argument("--lr-gamma", metavar="F", type=float, default=0.8, help="Learning rate decay factor")  # noqa: E501
    parser_train.add_argument("--lr-step", metavar="N", type=int, default=10, help="Learning rate decay step")  # noqa: E501
    parser_train.add_argument("--seed", metavar="N", type=int, default=42, help="Random seed")  # noqa: E501
    parser_train.add_argument("--no-ddp", dest="ddp", action="store_false", help="Disable DDP mode")  # noqa: E501
    parser_train.add_argument("--save-freq", metavar="N", type=int, default=500, help="Frequency of saving checkpoints")  # noqa: E501
    parser_train.add_argument("--early-stop", metavar="N", type=int, default=100, help="Number of epochs to wait before early stopping")  # noqa: E501
    parser_train.add_argument("--resume", metavar="PATH", help="Path to the checkpoint to resume training from")  # noqa: E501
    parser_train.add_argument("--pisp-inv", metavar="PATH", required=True, help="Path to the pisp model")  # noqa: E501
    parser_train.add_argument("--pisp-fwd", metavar="PATH", required=True, help="Path to the pisp model")  # noqa: E501

    parser_test = subparsers.add_parser("test", help="Test InvISP")
    parser_test.add_argument("--camera", metavar="[NAME,...]", type=lambda s: s.split(","), default=",".join(data.modules.EVERY_CAMERA_MODEL), help="Name of the dataset to use")  # noqa: E501
    parser_test.add_argument("--ckpt", metavar="[NAME,...]", default="best-*", help="Checkpoint names")  # noqa: E501
    parser_test.add_argument("-b", "--batch-size", metavar="N", type=int, default=0, help="Batch size for testing")  # noqa: E501

    parser_predict = subparsers.add_parser("predict", help="Predict InvISP")
    parser_predict.add_argument("--camera", metavar="[NAME,...]", type=lambda s: s.split(","), default=",".join(data.modules.EVERY_CAMERA_MODEL), help="Name of the dataset to use")  # noqa: E501
    parser_predict.add_argument("--ckpt", metavar="[NAME,...]", default="best-*", help="Checkpoint names")  # noqa: E501

    args = parser.parse_args()
    args.gpus = torch.cuda.device_count()

    if hasattr(args, "ddp") and args.ddp:
        args.strategy = "ddp"
        args.replace_sampler_ddp = False
    else:
        args.strategy = None

    return args


@rank_zero_only
def check_exists(path: str, is_dir: bool = True):
    """ Raise error if given directory already exists and is not empty.

    Args:
        path: Path to check.
        is_dir: Whether the path should be a directory.
    """
    _path = Path(path)

    if _path.exists():
        if is_dir:
            if not _path.is_dir():
                raise ValueError(f"Path {_path} already exists and is not a directory.")
            if any(_path.iterdir()):
                while True:
                    choice = input(f"'{_path}' is not empty. Overwrite? [Y/c/n]: ").lower()
                    if choice in ["y", "yes"] or choice.isspace() or choice == "":
                        break
                    elif choice in ["c", "clear"]:
                        shutil.rmtree(_path)
                        break
                    elif choice in ["n", "no"]:
                        print("Aborted.")
                        exit(1)
        else:
            if not _path.is_file():
                raise ValueError(f"{_path.as_posix()} already exists and is not a file.")
            while True:
                choice = input(f"'{_path}' already exists. Overwrite? [Y/n]: ").lower()
                if choice in ["y", "yes"]:
                    break
                elif choice in ["n", "no"]:
                    print("Aborted.")
                    exit(1)


def main():
    os.environ["LRU_CACHE_SIZE"] = "16"
    utils.env.load()
    if False:
        import logging
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    args = parse_args()

    model = ParamISPJointly(args)  # type: ignore

    if args.mode == "train":
        check_exists(Path(args.runs_root, args.runs_name).as_posix(), is_dir=True)

        datamodule = data.modules.CameraTrainingData(
            camera_models=args.camera, num_samples=args.num_samples,
            crop_size=args.crop_size, bayer_pattern=args.bayer_pattern, use_extra=True,
            batch_size=args.batch_size, num_workers=args.num_workers, ddp_mode=args.ddp)

        logger = TensorBoardLogger(save_dir=args.runs_root, name=args.runs_name,
                                   version="train", default_hp_metric=False)

        checkpoint_dir = Path(args.runs_root, args.runs_name, "checkpoints").as_posix()
        callbacks = [
            ModelCheckpoint(dirpath=checkpoint_dir, filename="best_raw-{val/psnr_raw_1ch:.4f}-{epoch:03d}",
                            monitor="val/psnr_raw_1ch", mode="max", save_top_k=1, save_last=True, auto_insert_metric_name=False),
            ModelCheckpoint(dirpath=checkpoint_dir, filename="best_rgb-{val/psnr_rgb_rec:.4f}-{epoch:03d}",
                            monitor="val/psnr_rgb_rec", mode="max", save_top_k=1, save_last=True, auto_insert_metric_name=False),
            ModelCheckpoint(dirpath=checkpoint_dir, filename="best_avg-{val/psnr_avg:.4f}-{epoch:03d}",
                            monitor="val/psnr_avg", mode="max", save_top_k=1, save_last=True, auto_insert_metric_name=False),
            ModelCheckpoint(dirpath=checkpoint_dir, filename="epoch-{epoch:03d}",
                            save_top_k=-1, every_n_epochs=args.save_freq, auto_insert_metric_name=False),
            EarlyStopping(monitor="val/psnr_avg", mode="max", patience=args.early_stop)
        ]

        trainer: Trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks, strategy = DDPStrategy(find_unused_parameters=False)) ##

        trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)

        trainer.save_checkpoint(Path(checkpoint_dir, "final.ckpt").as_posix())

    elif args.mode == "test":
        Path(args.runs_root, args.runs_name, "test").mkdir(parents=True, exist_ok=True)
       
        ckpt_paths = []
        for ckpt_name in args.ckpt.split(","):
            ckpt_paths.extend([Path(ckpt_name)])
        
        ckpt_paths = [ckpt_path.as_posix() for ckpt_path in ckpt_paths]
        ckpt_paths = sorted(set(ckpt_paths), reverse=True)
        
        if len(ckpt_paths) == 0:
            raise ValueError(f"No checkpoint found for '{args.ckpt}'")

        for ckpt_path in ckpt_paths:
            ckpt_name = Path(ckpt_path).stem

            results = {}

            for camera in args.camera:
                datamodule = data.modules.CameraTestingData(
                    camera, crop_type="5", bayer_pattern=args.bayer_pattern, use_extra=True,
                    batch_size=5, crop_size=1024, num_workers=args.num_workers)

                logger = CSVLogger(save_dir=Path(args.runs_root, args.runs_name).as_posix(),
                                   name="test", version=ckpt_name)
                logger.experiment.NAME_METRICS_FILE = f"metrics.{camera}.csv"
                logger.experiment.metrics_file_path = \
                    Path(logger.experiment.log_dir, logger.experiment.NAME_METRICS_FILE).as_posix()

                trainer: Trainer = Trainer.from_argparse_args(args, logger=logger)

                print(f"\033[0;33mTesting {camera} with {ckpt_name} ...\033[0m")
                results[camera] = trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)

            utils.io.saveyaml(results, Path(args.runs_root, args.runs_name, "test", f"metrics.yaml").as_posix())

    elif args.mode == "predict":

        Path(args.runs_root, args.runs_name, "predict").mkdir(exist_ok=True)

        ckpt_path = args.ckpt
        index_list = [0, 1, 2, 3, 4]

        datamodule = data.modules.CameraTestingData(
            args.camera[0], crop_type="full", bayer_pattern=args.bayer_pattern, use_extra=True,
            select_index=index_list, num_workers=args.num_workers)

        trainer: Trainer = Trainer.from_argparse_args(args)
        trainer.predict(model, datamodule=datamodule, ckpt_path=ckpt_path)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


class ParamISPJointly(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.demosaic = layers.bayer.Demosaic()

        self.args_rgb2raw = models.paramisp.CommonArgs(inverse=True)
        self.rgb2raw = models.paramisp.ParamISP(self.args_rgb2raw)
        
        self.args_raw2rgb = models.paramisp.CommonArgs(inverse=False) 
        self.raw2rgb = models.paramisp.ParamISP(self.args_raw2rgb)
        
        if args.mode == "train": # load pretrained weights
            if args.pisp_inv is not None:
                weight = utils.io.loadpt(args.pisp_inv)["state_dict"]
                self.rgb2raw.load_state_dict(weight)
            if args.pisp_fwd is not None:
                weight = utils.io.loadpt(args.pisp_fwd)["state_dict"]
                self.raw2rgb.load_state_dict(weight)

    def configure_optimizers(self):
        args = self.args

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                      lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_gamma)
        return [optimizer], [scheduler]

    def forward(self, batch: data.utils.ImageDataBatch, training_mode: bool = False):
        args = self.args
        
        raw_gt = batch["raw"]
        rgb_gt = batch["rgb"]
        
        if training_mode:
            raw_est, rgb2raw_loss = self.rgb2raw(batch, training_mode)
            rgb_est, raw2rgb_loss = self.raw2rgb(batch, training_mode)
            batch["raw"] = raw_est
            rgb_rc_est, raw2rgb_rc_loss = self.raw2rgb(batch, training_mode)
            batch["raw"] = raw_gt

            loss = rgb2raw_loss + raw2rgb_rc_loss + F.l1_loss(rgb_rc_est, rgb_gt)

            return raw_est, rgb_est, rgb_rc_est, loss

        else:
            raw_est = self.rgb2raw(batch)
            rgb_est = self.raw2rgb(batch)
            
            """ Apply quantization in RAW space """
            raw_est = raw_est.clip(0,1) # 0~1, float
            raw_est = ((raw_est * 2**14) + 0.5).type(torch.int) # 0~16383, int
            raw_est = raw_est/(2**14-1) # 0~1, float
            """"""
            batch["raw"] = raw_est
            rgb_rc_est = self.raw2rgb(batch)
            batch["raw"] = raw_gt

            return raw_est, rgb_est, rgb_rc_est

    def training_step(self, batch: data.utils.ImageDataBatch, batch_idx: int):
        raw_est, rgb_est, rgb_rc_est, loss = self(batch, training_mode=True)

        self.log("train/loss", loss, on_step=True, prog_bar=True, batch_size=raw_est.size(0))

        return loss

    def validation_step(self, batch: data.utils.ImageDataBatch, batch_idx: int):
        raw_gt = batch["raw"]
        rgb_gt = batch["rgb"]

        raw_est, rgb_est, rgb_rc_est = self(batch)
        metrics = {
            "psnr_raw_1ch": utils.metrics.psnr(raw_est, raw_gt, 1.),
            "ssim_raw_1ch": utils.metrics.ssim(raw_est, raw_gt),
            "psnr_rgb_rec": utils.metrics.psnr(rgb_rc_est, rgb_gt, 1.),
            "ssim_rgb_rec": utils.metrics.ssim(rgb_rc_est, rgb_gt),
        }
        metrics["psnr_avg"] = (metrics["psnr_raw_1ch"] + metrics["psnr_rgb_rec"]) / 2

        self.log_dict({
            **{f"val/{key}": value for key, value in metrics.items()},
            "step": float(self.current_epoch),
        }, on_step=False, on_epoch=True, batch_size=rgb_gt.size(0))

        if batch_idx == 0:
            writer: SummaryWriter = self.logger.experiment
            writer.add_image("val/raw_gt_vis",
                             make_grid(self.raw_visualize(raw_gt, batch["bayer_pattern"], batch["white_balance"], batch["color_matrix"]),
                                       nrow=4, value_range=(0, 1)),
                             global_step=self.current_epoch)
            writer.add_image("val/raw_est_vis",
                             make_grid(self.raw_visualize(raw_est, batch["bayer_pattern"], batch["white_balance"], batch["color_matrix"]),
                                       nrow=4, value_range=(0, 1)),
                             global_step=self.current_epoch)
            writer.add_image("val/rgb_gt_vis",
                             make_grid(rgb_gt,
                                       nrow=4, value_range=(0, 1)),
                             global_step=self.current_epoch)
            writer.add_image("val/rgb_rc_est_vis",
                             make_grid(rgb_rc_est,
                                       nrow=4, value_range=(0, 1)),
                             global_step=self.current_epoch)
        return metrics

    def raw_visualize(self, x: torch.Tensor, bayer_pattern: torch.Tensor,
                      white_balance: torch.Tensor, color_matrix: torch.Tensor):
        x = layers.bayer.apply_white_balance(x, bayer_pattern, white_balance)
        x = self.demosaic(x, bayer_pattern)
        x = layers.color.apply_color_matrix(x.clip(0, 1), color_matrix).clip(0, 1)
        x = layers.color.linear_to_srgb(x)
        return x

    def test_step(self, batch: data.utils.ImageDataBatch, step: int):
        
        rgb_gt = batch["rgb"]
        raw_gt = batch["raw"]

        raw_est, rgb_est, rgb_rc_est = self(batch)
        
        metrics = {
            "psnr_raw": utils.metrics.psnr(raw_est.clip(0,1), raw_gt, 1.),
            "ssim_raw": utils.metrics.ssim(raw_est.clip(0,1), raw_gt),
            "psnr_rgb_recon": utils.metrics.psnr(rgb_rc_est.clip(0,1), rgb_gt, 1.),
            "ssim_rgb_recon": utils.metrics.ssim(rgb_rc_est.clip(0,1), rgb_gt),
            "psnr_rgb_gt": utils.metrics.psnr(rgb_est.clip(0,1), rgb_gt, 1.),
            "ssim_rgb_gt": utils.metrics.ssim(rgb_est.clip(0,1), rgb_gt)
        }

        inst_infos = " / ".join(f"{camera_name.rsplit('/', 1)[-1]} [{index}]{inst_id} top={top} left={left}"
                                for camera_name, index, inst_id, (top, left) in
                                zip(batch["camera_name"], batch["index"], batch["inst_id"], batch["window_top_left"]))

        assert self.logger is not None
        self.logger.log_metrics({**metrics, "instances": inst_infos}, step=step)  # type: ignore
        self.log_dict(metrics, on_step=False, on_epoch=True, batch_size=rgb_gt.size(0))

        return metrics
    
    def predict_step(self, batch: data.utils.ImageDataBatch, batch_idx: int): ##### to use inverseISP Network ##### 
        args = self.args
        # crop for GPU memory
        h, w = batch["raw"].shape[-2:]
        max_h = 2400; max_w = 3600
        if h > max_h:
            cut_h = (((h - max_h)//2)//2)*2
            batch["raw"] = batch["raw"][..., cut_h:cut_h+max_h, :]
            batch["rgb"] = batch["rgb"][..., cut_h:cut_h+max_h, :]
        if w > max_w:
            cut_w = (((w - max_w)//2)//2)*2
            batch["raw"] = batch["raw"][..., :, cut_w:cut_w+max_w]
            batch["rgb"] = batch["rgb"][..., :, cut_w:cut_w+max_w]
        # to be devided by 4 for inference
        h, w = batch["raw"].shape[-2:]
        batch["raw"] = batch["raw"][..., :(h//4)*4, :(w//4)*4]
        batch["rgb"] = batch["rgb"][..., :(h//4)*4, :(w//4)*4]
        
        camera = args.camera[0] # D7000
        inst_id = batch["inst_id"][0] # 'r65d8aee2t'
        index = batch["index"][0].item() # 385
        
        out_path = os.path.join(args.runs_root, args.runs_name, "predict", camera) # predict folder
        if not os.path.exists(out_path): os.makedirs(out_path)

        with torch.no_grad():
            raw_est, raw_est_3ch, rgb_est, rgb_rc_est, rgb_rc_est_lin = self(batch)

            utils.io.saveimg(batch["rgb"], out_path, f"{inst_id}"+"_"+f"{str(index).zfill(4)}_gt.png") # gt
            utils.io.saveimg(rgb_rc_est_lin, out_path, f"{inst_id}"+"_"+f"{str(index).zfill(4)}_est.png") # est


if __name__ == "__main__":
    main()
