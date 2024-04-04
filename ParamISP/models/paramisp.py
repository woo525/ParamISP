from typing import Callable
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim
from torchvision.utils import make_grid
import kornia.color
import kornia.filters
import kornia.losses
from kornia.geometry.transform import rescale
import matplotlib.pyplot as plt

import sys
sys.path.append('./') 

import utils.metrics
import utils.io
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
        parser.add_argument("--inverse",   action="store_true",  help="Inverse the tone curve")
        arg.BayerPattern.add_to(parser)
        arg.NumWorkers.add_to(parser)
        arg.Profiler.add_to(parser)


class TrainArgs(
    CommonArgs,
    arg.CameraModels,
    arg.CropSize,
    arg.NumSamples,
    arg.BatchSize,
    arg.DDPMode,
    arg.Seed,
    arg.MaxEpochs,
    arg.AdamConfig,
    arg.StepLRConfig,
    arg.ResumePath,
    arg.PretrainedPath,
    arg.PreviousStageWeightPath,
    arg.SaveConfig,
):
    @classmethod
    def add_to(cls, parser: argparse.ArgumentParser):
        arg.CameraModels.add_to(parser)
        arg.CropSize.add_to(parser, 448)
        arg.NumSamples.add_to(parser)
        arg.BatchSize.add_to(parser, 8)
        arg.DDPMode.add_to(parser)
        arg.Seed.add_to(parser)
        arg.MaxEpochs.add_to(parser)
        arg.AdamConfig.add_to(parser, lr=2e-4)
        arg.StepLRConfig.add_to(parser, gamma=0.8, step=10)
        arg.ResumePath.add_to(parser)
        arg.PretrainedPath.add_to(parser)
        arg.PreviousStageWeightPath.add_to(parser)
        arg.SaveConfig.add_to(parser)


class TestArgs(
    CommonArgs,
    arg.CameraModels,
    arg.CheckpointPaths,
    arg.BatchSize,
):
    @classmethod
    def add_to(cls, parser: argparse.ArgumentParser):
        arg.CameraModels.add_to(parser)
        arg.CheckpointPaths.add_to(parser)
        arg.BatchSize.add_to(parser, 0)


class PredictArgs(
    CommonArgs,
    arg.CameraModel,
    arg.CheckpointPaths,
):
    index_file: str

    @classmethod
    def add_to(cls, parser: argparse.ArgumentParser):
        arg.CameraModel.add_to(parser)
        arg.CheckpointPaths.add_to(parser)


def parse_args():
    parser = arg.Parser(description=ParamISP.__doc__,
                        namespaces=[CommonArgs, TrainArgs, TestArgs, PredictArgs])
    CommonArgs.add_to(parser)
    subparsers = arg.Mode.add_subparsers(parser)
    TrainArgs.add_to(subparsers.add_parser("train"))
    TestArgs.add_to(subparsers.add_parser("test"))
    PredictArgs.add_to(subparsers.add_parser("predict"))
    return parser.parse_args()


def main():
    cli.init_env()
    args = CommonArgs(parse_args())

    model = ParamISP(args)

    if args.mode == "train":
        args = TrainArgs(args)
        
        datamodule = data.modules.CameraTrainingData(
            camera_models=args.camera_models, num_samples=args.num_samples,
            crop_size=args.crop_size, bayer_pattern=args.bayer_pattern, use_extra=True,
            batch_size=args.batch_size, num_workers=args.num_workers, ddp_mode=args.ddp)
        
        cli.train(args, model, datamodule)

    elif args.mode == "test":
        args = TestArgs(args)

        def get_datamodule(camera_model): return data.modules.CameraTestingData(
            camera_model, crop_type="5", bayer_pattern=args.bayer_pattern, use_extra=True,
            crop_size=1024, batch_size=5, num_workers=args.num_workers)

        cli.test(args, model, get_datamodule)

    elif args.mode == "predict":
        args = PredictArgs(args)

        index_list = [0,1,2,3,4] # index list
    
        datamodule = data.modules.CameraTestingData(
            args.camera_model, crop_type="full", bayer_pattern=args.bayer_pattern, use_extra=True,
            select_index=index_list, num_workers=args.num_workers)

        cli.predict(args, model, datamodule)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


class ParamISP(AlgorithmOnlyModel):

    def set_attributes(self):
        super().set_attributes()
        args = CommonArgs(self.args)
        self.automatic_optimization = False

        self.ms_ssim_loss = kornia.losses.MS_SSIMLoss() # do not use

        self.optics_dropout_probability = 0.2

        self.id_channels = 16 # camera id variant
        self.feat_channels = (3 * (1 + 2 + 4+8+16) + 3) # pixel, gradient, soft histogram, over-exposed mask

        self.main_channels = 64
        self.hyper_channels = 64

        self.tonenet = layers.nn.v2.GlobalNet(self.feat_channels, self.main_channels)
        self.localnet = layers.nn.v2.LocalNet(self.feat_channels, self.main_channels, num_block=2, num_scale=2)

        self.embed_id    = layers.nn.v2.Linear(self.id_channels, self.hyper_channels) # place holder; do not use
        self.embed_color = layers.nn.v2.Linear(24, self.hyper_channels) # white balance, color matrix
        self.embed_focallen = layers.nn.v2.Linear(15, self.hyper_channels) # focal length
        self.embed_aperture = layers.nn.v2.Linear(15, self.hyper_channels) # f-number
        self.embed_expstime = layers.nn.v2.Linear(15, self.hyper_channels) # exposure time
        self.embed_senstvty = layers.nn.v2.Linear(15, self.hyper_channels) # sensitivity

        self.embednet = nn.Sequential(
            nn.ELU(inplace=True), layers.nn.v2.Linear(self.hyper_channels, self.hyper_channels),
            nn.ELU(inplace=True), layers.nn.v2.Linear(self.hyper_channels, self.hyper_channels),
            nn.ELU(inplace=True), layers.nn.v2.Linear(self.hyper_channels,
                                                        self.hyper_channels, init_weights=lambda x: init.xavier_normal_(x, 0.1)),
        )

        self.hypertonenet = layers.nn.v2.HyperGlobalNet(self.tonenet, self.hyper_channels)
        self.hyperlocalnet = layers.nn.v2.SynHyperLocalNet(self.localnet, self.hyper_channels)

    def configure_optimizers(self):
        args = TrainArgs(self.args)

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                      lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_gamma)
        return [optimizer], [scheduler]

    def get_embedding(self, batch: data.utils.ImageDataBatch, training_mode: bool = False):
        args = CommonArgs(self.args)

        embed: torch.Tensor
        embed = 0 * self.embed_id(data.utils.embed_camera_id(data.utils.get_camera_id_batch(batch), self.id_channels))
     
        embed = embed + self.embed_color(
            torch.cat([data.utils.embed_white_balance(batch["white_balance"]),
                       data.utils.embed_color_matrix(batch["color_matrix"])], dim=-1))

        optics_mask = torch.ones(4, device=embed.device)
        optics_mask = torch.bernoulli(optics_mask * (1 - self.optics_dropout_probability))

        embed = embed + self.embed_focallen(data.utils.embed_focal_length(batch["focal_length"]))
        embed = embed + self.embed_aperture(data.utils.embed_aperture(batch["f_number"]))
        embed = embed + self.embed_expstime(data.utils.embed_exposure_time(batch["exposure_time"]))
        embed = embed + self.embed_senstvty(data.utils.embed_sensitivity(batch["iso_sensitivity"]))

        embed = self.embednet(embed)
        return embed

    def get_common_features(self, x: torch.Tensor, batch: data.utils.ImageDataBatch):
        args = CommonArgs(self.args)

        common_features = layers.nn.v2.over_exposed_mask(x)

        return common_features

    def get_input_features(self, x: torch.Tensor, z: torch.Tensor):
        args = CommonArgs(self.args)

        return torch.cat([x, layers.nn.v2.content_features(
            x, use_gradient=True, histogram_bins=[4, 8, 16]), z], dim=-3)

    def forward(self, batch: data.utils.ImageDataBatch, training_mode: bool = False, extra: bool = False):
        args = TrainArgs(self.args)

        embed = self.get_embedding(batch, training_mode)

        if args.inverse: # inverse
            x: torch.Tensor = batch["rgb"]
            if training_mode:
                x = x + torch.randn_like(x) * 0.4 / 255. # add dequantization noise
            
            common_features = self.get_common_features(x, batch)
            
            z = self.get_input_features(x, common_features)
            x = self.hypertonenet(x, z, embed)

            z = self.get_input_features(x, common_features)
            x = self.hyperlocalnet(x, z, embed)
          
            x = layers.color.apply_color_matrix(x, batch["color_matrix"].inverse())
            x = layers.bayer.apply_white_balance(x, batch["bayer_pattern"], 1/batch["white_balance"], mosaic_flag=False)
            x = layers.bayer.mosaic(x, batch["bayer_pattern"])

        else: # forward
            x = batch["raw"]
            
            if training_mode: # add dequantization noise
                x = x + torch.randn_like(x) * 0.4 / batch["quantized_level"].unsqueeze(-1).unsqueeze(-1)
                x = layers.bayer.gather(x, batch["bayer_pattern"])

            x = self.demosaic(x, batch["bayer_pattern"])
            x = layers.bayer.apply_white_balance(x, batch["bayer_pattern"], batch["white_balance"], mosaic_flag=False)
            x = x.clip(0, 1)
            x = layers.color.apply_color_matrix(x, batch["color_matrix"])

            common_features = self.get_common_features(x, batch)
    
            z = self.get_input_features(x, common_features)
            x = self.hyperlocalnet(x, z, embed)
        
            z = self.get_input_features(x, common_features)
            x = self.hypertonenet(x, z, embed)

        if training_mode:
            
            y = batch["raw" if args.inverse else "rgb"]
            loss = F.l1_loss(self.crop(x), self.crop(y))

            return x, loss

        else:
            return x

    def training_step(self, batch: data.utils.ImageDataBatch, batch_idx: int):
        args = TrainArgs(self.args)
        
        x, loss = self(batch, training_mode=True)
        
        optimizer = self.optimizers()
        optimizer.zero_grad()
  
        if self.current_epoch > 10:
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.manual_backward(loss)

        optimizer.step()

        self.log_on_training({"loss": loss}, batch_size=batch["raw"].size(0))

        return loss

    def validation_step(self, batch: data.utils.ImageDataBatch, batch_idx: int):
        args = TrainArgs(self.args)

        x = self(batch)
        y = batch["raw" if args.inverse else "rgb"]

        metrics = self.compute_metrics(x, y)

        self.log_on_validation(metrics, batch_size=batch["raw"].size(0))
        if batch_idx == 0:
            self.log_images_on_validation({"val/raw": batch["raw"]}, batch, raw=True)
            self.log_images_on_validation({"val/rgb": batch["rgb"]}, batch, raw=False)
            self.log_images_on_validation({"val/est": x}, batch, raw=args.inverse)

        return metrics

    def test_step(self, batch: data.utils.ImageDataBatch, batch_idx: int):
        args = TestArgs(self.args)

        y = batch["raw" if args.inverse else "rgb"]
        x = self(batch).clip(0,1) 

        metrics = self.compute_metrics(x, y)

        self.log_on_test(metrics, batch, batch_idx)

        return metrics

    def predict_step(self, batch: data.utils.ImageDataBatch, batch_idx: int):
        args = PredictArgs(self.args)

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

        inst_id = batch["inst_id"][0]
        index = batch["index"][0].item()
    
        if "weights" in args.ckpt_names[0]: # using official weights
            out_path = Path(self.args.runs_root, self.args.runs_name, self.args.mode, "official_weights", args.camera_model, str(index).zfill(3))
        else:
            out_path = Path(self.args.runs_root, self.args.runs_name, self.args.mode, args.ckpt_names[0], args.camera_model, str(index).zfill(3))
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path.as_posix()

        x = self(batch).clip(0, 1)
        
        if args.inverse:
            utils.io.savetiff(batch["raw"], out_path, f"{index}-{inst_id}-raw_gt.tiff", u16=True)
            utils.io.savetiff(x, out_path, f"{index}-{inst_id}-raw_est.tiff", u16=True)
        else:
            utils.io.saveimg(batch["rgb"], out_path, f"{index}-{inst_id}-rgb_gt.png")
            utils.io.saveimg(x, out_path, f"{index}-{inst_id}-rgb_est.png")


if __name__ == "__main__":
    main()
