from typing import Callable
from pathlib import Path
import shutil
import os
import sys
import logging

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import utils.path
import utils.env
import utils.io
import data.modules

from . import arg
from .image_logger import ImageLogger

from pytorch_lightning.strategies.ddp import DDPStrategy ##

@rank_zero_only
def check_exists(*path: str | Path, is_dir: bool = True):
    """ Raise error if given directory already exists and is not empty.

    Args:
        path: Path to check.
        is_dir: Whether the path should be a directory.
    """
    _path = Path(*path)

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


def init_env(less_logging: bool = False):
    os.environ["LRU_CACHE_SIZE"] = "16"
    utils.env.load()
    if less_logging:
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


class NamedProgressBar(TQDMProgressBar):
    def __init__(self, description: str):
        super().__init__()
        self.description = description

    def on_test_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                            batch, batch_idx: int, dataloader_idx: int):
        super().on_test_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)
        self.test_progress_bar.set_description(self.description)

    def on_predict_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                               batch, batch_idx: int, dataloader_idx: int):
        super().on_predict_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)
        self.predict_progress_bar.set_description(self.description)


def train(
    args: arg.Namespace,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    metric: str = "val/psnr", metric_mode: str = "max",
):
    """ Train a model.

    Args:
        model: Model to train.
        datamodule: Dataset to use.
        args: Arguments to use, should contain RunsPath, ResumePath, SaveConfig, and Profiler.
        metric: Metric for saving checkpoints and early stopping.
        best_metric: Whether to maximize or minimize the metric.
    """
    runs_path = arg.RunsPath(args).runs_path
    ckpt_dir = (runs_path/"checkpoints").as_posix()
    check_exists(runs_path, is_dir=True)

    try:
        profiler = arg.Profiler(args).profiler
    except (ValueError, AttributeError):
        print("[Warning] Profiler not specified", file=sys.stderr)
        profiler = None

    try:
        resume_path = arg.ResumePath(args).resume_path
    except (ValueError, AttributeError):
        print("[Warning] ResumePath not specified, training from scratch", file=sys.stderr)
        resume_path = None

    try:
        save_best = arg.SaveConfig(args).save_best
        save_freq = arg.SaveConfig(args).save_freq
        early_stop = arg.SaveConfig(args).early_stop
    except (ValueError, AttributeError):
        print("[Warning] SaveConfig not specified, saving only the last checkpoint", file=sys.stderr)
        save_best = 0
        save_freq = None
        early_stop = 0

    logger = TensorBoardLogger(save_dir=runs_path.parent.as_posix(), name=runs_path.name, version="train",
                               default_hp_metric=False)

    callbacks: list[Callback] = []

    if save_best > 0:
        callbacks.append(ModelCheckpoint(
            dirpath=ckpt_dir, filename="best-{"+metric+":.4f}-{epoch:03d}",
            monitor=metric, mode=metric_mode, save_top_k=save_best, save_last=True,
            auto_insert_metric_name=False))

    if save_freq is not None:
        callbacks.append(ModelCheckpoint(
            dirpath=ckpt_dir, filename="epoch-{epoch:03d}",
            save_top_k=-1, every_n_epochs=save_freq,
            auto_insert_metric_name=False))

    if early_stop > 0:
        callbacks.append(EarlyStopping(
            monitor=metric, mode=metric_mode, patience=early_stop))

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, callbacks=callbacks, profiler=profiler, strategy = DDPStrategy(find_unused_parameters=True)) ##

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_path)

    trainer.save_checkpoint(utils.path.purify(ckpt_dir, "final.ckpt"))


def test(
    args: arg.Namespace,
    model: pl.LightningModule,
    get_datamodule: Callable[[str], data.modules.CameraTestingData],
):
    """ Test a model.

    Args:
        args: Arguments to use, should contain RunsPath, CameraModels, CheckpointPaths, and Profiler.
        model: Model to test.
        get_datamodule: Function to get the datamodule for a camera model.

    Returns:
        The test results, as a dictionary.
    """
    runs_path = arg.RunsPath(args).runs_path
    # check_exists(runs_path, "test", is_dir=True)

    camera_models = arg.CameraModels(args).camera_models

    try:
        profiler = arg.Profiler(args).profiler
    except (ValueError, AttributeError):
        print("[Warning] Profiler not specified", file=sys.stderr)
        profiler = None

    try:
        ckpt_paths = arg.CheckpointPaths(args).ckpt_paths
    except (ValueError, AttributeError):
        ckpt_paths = None  # no checkpoint required

    def test_case(ckpt_path: Path | None):
        results = {}

        if ckpt_path is not None:
            log_dir = runs_path.as_posix(), "test", ckpt_path.stem
            cb_str = f"Testing {ckpt_path.stem}"
        else:
            log_dir = runs_path.parent.as_posix(), runs_path.name, "test"
            cb_str = "Testing"

        for camera_model in camera_models:
            datamodule = get_datamodule(camera_model)

            logger = CSVLogger(*log_dir)
            logger.experiment.NAME_METRICS_FILE = f"metrics.{camera_model}.csv"
            logger.experiment.metrics_file_path = \
                Path(logger.experiment.log_dir, logger.experiment.NAME_METRICS_FILE).as_posix()

            callbacks = [NamedProgressBar(f"{cb_str} ({camera_model})")]

            trainer: pl.Trainer = pl.Trainer.from_argparse_args(
                args, logger=logger, callbacks=callbacks, profiler=profiler)

            results[camera_model] = trainer.test(model, datamodule=datamodule,
                                                 ckpt_path=ckpt_path.as_posix() if ckpt_path else None)

        utils.io.saveyaml(results, *log_dir, "metrics.yaml")

    if ckpt_paths is not None:
        for ckpt_path in ckpt_paths:
            test_case(ckpt_path)
    else:
        test_case(None)


def predict(
    args: arg.Namespace,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
):
    """ Predict with a model.

    Args:
        args: Arguments to use, should contain RunsPath, CheckpointPaths, and Profiler.
        model: Model to predict with.
        datamodule: Dataset to use.
    """
    runs_path = arg.RunsPath(args).runs_path

    try:
        profiler = arg.Profiler(args).profiler
    except (ValueError, AttributeError):
        print("[Warning] Profiler not specified", file=sys.stderr)
        profiler = None

    try:
        ckpt_paths = arg.CheckpointPaths(args).ckpt_paths
    except (ValueError, AttributeError):
        ckpt_paths = None  # no checkpoint required

    def predict_case(ckpt_path: Path | None):
        if ckpt_path is not None:
            log_dir = runs_path.as_posix(), "predict", ckpt_path.stem
            cb_str = f"Predicting {ckpt_path.stem}"
        else:
            log_dir = runs_path.parent.as_posix(), runs_path.name, "predict"
            cb_str = "Predicting"

        logger = ImageLogger(*log_dir)
        callbacks = [NamedProgressBar(f"{cb_str}")]

        trainer: pl.Trainer = pl.Trainer.from_argparse_args(
            args, logger=logger, callbacks=callbacks, profiler=profiler)

        trainer.predict(model, datamodule=datamodule,
                        ckpt_path=ckpt_path.as_posix() if ckpt_path else None)

    if ckpt_paths is not None:
        for ckpt_path in ckpt_paths:
            predict_case(ckpt_path)
    else:
        predict_case(None)
