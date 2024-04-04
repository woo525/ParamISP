from typing import Literal, TypeAlias
from pathlib import Path
import re

import pytorch_lightning as pl
import torch
import torch.utils.data
import torch.utils.data.distributed
import numpy as np

from utils.path import projroot
import utils.env
import utils.io
from .utils import ImageDataBatch, ConcatDataset, collate_image_data, \
    PatchDataset, RandomCropPatchDataset, CenterCropPatchDataset, FiveCropPatchDataset, ThirteenCropPatchDataset, \
    RGBPatchDataset, RandomCropRGBPatchDataset, CenterCropRGBPatchDataset, FiveCropRGBPatchDataset, \
    RandomSampler, DistributedRandomSampler, EqualizedRandomSampler, DistributedEqualizedRandomSampler, \
    SelectedIndexSampler, DistributedSelectedIndexSampler, \
    CameraModel, EVERY_CAMERA_MODEL, CropType, EVERY_CROP_TYPE

import layers.bayer

def get_training_dataset(
    camera_model: CameraModel,
    crop_size: int = 512,
    bayer_pattern: np.ndarray | str | None = None,
    use_extra: bool = False,
):
    """ Build training dataset.

    Args:
        camera_model: Target camera device name of dataset.
        crop_size: Crop size of patches.
        bayer_pattern: Bayer pattern to align, None for random target (without align).
        use_extra: Whether to use extra metadata.

    Returns:
        Training dataset.
    """
    match camera_model:
        case "A7R3":
            data_dir = utils.env.get_or_throw("REALBLURSRC_PATCHSET_DIR")
            datalist_file = projroot/"data/datalist/A7R3.train.txt"
        case "D7000":
            data_dir = utils.env.get_or_throw("RAISE_PATCHSET_DIR")
            datalist_file = projroot/"data/datalist/D7000.train.txt"
        case "D90":
            data_dir = utils.env.get_or_throw("RAISE_PATCHSET_DIR")
            datalist_file = projroot/"data/datalist/D90.train.txt"
        case "D40":
            data_dir = utils.env.get_or_throw("RAISE_PATCHSET_DIR")
            datalist_file = projroot/"data/datalist/D40.train.txt"
        case "S7":
            data_dir = utils.env.get_or_throw("S7ISP_PATCHSET_DIR")
            datalist_file = projroot/"data/datalist/S7.train.txt"
        case _: raise ValueError(f"Invalid dataset type: {camera_model}")

    return RandomCropPatchDataset(
        datalist_file=datalist_file, data_dir=Path(data_dir),
        crop_size=crop_size, bayer_pattern=bayer_pattern, use_extra=use_extra)


def get_validation_dataset(
    camera_model: CameraModel,
    crop_size: int = 512,
    bayer_pattern: np.ndarray | str | None = None,
    use_extra: bool = False,
):
    """ Build validation dataset.

    Args:
        camera_model: Target camera device name of dataset.
        crop_size: Crop size of patches.
        bayer_pattern: Bayer pattern to align, None for random target (without align).
        use_extra: Whether to use extra metadata.

    Returns:
        Training dataset.
    """
    match camera_model:
        case "A7R3":
            data_dir = utils.env.get_or_throw("REALBLURSRC_PATCHSET_DIR")
            datalist_file = projroot/"data/datalist/A7R3.val.txt"
        case "D7000":
            data_dir = utils.env.get_or_throw("RAISE_PATCHSET_DIR")
            datalist_file = projroot/"data/datalist/D7000.val.txt"
        case "D90":
            data_dir = utils.env.get_or_throw("RAISE_PATCHSET_DIR")
            datalist_file = projroot/"data/datalist/D90.val.txt"
        case "D40":
            data_dir = utils.env.get_or_throw("RAISE_PATCHSET_DIR")
            datalist_file = projroot/"data/datalist/D40.train.txt"
        case "S7":
            data_dir = utils.env.get_or_throw("S7ISP_PATCHSET_DIR")
            datalist_file = projroot/"data/datalist/S7.val.txt"
        case _: raise ValueError(f"Invalid dataset type: {camera_model}")

    return FiveCropPatchDataset(
        datalist_file=datalist_file, data_dir=Path(data_dir),
        crop_size=crop_size, bayer_pattern=bayer_pattern, use_extra=use_extra)


def get_test_dataset(
    camera_model: CameraModel,
    crop_type: CropType = "13",
    crop_size: int = 512,
    bayer_pattern: np.ndarray | str | None = None,
    use_extra: bool = False,
):
    """ Build test dataset.

    Args:
        camera_model: Target camera device name of dataset.
        crop_size: Crop size of patches.
        bayer_pattern: Bayer pattern to align, None for random target (without align).
        use_extra: Whether to use extra metadata.

    Returns:
        Training dataset.
    """
    match camera_model:
        case "A7R3":
            data_dir = utils.env.get_or_throw("REALBLURSRC_PATCHSET_DIR")
            datalist_file = projroot/"data/datalist/A7R3.test.txt"
        case "D7000":
            data_dir = utils.env.get_or_throw("RAISE_PATCHSET_DIR")
            datalist_file = projroot/"data/datalist/D7000.test.txt"
        case "D90":
            data_dir = utils.env.get_or_throw("RAISE_PATCHSET_DIR")
            datalist_file = projroot/"data/datalist/D90.test.txt"
        case "D40":
            data_dir = utils.env.get_or_throw("RAISE_PATCHSET_DIR")
            datalist_file = projroot/"data/datalist/D40.test.txt"
        case "S7":
            data_dir = utils.env.get_or_throw("S7ISP_PATCHSET_DIR")
            datalist_file = projroot/"data/datalist/S7.test.txt"
        case _: raise ValueError(f"Invalid dataset type: {camera_model}")

    match crop_type:
        case "full":
            datamodule = PatchDataset(
                datalist_file=datalist_file, data_dir=Path(data_dir),
                bayer_pattern=bayer_pattern, use_extra=use_extra)
        case "center":
            datamodule = CenterCropPatchDataset(
                datalist_file=datalist_file, data_dir=Path(data_dir),
                crop_size=crop_size, bayer_pattern=bayer_pattern, use_extra=use_extra)
        case "5":
            datamodule = FiveCropPatchDataset(
                datalist_file=datalist_file, data_dir=Path(data_dir),
                crop_size=crop_size, bayer_pattern=bayer_pattern, use_extra=use_extra)
        case "13":
            datamodule = ThirteenCropPatchDataset(
                datalist_file=datalist_file, data_dir=Path(data_dir),
                crop_size=crop_size, bayer_pattern=bayer_pattern, use_extra=use_extra)
        case _: raise ValueError(f"Invalid crop type: {crop_type}")

    return datamodule


def get_random_dataloader(
    dataset: torch.utils.data.Dataset,
    num_samples: int,
    batch_size: int,
    num_workers: int,
    ddp_mode: bool = False,
):
    """ Build random dataloader.

    Args:
        dataset: Training dataset.
        num_samples: Sampling size for each epoch.
        batch_size: Batch size.
        num_workers: Number of workers for dataloader, 0 for main process only.
        ddp_mode: Whether to use DistributedDataParallel.

    Returns:
        Dataloader.
    """
    if ddp_mode:
        sampler = DistributedRandomSampler(dataset, num_samples)
    else:
        sampler = RandomSampler(dataset, num_samples)

    return torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, drop_last=True,
        sampler=sampler, collate_fn=collate_image_data,
        num_workers=num_workers, persistent_workers=num_workers > 0)


def get_multi_random_dataloader(
    datasets: list[torch.utils.data.Dataset],
    num_samples: int,
    batch_size: int,
    num_workers: int,
    ddp_mode: bool = False,
):
    """ Build random dataloader for multiple datasets.

    Args:
        datasets: Training datasets.
        num_samples: Sampling size for each epoch.
        batch_size: Batch size.
        num_workers: Number of workers for dataloader, 0 for main process only.
        ddp_mode: Whether to use DistributedDataParallel.

    Returns:
        Dataloader.
    """
    if ddp_mode:
        sampler = DistributedEqualizedRandomSampler(datasets, num_samples)
    else:
        sampler = EqualizedRandomSampler(datasets, num_samples)
    
    dataset = ConcatDataset(*datasets)

    return torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, drop_last=True,
        sampler=sampler, collate_fn=collate_image_data,
        num_workers=num_workers, persistent_workers=num_workers > 0)


def get_sequential_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    select_index: list[int] | None = None,
    ddp_mode: bool = False,
):
    """ Build random dataloader.

    Args:
        dataset: Training dataset.
        batch_size: Batch size.
        num_workers: Number of workers for dataloader, 0 for main process only.
        ddp_mode: Whether to use DistributedDataParallel.

    Returns:
        Dataloader.
    """
    if select_index:
        if ddp_mode:
            sampler = DistributedSelectedIndexSampler(dataset, select_index)
        else:
            sampler = SelectedIndexSampler(dataset, select_index)
    else:
        if ddp_mode:
            sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
        else:
            sampler = None

    return torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, collate_fn=collate_image_data,
        sampler=sampler, num_workers=num_workers, persistent_workers=num_workers > 0)


class CameraTestingData(pl.LightningDataModule):
    """ Single camera data for testing.

    Args:
        camera_model: Target camera device name of dataset.
        crop_type: Crop mode; 0(no crop), 1(center crop), 5 crop, 13 crop are available.
        crop_size: Crop size of patches.
        bayer_pattern: Bayer pattern to align, None for random target (without align).
        use_extra: Whether to use extra metadata.
        selected_index: Selected index for sequential sampling, applied only for predict mode.
        batch_size: Batch size, 0 for auto setting.
        num_workers: Number of workers for dataloader, 0 for main process only.
        ddp_mode: Whether to use DistributedDataParallel.
    """

    def __init__(self, camera_model: CameraModel, crop_type: CropType = "13",
                 crop_size: int = 512, bayer_pattern: np.ndarray | str | None = None, use_extra: bool = False,
                 select_index: list[int] | None = None, batch_size: int = 0, num_workers: int = 0, ddp_mode: bool = False):
        super().__init__()
        self.camera_model: CameraModel = camera_model
        self.crop_type: CropType = crop_type
        self.crop_size = crop_size
        self.bayer_pattern = bayer_pattern
        self.use_extra = use_extra
        self.select_index = select_index
        self.num_workers = num_workers
        self.ddp_mode = ddp_mode
        self.batch_size = batch_size

        if batch_size == 0:
            match crop_type:
                case "full" | "center" | "random": self.batch_size = 1
                case "5":  self.batch_size = 5
                case "13": self.batch_size = 13

    def test_dataloader(self):
        return get_sequential_dataloader(
            dataset=get_test_dataset(
                camera_model=self.camera_model,
                crop_type=self.crop_type,
                crop_size=self.crop_size,
                bayer_pattern=self.bayer_pattern,
                use_extra=self.use_extra,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            ddp_mode=self.ddp_mode,
        )

    def predict_dataloader(self):
        return get_sequential_dataloader(
            dataset=get_test_dataset(
                camera_model=self.camera_model,
                crop_type=self.crop_type,
                crop_size=self.crop_size,
                bayer_pattern=self.bayer_pattern,
                use_extra=self.use_extra,
            ),
            select_index=self.select_index,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            ddp_mode=self.ddp_mode,
        )


class CameraTrainingData(pl.LightningDataModule):
    """ Multiple camera data for training.

    Args:
        camera_models: Target camera devices of dataset.
        num_samples: Sampling size per camera model for each epoch.
        crop_size: Crop size of patches.
        bayer_pattern: Bayer pattern to align, None for random target (without align).
        use_extra: Whether to use extra metadata.
        batch_size: Batch size.
        num_workers: Number of workers for dataloader, 0 for main process only.
        ddp_mode: Whether to use DistributedDataParallel.
    """

    def __init__(self, camera_models: list[CameraModel] = EVERY_CAMERA_MODEL, num_samples: int = 1024,
                 crop_size: int = 512, bayer_pattern: np.ndarray | str | None = None, use_extra: bool = False,
                 batch_size: int = 4, num_workers: int = 0, ddp_mode: bool = False):
        super().__init__()
        self.camera_models = camera_models
        self.num_samples = num_samples
        self.crop_size = crop_size
        self.bayer_pattern = bayer_pattern
        self.use_extra = use_extra
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ddp_mode = ddp_mode

    def train_dataloader(self):
        datasets: list[torch.utils.data.Dataset] = [
            get_training_dataset(
                camera_model=camera_model,
                crop_size=self.crop_size,
                bayer_pattern=self.bayer_pattern,
                use_extra=self.use_extra,
            ) for camera_model in self.camera_models
        ]
        return get_multi_random_dataloader(
            datasets=datasets,
            num_samples=self.num_samples,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            ddp_mode=self.ddp_mode,
        )

    def val_dataloader(self):
        dataset = [
            get_validation_dataset(
                camera_model=camera_model,
                crop_size=self.crop_size,
                bayer_pattern=self.bayer_pattern,
                use_extra=self.use_extra,
            ) for camera_model in self.camera_models
        ]
        return get_sequential_dataloader(
            dataset=ConcatDataset(*dataset),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            ddp_mode=self.ddp_mode,
        )
