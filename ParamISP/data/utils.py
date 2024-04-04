from typing import Final, Callable, Sized, Literal, Iterator, TypeAlias, TypedDict
from pathlib import Path
from fractions import Fraction
import math
import re
import random

import torch
import torch.nn.functional as F
import torch.utils.data
import torch.distributed
import numpy as np

import utils.env
import utils.io
import utils.convert
import utils.camera
import layers.misc

PATCH_SIZE: Final = 512

CameraModel: TypeAlias = Literal["A7R3", "D7000", "D90", "D40", "S7"]
EVERY_CAMERA_MODEL: list[CameraModel] = re.findall(r"'(.+?)'", str(CameraModel))

CropType: TypeAlias = Literal["full", "center", "random", "5", "13"]
EVERY_CROP_TYPE: list[CropType] = re.findall(r"'(.+?)'", str(CropType))


def patch_name(imtype: Literal["raw", "rgb"], i: int, j: int) -> str:
    return f"{imtype}-{PATCH_SIZE}-{i:05}-{j:05}.tif"


class ImageData(TypedDict):
    # image contents
    raw: np.ndarray
    """ Normalized RAW image, with shape (H, W) """
    rgb: np.ndarray
    """ Normalized RGB image i.e. JPEG output from camera, with shape (H, W, 3) """
    # metadata
    bayer_pattern: np.ndarray
    """ Bayer pattern, with shape (2, 2) """
    white_balance: np.ndarray
    """ Normalized white balance, with shape (3,) """
    color_matrix: np.ndarray
    """ CameraRGB -> sRGB D65 conversion matrix, with shape (3, 3) """
    # extra metadata
    focal_length: int
    """ Focal length in mm, 0 if unknown """
    f_number: float
    """ F-number of aperture, 0 if unknown """
    exposure_time: float
    """ Exposure time in seconds, 0 if unknown """
    iso_sensitivity: int
    """ ISO sensitivity, 0 if unknown """
    # debugging metadata
    camera_name: str
    """ Camera name, e.g. "SONY/ILCE-7RM3" """
    inst_id: str
    """ Canonical instance ID """
    index: int
    """ Index of the image in the dataset """
    full_image_size: tuple[int, int]
    """ Full size (H, W) of original image """
    window_top_left: tuple[int, int]
    """ Top-left corner (I, J) of the window """
    transform: tuple[bool, bool, int]
    """ Transform representation applied to the RGB image, as (flip_v, flip_h, rot_ccw)
        e.g. (False, True, 3) for "Mirrored horizontal then rotated 90 CW" """
    quantized_level: np.ndarray
    """ Quantized level of each color channel of the RAW image, with shape (3,) """


class ImageDataBatch(TypedDict):
    raw: torch.Tensor
    """ Normalized RAW images, with shape (N, 1, H, W) """
    rgb: torch.Tensor
    """ Normalized RGB images i.e. JPEG outputs from camera, with shape (N, 3, H, W) """
    bayer_pattern: torch.Tensor
    """ Bayer pattern for each data, with shape (N, 1, 2, 2) """
    white_balance: torch.Tensor
    """ Normalized white balance for each data, with shape (N, 3) """
    color_matrix: torch.Tensor
    """ CameraRGB -> sRGB D65 conversion matrix for each data, with shape (N, 3, 3) """
    focal_length: torch.Tensor
    """ Focal length in mm for each data, with shape (N,) """
    f_number: torch.Tensor
    """ F-number of aperture for each data, with shape (N,) """
    exposure_time: torch.Tensor
    """ Exposure time in seconds for each data, with shape (N,) """
    iso_sensitivity: torch.Tensor
    """ ISO sensitivity for each data, with shape (N,) """
    camera_name: list[str]
    """ Camera name for each data, e.g. ["SONY/ILCE-7RM3", ...] """
    inst_id: list[str]
    """ Canonical instance ID for each data, e.g. ["BLR-02091-00001", ...] """
    index: torch.Tensor
    """ Index of the image in the dataset, with shape (N,) """
    full_image_size: list[tuple[int, int]]
    """ Full size (H, W) of original image for each data """
    window_top_left: list[tuple[int, int]]
    """ Top-left corner (I, J) of the window for each data """
    transform: list[tuple[bool, bool, int]]
    """ Transform representation applied to the RGB image for each data """
    quantized_level: torch.Tensor
    """ Quantized level of each color channel of the RAW image, with shape (N, 3) """


CropWindow: TypeAlias = tuple[int, int, int, int]


def refine_crop_size(crop_size: int | tuple[int, int]) -> tuple[int, int]:
    """ Refine crop size to be (H, W). """
    if isinstance(crop_size, int):
        assert crop_size % 2 == 0, f"Crop size must be even, but got {crop_size}"
        crop_size = (int(crop_size), int(crop_size))
    else:
        H, W = crop_size
        assert H % 2 == 0 or W % 2 == 0, f"Crop size must be even, but got {crop_size}"
        crop_size = int(H), int(W)
    return crop_size


def collate_image_data(data: list[ImageData]) -> ImageDataBatch:
    """ Collate a list of ImageData into a ImageDataBatch. """
    
    return ImageDataBatch(
        raw=torch.stack([utils.convert.as_torch_tensor(d["raw"]) for d in data], dim=0),
        rgb=torch.stack([utils.convert.as_torch_tensor(d["rgb"]) for d in data], dim=0),
        bayer_pattern=torch.stack([utils.convert.as_torch_tensor(d["bayer_pattern"]) for d in data], dim=0),
        white_balance=torch.stack([torch.from_numpy(d["white_balance"]) for d in data], dim=0),
        color_matrix=torch.stack([torch.from_numpy(d["color_matrix"]) for d in data], dim=0),
        focal_length=torch.tensor([d["focal_length"] for d in data]),
        f_number=torch.tensor([d["f_number"] for d in data]),
        exposure_time=torch.tensor([d["exposure_time"] for d in data]),
        iso_sensitivity=torch.tensor([d["iso_sensitivity"] for d in data]),
        camera_name=[d["camera_name"] for d in data],
        inst_id=[d["inst_id"] for d in data],
        index=torch.tensor([d["index"] for d in data]),
        full_image_size=[d["full_image_size"] for d in data],
        window_top_left=[d["window_top_left"] for d in data],
        transform=[d["transform"] for d in data],
        quantized_level=torch.stack([torch.from_numpy(d["quantized_level"]) for d in data], dim=0),
    )


def get_camera_id(camera_name: str) -> int:
    """ Get camera ID from camera name. """
    camera_name = camera_name.upper().split("/")[-1]
    camera_names = {
        "ILCE-7RM3":    1,  # Sony α7R III
        "NIKON D7000":  2,
        "NIKON D90":    3,
        "NIKON D40":    4,
        "SM-G935F":     5,  # Samsung Galaxy S7
    }
    return camera_names.get(camera_name, 0)


def get_camera_id_batch(batch: ImageDataBatch) -> torch.Tensor:
    """ Get camera ID from camera name in batch. """
    device = batch["raw"].device
    camera_ids = [get_camera_id(name) for name in batch["camera_name"]]
    return torch.tensor(camera_ids, device=device)


def get_random_camera_names(length: int) -> list[str]:
    """ Get random camera names. """
    camera_names = [
        "SONY/ILCE-7RM3",
        "NIKON CORPORATION/NIKON D7000",
        "NIKON CORPORATION/NIKON D90",
        "NIKON CORPORATION/NIKON D40",
        "SAMSUNG/SM-G935F",
    ]
    return random.choices(camera_names, k=length)


def embed_camera_id(camera_ids: torch.Tensor, n_dim: int = 32) -> torch.Tensor:
    """ Embed camera ID into batch. 

    Args:
        camera_ids: Camera ID, with shape (N,).

    Returns:
        Camera ID embedding, with shape (N, n_dim).
    """
    assert camera_ids.ndim == 1
    assert 0 <= camera_ids.min() and camera_ids.max() < n_dim
    return F.one_hot(camera_ids, n_dim).float()


def embed_white_balance(white_balance: torch.Tensor) -> torch.Tensor:
    """ Embed white balance into batch. 

    Args:
        white_balance: White balance, with shape (N, 3).

    Returns:
        White balance embedding, with shape (N, 6).
    """
    assert white_balance.ndim >= 2 and white_balance.shape[-1] == 3
    assert 0 < white_balance.min()
    return torch.cat([white_balance, 1/white_balance], dim=-1)


def random_white_balance(white_balance: torch.Tensor) -> torch.Tensor:
    """ Random sample white balance regards to the given input size. """
    assert white_balance.ndim >= 2 and white_balance.shape[-1] == 3
    white_balance = torch.ones_like(white_balance)
    white_balance[:, 0] += torch.rand_like(white_balance[:, 0]) * 1.5
    white_balance[:, 2] += torch.rand_like(white_balance[:, 1]) * 1.5
    return white_balance


def embed_color_matrix(color_matrix: torch.Tensor) -> torch.Tensor:
    """ Embed color matrix into batch. 

    Args:
        color_matrix: Color matrix, with shape (N, 3, 3).

    Returns:
        Color matrix embedding, with shape (N, 18).
    """
    assert color_matrix.ndim >= 3 and color_matrix.shape[-2:] == (3, 3)
    return torch.cat([
        color_matrix.flatten(-2),
        color_matrix.inverse().flatten(-2),
    ], dim=-1)


def known_color_matrix(camera_names: list[str]) -> torch.Tensor:
    """ Get known color matrix from camera ID. """
    color_matrices = {
        "SONY/ILCE-7RM3": torch.tensor([
            [3.6078298,  -0.42575794, -0.18207195],
            [-2.1520138,  7.9994545,  -2.8474407],
            [0.10570342, -0.7846928,   3.6789894],
        ]),
        "NIKON CORPORATION/NIKON D7000": torch.tensor([
            [3.6998718,  -0.5884627,  -0.11140898],
            [-1.5856934,  7.326727,   -2.7410336],
            [0.04789164, -1.0564402,   4.0085487],
        ]),
        "NIKON CORPORATION/NIKON D90": torch.tensor([
            [3.824437,   -0.63158756, -0.19284935],
            [-0.0474462,  4.2565393,  -1.2090931],
            [0.31611648, -0.9302971,   3.6141806],
        ]),
        "NIKON CORPORATION/NIKON D40": torch.tensor([
            [3.5294511,  -0.48798594, -0.04146527],
            [-0.08414898, 4.286089,   -1.20194],
            [0.04546179, -0.49990866,  3.4544468],
        ]),
        "SAMSUNG/SM-G935F": torch.tensor([
            [2.237324,    -1.4440103,   0.20668642],
            [-0.2603548,   1.4750377,  -0.2146829],
            [-0.02993463, -0.93290716,  1.9628417],
        ]),
    }
    color_matrix = torch.stack([color_matrices[name] for name in camera_names], dim=0)
    return color_matrix


def embed_focal_length(focal_length: torch.Tensor) -> torch.Tensor:
    """ Embed focal length into batch.

    Args:
        focal_length: Focal length, with shape (N,).

    Returns:
        Focal length embedding, with shape (N, 15).
    """
    assert focal_length.ndim == 1
    focal_length = focal_length.clip(min=1e-12)
    return torch.stack([
        (focal_length / 200.),
        (focal_length / 200.).sqrt(),
        (focal_length / 200.) ** .25,
        1. / (focal_length / 4.2),
        1. / (focal_length / 4.2).sqrt(),
        1. / (focal_length / 4.2) ** .25,
        (focal_length.log2() - 1.9) / 5.8,
        (focal_length.log2()).sin(),
        (focal_length.log2()).cos(),
        (focal_length / 50.).sin(),
        (focal_length / 50.).cos(),
        (focal_length / 30.).sin(),
        (focal_length / 30.).cos(),
        (focal_length / 20.).sin(),
        (focal_length / 20.).cos(),
    ], dim=-1)


def embed_aperture(f_number: torch.Tensor) -> torch.Tensor:
    """ Embed aperture into batch. 

    Args:
        f_number: Aperture, with shape (N,).

    Returns:
        Aperture embedding, with shape (N, 15).
    """
    assert f_number.ndim == 1
    f_number = f_number.clip(min=1e-12)
    return torch.stack([
        (f_number / 32.),
        (f_number / 32.).sqrt(),
        (f_number / 32.) ** .25,
        1. / (f_number / 1.7),
        1. / (f_number / 1.7).sqrt(),
        1. / (f_number / 1.7) ** .25,
        (f_number.log2() - 0.7) / 4.3,
        (f_number.log2()).sin(),
        (f_number.log2()).cos(),
        (f_number / 11.).sin(),
        (f_number / 11.).cos(),
        (f_number / 5.).sin(),
        (f_number / 5.).cos(),
        (f_number / 2.).sin(),
        (f_number / 2.).cos(),
    ], dim=-1)


def embed_exposure_time(exposure_time: torch.Tensor) -> torch.Tensor:
    """ Embed exposure time into batch. 

    Args:
        exposure_time: Exposure time, with shape (N,).

    Returns:
        Exposure time embedding, with shape (N, 15).
    """
    assert exposure_time.ndim == 1
    exposure_time = exposure_time.clip(min=1e-12)
    return torch.stack([
        (exposure_time / 30.),
        (exposure_time / 30.).sqrt(),
        (exposure_time / 30.) ** .25,
        1. / (exposure_time * 28800.),
        1. / (exposure_time * 28800.).sqrt(),
        1. / (exposure_time * 28800.) ** .25,
        (exposure_time.log2() + 15.) / 20.,
        (exposure_time.log2()).sin(),
        (exposure_time.log2()).cos(),
        (exposure_time / 11.).sin(),
        (exposure_time / 11.).cos(),
        (exposure_time / 5.).sin(),
        (exposure_time / 5.).cos(),
        (exposure_time / 2.).sin(),
        (exposure_time / 2.).cos(),
    ], dim=-1)


def embed_sensitivity(sensitivity: torch.Tensor) -> torch.Tensor:
    """ Embed sensitivity into batch. 

    Args:
        sensitivity: Sensitivity, with shape (N,).

    Returns:
        Sensitivity embedding, with shape (N, 15).
    """
    assert sensitivity.ndim == 1
    sensitivity = sensitivity.clip(min=1e-12)
    return torch.stack([
        (sensitivity / 8000.),
        (sensitivity / 8000.).sqrt(),
        (sensitivity / 8000.) ** .25,
        1. / (sensitivity / 64.),
        1. / (sensitivity / 64.).sqrt(),
        1. / (sensitivity / 64.) ** .25,
        (sensitivity.log2() - 6.) / 7.,
        (sensitivity.log2()).sin(),
        (sensitivity.log2()).cos(),
        (sensitivity / 3100.).sin(),
        (sensitivity / 3100.).cos(),
        (sensitivity / 1300.).sin(),
        (sensitivity / 1300.).cos(),
        (sensitivity / 500.).sin(),
        (sensitivity / 500.).cos(),
    ], dim=-1)


def embed_hyperparams(
    batch: ImageDataBatch,
    use_camera_id: bool = True,
    use_white_balance: bool = False,
    use_color_matrix: bool = False,
    use_optical_configs: bool = False,
    use_crop_position: bool = False,
    use_quantized_level: bool = False,
) -> torch.Tensor:
    """ Encode hyperparameters into a vector.

    Args:
        batch: ImageDataBatch
        use_camera_id: Embed camera ID
        use_white_balance: Whether to embed white balance
        use_color_matrix: Whether to embed color matrix
        use_optical_configs: Whether to embed focal length, F-number, exposure time, and ISO sensitivity
        use_crop_position: Whether to embed absolute and relative crop position
        use_quantized_level: Whether to embed quantized level

    Returns:
        Hyperparameters vector, with shape (N, 3 + 9 + 4 + 4 + 3)
    """
    device = batch["raw"].device
    N = len(batch["raw"])

    if use_camera_id:
        h_id = embed_camera_id(get_camera_id_batch(batch), 42)  # (N, 42)
    else:
        h_id = torch.zeros(N, 42, device=device)  # (N, 42)

    if use_white_balance:
        h_wb = embed_white_balance(batch["white_balance"])  # (N, 6)
    else:
        h_wb = torch.zeros(N, 6, device=device)  # (N, 6)

    if use_color_matrix:
        h_col = embed_color_matrix(batch["color_matrix"])  # (N, 18)
    else:
        h_col = torch.zeros(N, 18, device=device)  # (N, 18)

    if use_optical_configs:
        focallen = batch["focal_length"].float()
        aperture = batch["f_number"].float()
        shtspeed = batch["exposure_time"].float()
        senstvty = batch["iso_sensitivity"].float()

        # if values are empty, use most common values heuristically
        eps = 1e-12
        if focallen.max() < eps:
            focallen = torch.ones(N, device=device) * 80.
        if aperture.max() < eps:
            aperture = torch.ones(N, device=device) * 2.8
        if shtspeed.max() < eps:
            shtspeed = torch.ones(N, device=device) / 125.
        if senstvty.max() < eps:
            senstvty = torch.ones(N, device=device) * 160.

        # normalize values close to [0, 1] or [-1, 1]
        if utils.env.get("HYPVER", "1") == "2":
            h_opt = torch.stack([
                focallen/200., torch.sqrt(focallen/200.), 1/(focallen/4.200), 1/torch.sqrt(focallen/4.200), (focallen.log2()-1.9)/5.8,  # noqa: E501
                aperture/32.0, torch.sqrt(aperture/32.0), 1/(aperture/1.700), 1/torch.sqrt(aperture/1.700), (aperture.log2()-0.7)/4.3,  # noqa: E501
                shtspeed/30.0, torch.sqrt(shtspeed/30.0), 1/(shtspeed*28800), 1/torch.sqrt(shtspeed*28800), (shtspeed.log2()+15.)/20.,  # noqa: E501
                senstvty/8000, torch.sqrt(senstvty/8000), 1/(senstvty/64.00), 1/torch.sqrt(senstvty/64.00), (senstvty.log2()-6.0)/7.0,  # noqa: E501
            ], dim=-1)  # (N, 20)
        else:  # 1
            h_opt = torch.stack([
                focallen/200., torch.sqrt(focallen/200.), 1/(focallen/4.000), 1/torch.sqrt(focallen/4.000), (focallen.log2()-5)/3,  # noqa: E501
                aperture/32.0, torch.sqrt(aperture/32.0), 1/(aperture/2.000), 1/torch.sqrt(aperture/2.000), (aperture.log2()-3)/2,  # noqa: E501
                shtspeed/30.0, torch.sqrt(shtspeed/30.0), 1/(shtspeed*25000), 1/torch.sqrt(shtspeed*25000), (shtspeed.log2()+5)/9,  # noqa: E501
                senstvty/8000, torch.sqrt(senstvty/8000), 1/(senstvty/64.00), 1/torch.sqrt(senstvty/64.00), (senstvty.log2()-9)/3,  # noqa: E501
            ], dim=-1)  # (N, 20)
    else:
        h_opt = torch.zeros(N, 20, device=device)  # (N, 20)

    if use_crop_position:
        top_left  = torch.as_tensor(batch["window_top_left"], device=device, dtype=torch.float32)  # (N, 2)
        full_size = torch.as_tensor(batch["full_image_size"], device=device, dtype=torch.float32)  # (N, 2)
        crop_size = torch.as_tensor(batch["raw"].shape[-2:],  device=device, dtype=torch.float32)  # (2)
        crop_size = crop_size.unsqueeze(0).repeat(N, 1)  # (N, 2)

        # normalize coordinates closely to [-1, 1]
        top_left  = top_left  / 2000.  # (N, 2)
        full_size = full_size / 2000.  # (N, 2)
        crop_size = crop_size / 2000.  # (N, 2)

        pos_abs = top_left + (crop_size / 2) - (full_size / 2)  # (N, 2)
        pos_rel = pos_abs / (full_size / 2)  # (N, 2)

        pos = torch.cat([pos_abs, pos_rel], dim=1)  # (N, 4)
        h_pos = torch.cat([
            pos,
            torch.sin(pos), torch.sin(2 * pos), torch.sin(4 * pos), torch.sin(8 * pos),
            torch.cos(pos), torch.cos(2 * pos), torch.cos(4 * pos), torch.cos(8 * pos),
            torch.sin(16 * pos), torch.sin(32 * pos), torch.sin(64 * pos), torch.sin(128 * pos),
            torch.cos(16 * pos), torch.cos(32 * pos), torch.cos(64 * pos), torch.cos(128 * pos),
            torch.sin(256 * pos), torch.sin(512 * pos), torch.sin(1024 * pos), torch.sin(2048 * pos),
            torch.cos(256 * pos), torch.cos(512 * pos), torch.cos(1024 * pos), torch.cos(2048 * pos),
        ], dim=-1)  # (N, 4*(1+6*4)) = (N, 4*25) = (N, 100)
    else:
        h_pos = torch.zeros(N, 100, device=device)  # (N, 100)

    if use_quantized_level:
        quants = batch["quantized_level"].float()  # (N, 3)
        h_quant = torch.cat([quants / 4096., torch.log2(quants / 255.)], dim=-1)  # (N, 6)
    else:
        h_quant = torch.zeros(N, 6, device=device)  # (N, 6)

    embedding = torch.cat([h_id, h_wb, h_col, h_opt, h_pos, h_quant], dim=1)  # (N, 42+6+18+20+100+6) = (N, 192)

    return embedding


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, datalist_file: Path, data_dir: Path,
                 bayer_pattern: np.ndarray | str | None = None, use_extra: bool = False):
        """ Load patched dataset.

        Args:
            datalist_file: text file containing the instances name.
            data_dir: directory containing the patched dataset.
            bayer_pattern: Match the bayer pattern of the output if not None.
        """
        self.data_dir = data_dir

        if isinstance(bayer_pattern, str):
            self.bayer_flag = utils.camera.BayerPatternFlag.from_str(bayer_pattern)
            self.bayer_pattern = utils.camera.BayerPatternFlag.numpy(self.bayer_flag)
        elif bayer_pattern is not None:
            self.bayer_pattern = np.asarray(bayer_pattern)
            self.bayer_flag = utils.camera.BayerPatternFlag.from_numpy(self.bayer_pattern)
        else:
            self.bayer_pattern = None
            self.bayer_flag = None

        self.use_extra = use_extra

        with open(datalist_file, "r") as f:
            self.datalist = [x.strip() for x in f.readlines()]
        
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index: int) -> ImageData:
        inst_id = self.datalist[index]

        metadata = self.load_metadata(inst_id)
        image_size    = metadata["image_size"]
        bayer_pattern = metadata["bayer_pattern"]

        window = self.compute_window(image_size, source_bayer_pattern=bayer_pattern)
    
        return self.load_image_data(index, inst_id, window, metadata)

    def compute_window(self, size: tuple[int, int], *,
                       source_bayer_pattern: np.ndarray | None = None) -> CropWindow:
        """ Compute the crop window of the image.

        Args:
            size: (H, W) of the full image size.
            variant: identifier for cropping multiple patches from a single instance.
            source_bayer_pattern: Bayer pattern of the output if not None.

        Returns:
            (I, J, H, W) of the crop window.
        """
        I, J = 0, 0
        if self.bayer_flag is not None:
            assert source_bayer_pattern is not None
            src_flag = utils.camera.BayerPatternFlag.from_numpy(source_bayer_pattern)
            dst_flag = self.bayer_flag

            if (src_flag ^ dst_flag) & 0b01:
                I = 1
            if (src_flag ^ dst_flag) & 0b10:
                J = 1

        H, W = size
        H -= I
        W -= J
        H -= H % 2
        W -= W % 2

        return (I, J, H, W)

    def load_image_data(self, index: int, inst_id: str, window: CropWindow, metadata: dict) -> ImageData:
        """ Get the image data of the instance included in the window. """
        # destruct metadata
        image_size    = metadata["image_size"]
        bayer_pattern = metadata["bayer_pattern"]
        black_level   = metadata["black_level"]
        white_level   = metadata["white_level"]
        white_balance = metadata["white_balance"]
        camera_matrix = metadata["camera_matrix"]
        camera_name   = metadata["camera_name"]

        # load image
        raw = self._load_image_in_window(self._load_raw_patch, inst_id, window)
        rgb = self._load_image_in_window(self._load_rgb_patch, inst_id, window)
        
        # shift bayer pattern to match the output
        I, J, H, W = window
        if I % 2 == 1:
            bayer_pattern = np.roll(bayer_pattern, 1, axis=0)
        if J % 2 == 1:
            bayer_pattern = np.roll(bayer_pattern, 1, axis=1)

        # normalize data
        rgb = utils.convert.dequantize(rgb, augmentation=False)
        raw = utils.convert.dequantize(raw, augmentation=False,
                                       in_range=(black_level, white_level), bayer_pattern=bayer_pattern)
        white_balance = utils.camera.normalize_whitebalance(white_balance.astype(np.float32))
        color_matrix  = camera_matrix.astype(np.float32)

        # load extra metadata
        if self.use_extra and (extra := self.load_extra_metadata(inst_id)):
            focal_length    = extra["focal_length"]
            f_number        = extra["f_number"]
            exposure_time   = extra["exposure_time"]
            iso_sensitivity = extra["iso_sensitivity"]
            orientation     = extra["orientation"]
        else:
            focal_length    = 0
            f_number        = 0
            exposure_time   = 0
            iso_sensitivity = 0
            orientation     = ""

        match orientation.strip():
            case "Rotated 180":                             transform = (False, False, 2)
            case "Rotated 90 CCW":                          transform = (False, False, 1)
            case "Rotated 90 CW":                           transform = (False, False, 3)
            case "Mirrored vertical":                       transform = (True,  False, 0)
            case "Mirrored horizontal":                     transform = (False, True,  0)
            case "Mirrored horizontal then rotated 90 CCW": transform = (False, True,  1)
            case "Mirrored horizontal then rotated 90 CW":  transform = (False, True,  3)
            case _:                                         transform = (False, False, 0)

        # create image data with converting data to torch tensor
        return ImageData(
            raw=raw,
            rgb=rgb,
            bayer_pattern=bayer_pattern,
            white_balance=white_balance,
            color_matrix=color_matrix,
            focal_length=focal_length,
            f_number=f_number,
            exposure_time=exposure_time,
            iso_sensitivity=iso_sensitivity,
            camera_name=camera_name,
            inst_id=inst_id,
            index=index,
            full_image_size=tuple(image_size[:2]),
            window_top_left=tuple(window[:2]),
            transform=transform,
            quantized_level=white_level-np.asarray(black_level),
        )

    def load_metadata(self, inst_id: str) -> dict:
        """ Load metadata of the instance. """
        return utils.io.loadpt(self.data_dir/inst_id/"metadata.pt")

    def load_extra_metadata(self, inst_id: str) -> dict | None:
        """ Load extra metadata of the instance. """
        try:
            extra = utils.io.loadyaml(self.data_dir/inst_id/"extra.yml")
            extra["exposure_time"] = float(Fraction(extra["exposure_time"]))
            extra["f_number"]      = float(Fraction(extra["f_number"]))
            extra["focal_length"]  = float(Fraction(extra["focal_length"]))
            extra["iso_sensitivity"] = int(extra["iso_sensitivity"])
            return extra
        except FileNotFoundError:
            return None

    def load_extra_metadata_asis(self, inst_id: str) -> dict | None:
        """ Load extra metadata of the instance. """
        try:
            extra = utils.io.loadyaml(self.data_dir/inst_id/"extra.yml")
            extra["exposure_time"] = (Fraction(extra["exposure_time"]))
            extra["f_number"]      = (Fraction(extra["f_number"]))
            extra["focal_length"]  = (Fraction(extra["focal_length"]))
            extra["iso_sensitivity"] = int(extra["iso_sensitivity"])
            return extra
        except FileNotFoundError:
            return None

    def _load_raw_patch(self, inst_id: str, i: int, j: int) -> np.ndarray:
        """ Load RAW image patch located at given index in the instance. """
        return utils.io.loadtiff(self.data_dir/inst_id/patch_name("raw", i, j)).squeeze(-1)

    def _load_rgb_patch(self, inst_id: str, i: int, j: int) -> np.ndarray:
        """ Load RGB image patch located at given index in the instance. """
        return utils.io.loadtiff(self.data_dir/inst_id/patch_name("rgb", i, j))

    def _load_image_in_window(self, patch_loader: Callable,
                              inst_id: str, window: CropWindow) -> np.ndarray:
        """ Load partial image in the window. """
        I, J, H, W = window
        
        # Least-size concatenated patches containing the window
        fI = (I // PATCH_SIZE) * PATCH_SIZE
        fJ = (J // PATCH_SIZE) * PATCH_SIZE
        patch = np.concatenate([
            np.concatenate([
                patch_loader(inst_id, i, j)
                for j in range(fJ, J+W, PATCH_SIZE)
            ], axis=1)
            for i in range(fI, I+H, PATCH_SIZE)
        ], axis=0)
        
        # Crop the patch fit to the window
        rI, rJ = I-fI, J-fJ
        patch = patch[rI:rI+H, rJ:rJ+W]

        return patch


class RGBPatchDataset(torch.utils.data.Dataset):
    def __init__(self, datalist_file: Path, data_dir: Path):
        """ Load patched dataset.

        Args:
            datalist_file: text file containing the instances name.
            data_dir: directory containing the patched dataset.
        """
        self.data_dir = data_dir

        with open(datalist_file, "r") as f:
            self.datalist = [x.strip() for x in f.readlines()]

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index: int) -> np.ndarray:
        inst_id = self.datalist[index]

        metadata = self.load_metadata(inst_id)
        image_size    = metadata["image_size"]

        window = self.compute_window(image_size)

        return self.load_image_data(index, inst_id, window)

    def compute_window(self, size: tuple[int, int]) -> CropWindow:
        """ Compute the crop window of the image.

        Args:
            size: (H, W) of the full image size.
            variant: identifier for cropping multiple patches from a single instance.
            source_bayer_pattern: Bayer pattern of the output if not None.

        Returns:
            (I, J, H, W) of the crop window.
        """
        I, J = 0, 0
        H, W = size
        H -= I
        W -= J
        H -= H % 2
        W -= W % 2

        return (I, J, H, W)

    def load_image_data(self, index: int, inst_id: str, window: CropWindow) -> np.ndarray:
        """ Get the image data of the instance included in the window. """
        # load image
        rgb = self._load_image_in_window(self._load_rgb_patch, inst_id, window)
        # normalize data
        rgb = utils.convert.dequantize(rgb, augmentation=False)
        return rgb

    def load_metadata(self, inst_id: str) -> dict:
        """ Load metadata of the instance. """
        return utils.io.loadpt(self.data_dir/inst_id/"metadata.pt")

    def _load_raw_patch(self, inst_id: str, i: int, j: int) -> np.ndarray:
        """ Load RAW image patch located at given index in the instance. """
        return utils.io.loadtiff(self.data_dir/inst_id/patch_name("raw", i, j)).squeeze(-1)

    def _load_rgb_patch(self, inst_id: str, i: int, j: int) -> np.ndarray:
        """ Load RGB image patch located at given index in the instance. """
        return utils.io.loadtiff(self.data_dir/inst_id/patch_name("rgb", i, j))

    def _load_image_in_window(self, patch_loader: Callable,
                              inst_id: str, window: CropWindow) -> np.ndarray:
        """ Load partial image in the window. """
        I, J, H, W = window
        
        # Least-size concatenated patches containing the window
        fI = (I // PATCH_SIZE) * PATCH_SIZE
        fJ = (J // PATCH_SIZE) * PATCH_SIZE
        patch = np.concatenate([
            np.concatenate([
                patch_loader(inst_id, i, j)
                for j in range(fJ, J+W, PATCH_SIZE)
            ], axis=1)
            for i in range(fI, I+H, PATCH_SIZE)
        ], axis=0)

        # Crop the patch fit to the window
        rI, rJ = I-fI, J-fJ
        patch = patch[rI:rI+H, rJ:rJ+W]

        return patch


class CenterCropPatchDataset(PatchDataset):
    def __init__(self, datalist_file: Path, data_dir: Path,
                 crop_size: int | tuple[int, int], bayer_pattern: np.ndarray | str | None = None,
                 use_extra: bool = False):
        """ Load center cropped patch from dataset.

        Args:
            datalist_file: text file containing the instances name.
            data_dir: directory containing the patched dataset.
            crop_size: (H, W) of the crop size.
            bayer_pattern: Match the bayer pattern of the output if not None.
        """
        super().__init__(datalist_file=datalist_file, data_dir=data_dir,
                         bayer_pattern=bayer_pattern, use_extra=use_extra)
        self.crop_size = refine_crop_size(crop_size)

    def compute_window(self, size: tuple[int, int], *,
                       source_bayer_pattern: np.ndarray | None = None) -> CropWindow:
        IH, IW = size
        CH, CW = self.crop_size

        assert CH <= IH and CW <= IW, f"crop_size={CH}x{CW} > image_size={IH}x{IW}"

        # center crop
        CI = (IH - CH) // 2
        CJ = (IW - CW) // 2

        # shift bayer pattern to match the output
        if self.bayer_flag is not None:
            assert source_bayer_pattern is not None
            src_flag = utils.camera.BayerPatternFlag.from_numpy(source_bayer_pattern)
            dst_flag = self.bayer_flag

            IFLAG = ((src_flag ^ dst_flag) >> 0) & 0b1
            JFLAG = ((src_flag ^ dst_flag) >> 1) & 0b1
            CI += (CI + IFLAG) % 2
            CJ += (CJ + JFLAG) % 2

            if CI + CH > IH:
                assert CI >= 2, f"Not enough space to fit crop window with perserving bayer pattern"
                CI -= 2
            if CJ + CW > IW:
                assert CJ >= 2, f"Not enough space to fit crop window with perserving bayer pattern"
                CJ -= 2

        return CI, CJ, CH, CW


class CenterCropRGBPatchDataset(RGBPatchDataset):
    def __init__(self, datalist_file: Path, data_dir: Path, crop_size: int | tuple[int, int]):
        """ Load center cropped patch from dataset.

        Args:
            datalist_file: text file containing the instances name.
            data_dir: directory containing the patched dataset.
            crop_size: (H, W) of the crop size.
        """
        super().__init__(datalist_file=datalist_file, data_dir=data_dir)
        self.crop_size = refine_crop_size(crop_size)

    def compute_window(self, size: tuple[int, int]) -> CropWindow:
        IH, IW = size
        CH, CW = self.crop_size

        assert CH <= IH and CW <= IW, f"crop_size={CH}x{CW} > image_size={IH}x{IW}"

        # center crop
        CI = (IH - CH) // 2
        CJ = (IW - CW) // 2

        return CI, CJ, CH, CW


class RandomCropPatchDataset(PatchDataset):
    def __init__(self, datalist_file: Path, data_dir: Path,
                 crop_size: int | tuple[int, int], bayer_pattern: np.ndarray | str | None = None,
                 use_extra: bool = False):
        """ Load random cropped patch from dataset.

        Args:
            datalist_file: text file containing the instances name.
            data_dir: directory containing the patched dataset.
            crop_size: (H, W) of the crop size.
            bayer_pattern: Match the bayer pattern of the output if not None.
        """
        super().__init__(datalist_file=datalist_file, data_dir=data_dir,
                         bayer_pattern=bayer_pattern, use_extra=use_extra)
        self.crop_size = refine_crop_size(crop_size)

    def compute_window(self, size: tuple[int, int], *,
                       source_bayer_pattern: np.ndarray | None = None) -> CropWindow:
        IH, IW = size
        CH, CW = self.crop_size

        assert CH <= IH and CW <= IW, f"crop_size={CH}x{CW} > image_size={IH}x{IW}"

        # random crop
        CI = np.random.randint(0, IH-CH)
        CJ = np.random.randint(0, IW-CW)

        # shift bayer pattern to match the output
        if self.bayer_flag is not None:
            assert source_bayer_pattern is not None
            src_flag = utils.camera.BayerPatternFlag.from_numpy(source_bayer_pattern)
            dst_flag = self.bayer_flag

            IFLAG = ((src_flag ^ dst_flag) >> 0) & 0b1
            JFLAG = ((src_flag ^ dst_flag) >> 1) & 0b1
            CI += (CI + IFLAG) % 2
            CJ += (CJ + JFLAG) % 2

            if CI + CH > IH:
                assert CI >= 2, f"Not enough space to fit crop window with perserving bayer pattern"
                CI -= 2
            if CJ + CW > IW:
                assert CJ >= 2, f"Not enough space to fit crop window with perserving bayer pattern"
                CJ -= 2

        return CI, CJ, CH, CW


class RandomCropRGBPatchDataset(RGBPatchDataset):
    def __init__(self, datalist_file: Path, data_dir: Path, crop_size: int | tuple[int, int]):
        """ Load random cropped patch from dataset.

        Args:
            datalist_file: text file containing the instances name.
            data_dir: directory containing the patched dataset.
            crop_size: (H, W) of the crop size.
        """
        super().__init__(datalist_file=datalist_file, data_dir=data_dir)
        self.crop_size = refine_crop_size(crop_size)

    def compute_window(self, size: tuple[int, int]) -> CropWindow:
        IH, IW = size
        CH, CW = self.crop_size

        assert CH <= IH and CW <= IW, f"crop_size={CH}x{CW} > image_size={IH}x{IW}"

        # random crop
        CI = np.random.randint(0, IH-CH)
        CJ = np.random.randint(0, IW-CW)

        return CI, CJ, CH, CW


class FiveCropPatchDataset(PatchDataset):
    def __init__(self, datalist_file: Path, data_dir: Path,
                 crop_size: int | tuple[int, int], bayer_pattern: np.ndarray | str | None = None,
                 use_extra: bool = False):
        """ Load cropped patch from dataset, which contains four corners and the central crop.

        Args:
            datalist_file: text file containing the instances name.
            data_dir: directory containing the patched dataset.
            crop_size: (H, W) of the crop size.
            bayer_pattern: Match the bayer pattern of the output if not None.
        """
        super().__init__(datalist_file=datalist_file, data_dir=data_dir,
                         bayer_pattern=bayer_pattern, use_extra=use_extra)
        self.crop_size = refine_crop_size(crop_size)

    def __len__(self):
        return len(5 * self.datalist)

    def __getitem__(self, index: int) -> ImageData:
        index, variant = index // 5, index % 5
        inst_id = self.datalist[index]
        
        metadata = self.load_metadata(inst_id)
        image_size    = metadata["image_size"]
        bayer_pattern = metadata["bayer_pattern"]

        window = self.compute_window(image_size, variant, source_bayer_pattern=bayer_pattern)

        return self.load_image_data(index, inst_id, window, metadata)

    def compute_window(self, size: tuple[int, int], variant: int, *,
                       source_bayer_pattern: np.ndarray | None = None) -> CropWindow:
        IH, IW = size
        CH, CW = self.crop_size

        assert CH <= IH and CW <= IW, f"crop_size={CH}x{CW} > image_size={IH}x{IW}"

        match variant:
            case 0:  # center crop
                CI = (IH - CH) // 2
                CJ = (IW - CW) // 2
            case 1:  # top-left crop
                CI = 0
                CJ = 0
            case 2:  # top-right crop
                CI = 0
                CJ = IW - CW
            case 3:  # bottom-left crop
                CI = IH - CH
                CJ = 0
            case 4:  # bottom-right crop
                CI = IH - CH
                CJ = IW - CW
            case _: raise ValueError(f"Unknown variant {variant}")

        # shift bayer pattern to match the output
        if self.bayer_flag is not None:
            assert source_bayer_pattern is not None
            src_flag = utils.camera.BayerPatternFlag.from_numpy(source_bayer_pattern)
            dst_flag = self.bayer_flag

            IFLAG = ((src_flag ^ dst_flag) >> 0) & 0b1
            JFLAG = ((src_flag ^ dst_flag) >> 1) & 0b1
            CI += (CI + IFLAG) % 2
            CJ += (CJ + JFLAG) % 2

            if CI + CH > IH:
                assert CI >= 2, f"Not enough space to fit crop window with perserving bayer pattern"
                CI -= 2
            if CJ + CW > IW:
                assert CJ >= 2, f"Not enough space to fit crop window with perserving bayer pattern"
                CJ -= 2

        return CI, CJ, CH, CW


class FiveCropRGBPatchDataset(RGBPatchDataset):
    def __init__(self, datalist_file: Path, data_dir: Path, crop_size: int | tuple[int, int]):
        """ Load cropped patch from dataset, which contains four corners and the central crop.

        Args:
            datalist_file: text file containing the instances name.
            data_dir: directory containing the patched dataset.
            crop_size: (H, W) of the crop size.
        """
        super().__init__(datalist_file=datalist_file, data_dir=data_dir)
        self.crop_size = refine_crop_size(crop_size)

    def __len__(self):
        return len(5 * self.datalist)

    def __getitem__(self, index: int) -> np.ndarray:
        index, variant = index // 5, index % 5
        inst_id = self.datalist[index]

        metadata = self.load_metadata(inst_id)
        image_size    = metadata["image_size"]

        window = self.compute_window(image_size, variant)

        return self.load_image_data(index, inst_id, window)

    def compute_window(self, size: tuple[int, int], variant: int) -> CropWindow:
        IH, IW = size
        CH, CW = self.crop_size

        assert CH <= IH and CW <= IW, f"crop_size={CH}x{CW} > image_size={IH}x{IW}"

        match variant:
            case 0:  # center crop
                CI = (IH - CH) // 2
                CJ = (IW - CW) // 2
            case 1:  # top-left crop
                CI = 0
                CJ = 0
            case 2:  # top-right crop
                CI = 0
                CJ = IW - CW
            case 3:  # bottom-left crop
                CI = IH - CH
                CJ = 0
            case 4:  # bottom-right crop
                CI = IH - CH
                CJ = IW - CW
            case _: raise ValueError(f"Unknown variant {variant}")

        return CI, CJ, CH, CW


class ThirteenCropPatchDataset(PatchDataset):
    def __init__(self, datalist_file: Path, data_dir: Path,
                 crop_size: int | tuple[int, int], bayer_pattern: np.ndarray | str | None = None,
                 use_extra: bool = False):
        """ Load cropped patch from dataset, which contains five-crops from each quarters.

        Args:
            datalist_file: text file containing the instances name.
            data_dir: directory containing the patched dataset.
            crop_size: (H, W) of the crop size.
            bayer_pattern: Match the bayer pattern of the output if not None.
        """
        super().__init__(datalist_file=datalist_file, data_dir=data_dir,
                         bayer_pattern=bayer_pattern, use_extra=use_extra)
        self.crop_size = refine_crop_size(crop_size)

    def __len__(self):
        return len(13 * self.datalist)

    def __getitem__(self, index: int) -> ImageData:
        index, variant = index // 13, index % 13
        inst_id = self.datalist[index]

        metadata = self.load_metadata(inst_id)
        image_size    = metadata["image_size"]
        bayer_pattern = metadata["bayer_pattern"]

        window = self.compute_window(image_size, variant, source_bayer_pattern=bayer_pattern)

        return self.load_image_data(index, inst_id, window, metadata)

    def compute_window(self, size: tuple[int, int], variant: int, *,
                       source_bayer_pattern: np.ndarray | None = None) -> CropWindow:
        IH, IW = size
        CH, CW = self.crop_size

        assert CH <= IH and CW <= IW, f"crop_size={CH}x{CW} > image_size={IH}x{IW}"

        match variant:
            case 0:  # center crop
                CI = (IH - CH) // 2
                CJ = (IW - CW) // 2
            case 1:  # top-left crop
                CI = 0
                CJ = 0
            case 2:  # top-right crop
                CI = 0
                CJ = IW - CW
            case 3:  # bottom-left crop
                CI = IH - CH
                CJ = 0
            case 4:  # bottom-right crop
                CI = IH - CH
                CJ = IW - CW
            case 5:  # top-middle crop
                CI = 0
                CJ = (IW - CW) // 2
            case 6:  # bottom-middle crop
                CI = IH - CH
                CJ = (IW - CW) // 2
            case 7:  # left-middle crop
                CI = (IH - CH) // 2
                CJ = 0
            case 8:  # right-middle crop
                CI = (IH - CH) // 2
                CJ = IW - CW
            case 9:  # center of top-left-quarter crop
                CI = IH // 4 - CH // 2
                CJ = IW // 4 - CW // 2
            case 10:  # center of top-right-quarter crop
                CI = IH // 4 - CH // 2
                CJ = 3 * IW // 4 - CW // 2
            case 11:  # center of bottom-left-quarter crop
                CI = 3 * IH // 4 - CH // 2
                CJ = IW // 4 - CW // 2
            case 12:  # center of bottom-right-quarter crop
                CI = 3 * IH // 4 - CH // 2
                CJ = 3 * IW // 4 - CW // 2
            case _: raise ValueError(f"Unknown variant {variant}")

        # shift bayer pattern to match the output
        if self.bayer_flag is not None:
            assert source_bayer_pattern is not None
            src_flag = utils.camera.BayerPatternFlag.from_numpy(source_bayer_pattern)
            dst_flag = self.bayer_flag

            IFLAG = ((src_flag ^ dst_flag) >> 0) & 0b1
            JFLAG = ((src_flag ^ dst_flag) >> 1) & 0b1
            CI += (CI + IFLAG) % 2
            CJ += (CJ + JFLAG) % 2

            if CI + CH > IH:
                assert CI >= 2, f"Not enough space to fit crop window with perserving bayer pattern"
                CI -= 2
            if CJ + CW > IW:
                assert CJ >= 2, f"Not enough space to fit crop window with perserving bayer pattern"
                CJ -= 2

        return CI, CJ, CH, CW


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets: torch.utils.data.Dataset):
        """ Concatenate multiple datasets. """
        super().__init__()
        self.datasets = datasets

    def __len__(self) -> int:
        return sum([len(d) for d in self.datasets])  # type: ignore

    def __getitem__(self, index: int):
        for dataset in self.datasets:
            dataset_length = len(dataset)  # type: ignore
            if index < dataset_length:
                return dataset[index]
            index -= dataset_length
        raise IndexError("Index out of range")


class RandomSampler(torch.utils.data.Sampler[int]):
    def __init__(self, dataset: torch.utils.data.Dataset, num_samples: int, generator=None) -> None:
        """ Sample elements randomly.

        Args:
            dataset: Dataset to sample from.
            num_samples: Number of samples to draw.
            generator: Generator used in sampling.
        """
        assert isinstance(num_samples, int) and num_samples > 0, \
            f"`num_samples` should be a positive integer value, got {num_samples}"

        self.dataset = dataset
        self.num_samples = num_samples
        self._psize = 64

        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = generator

    def __iter__(self) -> Iterator[int]:
        dataset_len = len(self.dataset)  # type: ignore
        for _ in range(self.num_samples // self._psize):
            yield from torch.randint(
                low=0, high=dataset_len, size=(self._psize,),
                dtype=torch.int64, generator=self.generator).tolist()
        yield from torch.randint(
            low=0, high=dataset_len, size=(self.num_samples % self._psize,),
            dtype=torch.int64, generator=self.generator).tolist()

    def __len__(self) -> int:
        return self.num_samples


class EqualizedRandomSampler(torch.utils.data.Sampler[int]):
    def __init__(self, datasets: list[torch.utils.data.Dataset], num_samples: int, generator=None) -> None:
        """ Sample elements randomly, with equalized probability for each datasets.

        Args:
            datasets: List of datasets to sample from.
            num_samples: Number of samples to draw for each dataset.
            generator: Generator used in sampling.
        """
        assert isinstance(num_samples, int) and num_samples > 0, \
            f"`num_samples` should be a positive integer value, got {num_samples}"

        self.datasets = datasets
        self.num_samples = num_samples
        self._psize = 64

        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = generator

    def __iter__(self) -> Iterator[int]:
        indices = []
        offset = 0
        for dataset in self.datasets:
            dataset_len = len(dataset)  # type: ignore
            indices.append(offset + torch.randint(0, dataset_len, (self.num_samples,),
                                                  dtype=torch.int64, generator=self.generator))
            offset += dataset_len
        indices = torch.cat(indices)
        indices = indices[torch.randperm(indices.size(0), generator=self.generator)]
        yield from indices.tolist()

    def __len__(self) -> int:
        return self.num_samples * len(self.datasets)


class DistributedRandomSampler(torch.utils.data.Sampler[int]):
    def __init__(self, dataset: torch.utils.data.Dataset, num_samples: int,
                 num_replicas: int | None = None, rank: int | None = None,
                 seed: int = 0):
        """ Sample elements randomly with distributed learning.

        Args:
            dataset: Dataset to sample from.
            num_samples: Number of samples to draw.
            num_replicas: Number of processes participating in distributed training.
                By default, :attr:`world_size` is retrieved from the current distributed group.
            rank: Rank of the current process within :attr:`num_replicas`.
                By default, :attr:`rank` is retrieved from the current distributed group.
            seed: Random seed used to shuffle the sampler.
                This number should be identical across all processes in the distributed group.
        """
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        assert isinstance(rank, int) and isinstance(num_replicas, int)
        assert isinstance(num_samples, int) and num_samples > 0, \
            f"`num_samples` should be a positive integer value, got {num_samples}"

        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank={rank}, should be in the interval [0, {num_replicas-1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed

        self.num_samples = math.ceil(num_samples / self.num_replicas)
        self._psize = 64

    def __iter__(self) -> Iterator[int]:
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        dataset_len = len(self.dataset)  # type: ignore
        indices = torch.randint(0, dataset_len, (self.num_samples * self.num_replicas,),
                                dtype=torch.int64, generator=g)
        yield from indices.tolist()[self.rank::self.num_replicas]

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int):
        """ Sets the epoch for this sampler. This ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this sampler
        will yield the same ordering. """
        self.epoch = epoch


class DistributedEqualizedRandomSampler(torch.utils.data.Sampler[int]):
    def __init__(self, datasets: list[torch.utils.data.Dataset], num_samples: int,
                 num_replicas: int | None = None, rank: int | None = None,
                 seed: int = 0):
        """ Sample elements randomly with distributed learning, with equalized probability for
        each datasets.

        Args:
            datasets: List of datasets to sample from.
            num_samples: Number of samples to draw for each dataset.
            num_replicas: Number of processes participating in distributed training.
                By default, :attr:`world_size` is retrieved from the current distributed group.
            rank: Rank of the current process within :attr:`num_replicas`.
                By default, :attr:`rank` is retrieved from the current distributed group.
            seed: Random seed used to shuffle the sampler.
                This number should be identical across all processes in the distributed group.
        """
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        assert isinstance(rank, int) and isinstance(num_replicas, int)
        assert isinstance(num_samples, int) and num_samples > 0, \
            f"`num_samples` should be a positive integer value, got {num_samples}"

        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank={rank}, should be in the interval [0, {num_replicas-1}]")

        self.datasets = datasets
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed

        self.num_samples = math.ceil(num_samples / self.num_replicas)
        self._psize = 64

    def __iter__(self) -> Iterator[int]:
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = []
        offset = 0
        for dataset in self.datasets:
            dataset_len = len(dataset)  # type: ignore
            indices.append(offset + torch.randint(0, dataset_len, (self.num_samples * self.num_replicas,),
                                                  dtype=torch.int64, generator=g))
            offset += dataset_len
        indices = torch.cat(indices)
        indices = indices[torch.randperm(indices.size(0), generator=g)]

        yield from indices.tolist()[self.rank::self.num_replicas]

    def __len__(self) -> int:
        return self.num_samples * len(self.datasets)

    def set_epoch(self, epoch: int):
        """ Sets the epoch for this sampler. This ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this sampler
        will yield the same ordering. """
        self.epoch = epoch


class SelectedIndexSampler(torch.utils.data.Sampler[int]):
    def __init__(self, dataset: torch.utils.data.Dataset, indices: list[int]) -> None:
        """ Sample elements from a dataset with selected indices.

        Args:
            dataset: Dataset to sample from.
            indices: List of indices to sample from.
        """
        self.dataset = dataset
        self.indices = sorted(set(indices))

    def __iter__(self) -> Iterator[int]:
        yield from self.indices

    def __len__(self) -> int:
        return len(self.indices)


class DistributedSelectedIndexSampler(torch.utils.data.Sampler[int]):
    def __init__(self, dataset: torch.utils.data.Dataset, indices: list[int],
                 num_replicas: int | None = None, rank: int | None = None):
        """ Sample elements from a dataset with selected indices, with distributed learning.

        Args:
            datasets: List of datasets to sample from.
            indices: List of indices to sample from.
            num_replicas: Number of processes participating in distributed training.
                By default, :attr:`world_size` is retrieved from the current distributed group.
            rank: Rank of the current process within :attr:`num_replicas`.
                By default, :attr:`rank` is retrieved from the current distributed group.
        """
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        assert isinstance(rank, int) and isinstance(num_replicas, int)

        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank={rank}, should be in the interval [0, {num_replicas-1}]")

        self.dataset = dataset
        self.indices = sorted(set(indices))
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        yield from self.indices[self.rank::self.num_replicas]

    def __len__(self) -> int:
        return len(self.indices[self.rank::self.num_replicas])

    def set_epoch(self, epoch: int):
        """ Sets the epoch for this sampler. This ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this sampler
        will yield the same ordering. """
        self.epoch = epoch


def random_noise_levels():
    log_min_shot_noise = np.log(0.0001)
    log_max_shot_noise = np.log(0.012)
    log_shot_noise = np.random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = np.exp(log_shot_noise)

    def line(x): return 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + np.random.normal(0, 0.26)
    read_noise = np.exp(log_read_noise)

    return float(shot_noise), float(read_noise)


def add_noise(img: torch.Tensor, shot_noise=0.01, read_noise=0.0005):
    variance = img * shot_noise + read_noise
    noise = torch.randn_like(img) * torch.sqrt(variance)
    return img + noise
