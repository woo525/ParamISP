from typing import Literal, Callable
import functools
import sys
import json
import yaml
import csv
from time import sleep

import torch
import numpy as np
import numpy.typing as npt
import scipy.io
import PIL.Image as pil
import rawpy
import tifffile
import exifread

from .path import purify
from .convert import as_numpy_image, as_torch_tensor, as_u8image, quantize, dequantize, is_integer_dtype
from .plot import rawgrid
from ._typing.rawpy import RawPy


def _loader(fn):
    @functools.wraps(fn)
    def wrapper(*path, **kwargs):
        try:
            return fn(*path, **kwargs)
        except Exception as e:
            print(f"Failed to load {purify(*path)}:", e, file=sys.stderr)
            raise e
    return wrapper


def _saver(fn):
    @functools.wraps(fn)
    def wrapper(img, *path, **kwargs):
        try:
            return fn(img, *path, **kwargs)
        except Exception as e:
            print(f"Failed to save {purify(*path)}:", e, file=sys.stderr)
            raise e
    return wrapper


def _debatch(img) -> torch.Tensor:
    img = as_torch_tensor(img)

    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    elif img.ndim == 5 and img.shape[0] == 1 and img.shape[1] == 1:
        img = img[0, 0]

    if img.ndim == 4:
        img = rawgrid([img], padding=0)
    elif img.ndim == 5:
        img = rawgrid(img, padding=0)

    return img


@_loader
def loadimg(*path) -> npt.NDArray[np.uint8 | np.uint16]:
    """ Load an image from a path.
    The return array shape is `H W` for grayscale images and `H W C` for RGB or RGBA images.
    See `PIL.Image.open` for more details. """
    path = purify(*path)
    for i in range(3):
        try:
            with pil.open(path) as im:
                return np.asarray(im)
        except OSError:
            sleep(i + 1)
    with pil.open(path) as im:
        return np.asarray(im)


@_saver
def saveimg(img: npt.ArrayLike, *path, **kwargs):
    """ Save an image to a path.
    See `PIL.Image.save` for more details. """
    path = purify(*path)
    img = _debatch(img)
    img = as_u8image(img)
    img = as_numpy_image(img)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    pil.fromarray(img).save(path, **kwargs)


@_loader
def loadraw(*path) -> RawPy:
    """ Load a raw image from a path.
    See `rawpy.imread` for more details. """
    path = purify(*path)
    return rawpy.imread(path)


@_loader
def loadexif(*path) -> dict[str, str]:
    """ Load EXIF data from a path.
    See `exifread.process_file` for more details. """
    path = purify(*path)
    with open(path, "rb") as f:
        return exifread.process_file(f)


@_loader
def loadtiff(*path) -> npt.NDArray[np.uint16 | np.float32]:
    """ Load a TIFF image from a path.
    See `tifffile.imread` for more details. """
    path = purify(*path)
    return tifffile.imread(path)


@_saver
def savetiff(img: npt.ArrayLike, *path, u16: bool = False, **kwargs):
    """ Save a TIFF image to a path.
    See `tifffile.imsave` for more details. """
    path = purify(*path)
    img = _debatch(img)

    if u16 and img.dtype not in [np.uint16, np.int16]:
        if is_integer_dtype(img.dtype):  # type: ignore
            img = dequantize(img, augmentation=False)
        img = as_numpy_image(img)
        img = quantize(img, out_dtype=np.uint16)

    tifffile.imwrite(path, img, **kwargs)


@_loader
def loadnpy(*path, mode: Literal["r+", "r", "w+", "c"] | None = "r") -> np.ndarray:
    """ Load a numpy array from a path.
    See `np.load` for more details. """
    path = purify(*path)
    return np.load(path, mmap_mode=mode)


@_saver
def savenpy(matrix: npt.ArrayLike, *path):
    """ Save a numpy array to a path.
    See `np.save` for more details. """
    path = purify(*path)
    np.save(path, matrix)


@_saver
def savenpz(matrices: dict[str, npt.ArrayLike], *path, compress: bool = True):
    """ Save a numpy array to a path.
    See `np.savez` for more details. """
    path = purify(*path)
    (np.savez_compressed if compress else np.savez)(path, **matrices)


@_loader
def loadnptxt(*path, dtype: npt.DTypeLike = float, delimiter=" ", comments="#") -> np.ndarray:
    """ Load a numpy array from a text file.
    See `np.loadtxt` for more details. """
    path = purify(*path)
    return np.loadtxt(path, dtype=dtype, delimiter=delimiter, comments=comments)


@_saver
def savenptxt(matrix: npt.ArrayLike, *path, fmt="%.18e", delimiter=" ", newline="\n", header="", footer="", comments="#"):
    """ Save a numpy array to a text file.
    See `np.savetxt` for more details. """
    path = purify(*path)
    np.savetxt(path, matrix, fmt=fmt, delimiter=delimiter, newline=newline,
               header=header, footer=footer, comments=comments)


@_loader
def loadmat(*path) -> dict[str, npt.NDArray]:
    """ Load a MATLAB array from a path.
    See `scipy.io.loadmat` for more details. """
    path = purify(*path)
    return scipy.io.loadmat(path)


@_saver
def savemat(matrices: npt.ArrayLike, *path):
    """ Save a MATLAB array to a path.
    See `scipy.io.savemat` for more details. """
    path = purify(*path)
    scipy.io.savemat(path, matrices)


@_loader
def loadpt(*path, map_location=None) -> dict:
    """ Load a PyTorch tensor from a path.
    See `torch.load` for more details. """
    path = purify(*path)
    return torch.load(path, map_location=map_location)


@_saver
def savept(tensor: torch.Tensor | dict, *path):
    """ Save a PyTorch tensor to a path.
    See `torch.save` for more details. """
    path = purify(*path)
    torch.save(tensor, path)


@_loader
def loadjson(*path):
    """ Load a JSON object from a path.
    See `json.load` for more details. """
    path = purify(*path)
    with open(path, "r") as f:
        return json.load(f)


@_saver
def savejson(obj, *path):
    """ Save a JSON object to a path.
    See `json.dump` for more details. """
    path = purify(*path)
    with open(path, "w") as f:
        json.dump(obj, f)


@_loader
def loadyaml(*path):
    """ Load a YAML object from a path.
    See `yaml.safe_load` for more details. """
    path = purify(*path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


@_saver
def saveyaml(obj, *path):
    """ Save a YAML object to a path.
    See `yaml.dump` for more details. """
    path = purify(*path)
    with open(path, "w") as f:
        yaml.dump(obj, f)


class csvreader:
    """ Reader to load a CSV file from a path.
        See `csv.reader` for more details. """

    def __init__(self, *path, delimiter=","):
        self.path = purify(*path)
        self.delimiter = delimiter

    def __enter__(self):
        self.f = open(self.path, "r", newline="")
        self.r = csv.reader(self.f, delimiter=self.delimiter, quotechar='"', escapechar="\\")
        return self.r

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()


class csvwriter:
    """ Writer to save a CSV file to a path.
        See `csv.writer` for more details. """

    def __init__(self, *path, delimiter=","):
        self.path = purify(*path)
        self.delimiter = delimiter

    def __enter__(self):
        self.f = open(self.path, "w", newline="")
        self.w = csv.writer(self.f, delimiter=self.delimiter, quotechar='"', escapechar="\\")
        return self.w

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()
