from typing import TypeVar, TypeGuard
from warnings import warn

import torch
import numpy as np
import numpy.typing as nptype
from toolz import isiterable as _isiterable


TTensor = TypeVar("TTensor", torch.Tensor, np.ndarray)


def isiterable(x) -> TypeGuard[list]:
    """ Check if the object is iterable. """
    return _isiterable(x)


def as_torch_dtype(dtype: torch.dtype | type | str) -> torch.dtype:
    """ Convert dtype to torch.dtype. """
    if dtype in ["bool", bool, np.bool8, torch.bool]:
        return torch.bool
    if dtype in ["uint8", "u8", np.uint8, torch.uint8]:
        return torch.uint8
    if dtype in ["int8", "i8", np.int8, torch.int8]:
        return torch.int8
    if dtype in ["int16", "i16", np.int16, torch.int16]:
        return torch.int16
    if dtype in ["int32", "i32", np.int32, torch.int32]:
        return torch.int32
    if dtype in ["int64", "i64", int, np.int64, torch.int64]:
        return torch.int64
    if dtype in ["float16", "f16", np.float16, torch.float16]:
        return torch.float16
    if dtype in ["float32", "f32", np.float32, torch.float32]:
        return torch.float32
    if dtype in ["float64", "f64", float, np.float64, torch.float64]:
        return torch.float64
    if dtype in ["complex64", "c64", np.complex64, torch.complex64]:
        return torch.complex64
    if dtype in ["complex128", "c128", np.complex128, torch.complex128]:
        return torch.complex128
    raise ValueError(f"Unknown dtype: {dtype}")


def as_np_dtype(dtype: torch.dtype | type | str) -> type:
    """ Convert dtype to numpy.dtype. """
    if dtype in ["bool", bool, np.bool8, torch.bool]:
        return np.bool8
    if dtype in ["uint8", "u8", np.uint8, torch.uint8]:
        return np.uint8
    if dtype in ["uint16", "u16", np.uint16]:
        return np.uint16
    if dtype in ["uint32", "u32", np.uint32]:
        return np.uint32
    if dtype in ["uint64", "u64", np.uint64]:
        return np.uint64
    if dtype in ["int8", "i8", np.int8, torch.int8]:
        return np.int8
    if dtype in ["int16", "i16", np.int16, torch.int16]:
        return np.int16
    if dtype in ["int32", "i32", np.int32, torch.int32]:
        return np.int32
    if dtype in ["int64", "i64", int, np.int64, torch.int64]:
        return np.int64
    if dtype in ["float16", "f16", np.float16, torch.float16]:
        return np.float16
    if dtype in ["float32", "f32", np.float32, torch.float32]:
        return np.float32
    if dtype in ["float64", "f64", float, np.float64, torch.float64]:
        return np.float64
    if dtype in ["float128", "f128", np.float128]:
        return np.float128
    if dtype in ["complex64", "c64", np.complex64, torch.complex64]:
        return np.complex64
    if dtype in ["complex128", "c128", np.complex128, torch.complex128]:
        return np.complex128
    if dtype in ["complex256", "c256", np.complex256]:
        return np.complex256
    raise ValueError(f"Unknown dtype: {dtype}")


def is_floating_torch_dtype(dtype: torch.dtype) -> bool:
    """ Check if the dtype is a floating point torch.dtype. """
    return dtype in [torch.float16, torch.float32, torch.float64]


def is_integer_torch_dtype(dtype: torch.dtype) -> bool:
    """ Check if the dtype is an integer torch.dtype. """
    return dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]


def is_complex_torch_dtype(dtype: torch.dtype) -> bool:
    """ Check if the dtype is a complex torch.dtype. """
    return dtype in [torch.complex64, torch.complex128]


def is_floating_dtype(dtype: torch.dtype | type | str) -> bool:
    """ Check if the dtype is a floating point dtype. """
    return np.issubdtype(as_np_dtype(dtype), np.floating)


def is_integer_dtype(dtype: torch.dtype | type | str) -> bool:
    """ Check if the dtype is an integer dtype. """
    return np.issubdtype(as_np_dtype(dtype), np.integer)


def is_complex_dtype(dtype: torch.dtype | type | str) -> bool:
    """ Check if the dtype is a complex dtype. """
    return np.issubdtype(as_np_dtype(dtype), np.complexfloating)


def is_incompatible_dtype(dtype: np.dtype) -> bool:
    """ Check if the numpy.dtype is incompatible with torch.dtype. """
    return dtype in [np.uint16, np.uint32, np.uint64, np.float128, np.complex256]


def as_signed(x: TTensor) -> TTensor:
    """ Convert the unsigned integer tensor to be signed. """
    if isinstance(x, torch.Tensor):
        if x.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            return x
        if x.dtype == torch.uint8:
            return x.to(torch.int8 if x.max() < 2**7 else torch.int16)
    else:
        if x.dtype in [np.int8, np.int16, np.int32, np.int64]:
            return x
        if x.dtype == np.uint8:
            return x.astype(np.int8 if x.max() < 2**7 else np.int16)
        elif x.dtype == np.uint16:
            return x.astype(np.int16 if x.max() < 2**15 else np.int32)
        elif x.dtype == np.uint32:
            return x.astype(np.int32 if x.max() < 2**31 else np.int64)
        elif x.dtype == np.uint64:
            if x.max() >= 2**63:
                raise ValueError("Input tensor value is too large to be converted to signed.")
            return x.astype(np.int64)

    raise ValueError(f"Unsupported dtype: {x.dtype}")


def as_unsigned(x: TTensor) -> TTensor:
    """ Convert the signed integer tensor to be unsigned. """
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.int8:
            return x.to(torch.uint8)
    else:
        if x.dtype == np.int8:
            return x.astype(np.uint8)
        elif x.dtype == np.int16:
            return x.astype(np.uint16)
        elif x.dtype == np.int32:
            return x.astype(np.uint32)
        elif x.dtype == np.int64:
            return x.astype(np.uint64)

    raise ValueError(f"Unsupported dtype: {x.dtype}")


def as_type(x: TTensor, dtype: type) -> TTensor:
    """ Convert the tensor data type. """
    if isinstance(x, torch.Tensor):
        return x.to(as_torch_dtype(dtype))
    else:
        return x.astype(as_np_dtype(dtype))


def numpy_to_torch(x: np.ndarray, strict: bool | None = None) -> torch.Tensor:
    """ Convert the numpy tensor to be torch tensor with compatible dtype.

    Args:
        x: numpy array
        strict: if True, raise error if the dtype is incompatible with torch.

    Returns:
        torch.Tensor: The tensor with compatible dtype.
    """
    dtype = x.dtype
    if isinstance(x, np.ndarray):
        if dtype == np.uint16:
            if x.max() < 2**15:
                x = x.astype(np.int16)
            else:
                if strict:
                    raise ValueError("Cannot convert uint16 array with values >= 2**15 to tensor."
                                     "Set strict=True to ignore this error.")
                if strict is None:
                    warn("uint16 array with values >= 2**15 will be converted to torch.int32. "
                         "Set strict=True to ignore this warnings.")
                x = x.astype(np.int32)
        elif dtype == np.uint32:
            if x.max() < 2**31:
                x = x.astype(np.int32)
            else:
                if strict:
                    raise ValueError("Cannot convert uint32 array with values >= 2**31 to tensor."
                                     "Set strict=True to ignore this error.")
                if strict is None:
                    warn("uint32 array with values >= 2**31 will be converted to torch.int64. "
                         "Set strict=True to avoid warnings.")
                x = x.astype(np.int64)
        elif dtype == np.uint64:
            if x.max() >= 2**63:
                raise ValueError("Cannot convert uint64 array with values >= 2**63 to tensor")
            x = x.astype(np.int64)

    return torch.as_tensor(x)


def torch_to_numpy(x: torch.Tensor) -> np.ndarray:
    """ Convert the torch tensor to be numpy tensor. """
    return x.detach().cpu().numpy()


def _unsqueeze(x: TTensor, dim: int) -> TTensor:
    """ Add a new dimension to the tensor. """
    if isinstance(x, torch.Tensor):
        return x.unsqueeze(dim)
    else:
        return np.expand_dims(x, dim)


def as_tensor_shape(x: TTensor, multichannel: bool = True, batch: bool = False) -> TTensor:
    """ Reshape the tensor to be :math:`(C, H, W)` form.

    Args:
        x: input tensor.
        multichannel: if False, output will be in :math:`(H, W)` form if :math:`C = 1`.
        batch: if True, force output to be :math:`(N, C, H, W)` form.
    """
    assert not (batch and not multichannel), "Cannot reshape batch tensor to `(N, H, W)` form."

    while x.ndim < 2:
        x = _unsqueeze(x, 0)

    if x.ndim == 2 and multichannel:
        x = _unsqueeze(x, 0)
    if x.ndim >= 3 and x.shape[-3] not in [1, 3] and x.shape[-1] in [1, 3]:
        x = x.swapaxes(-1, -2).swapaxes(-2, -3)
    if x.ndim == 3 and x.shape[-3] == 1 and not multichannel:
        x = x.squeeze(-3)
    if x.ndim == 3 and batch:
        x = _unsqueeze(x, 0)

    return x


def as_image_shape(x: TTensor, multichannel: bool = False, batch: bool = False) -> TTensor:
    """ Reshape the tensor to be :math:`(H, W, C)` form.

    Args:
        x: input numpy array.
        multichannel: if False, output will be in :math:`(H, W)` form if :math:`C = 1`.
        batch: if True, force output to be :math:`(N, H, W, C)` form.
    """
    assert not (batch and not multichannel), "Cannot reshape batch numpy array to `(N, H, W)` form."

    while x.ndim < 2:
        x = _unsqueeze(x, 0)

    if x.ndim == 2 and multichannel:
        x = _unsqueeze(x, -1)
    if x.ndim >= 3 and x.shape[-1] not in [1, 3] and x.shape[-3] in [1, 3]:
        x = x.swapaxes(-3, -2).swapaxes(-2, -1)
    if x.ndim == 3 and x.shape[-1] == 1 and not multichannel:
        x = x.squeeze(-1)
    if x.ndim == 3 and batch:
        x = _unsqueeze(x, 0)

    return x


def as_torch_tensor(x: torch.Tensor | nptype.ArrayLike,
                    multichannel: bool = True, batch: bool = False) -> torch.Tensor:
    """ Convert arraylike to torch tensor.

    Args:
        x: input image.
        multichannel: if False, output will be in :math:`(H, W)` form if :math:`C = 1`.
        batch: if True, force output to be :math:`(N, C, H, W)` form.
    """
    if not isinstance(x, torch.Tensor):
        x = numpy_to_torch(np.asarray(x))

    x = as_tensor_shape(x, multichannel=multichannel, batch=batch)

    return x


def as_numpy_image(x: torch.Tensor | nptype.ArrayLike,
                   multichannel: bool = True, batch: bool = False) -> np.ndarray:
    """ Convert arraylike to numpy array.

    Args:
        x: input image.
        multichannel: if False, output will be in :math:`(H, W)` form if :math:`C = 1`.
        batch: if True, force output to be :math:`(N, H, W, C)` form.
    """
    if isinstance(x, torch.Tensor):
        x = torch_to_numpy(x)
    else:
        x = np.asarray(x)

    x = as_image_shape(x, multichannel=multichannel, batch=batch)

    return x


def dequantize(
    x: TTensor, *,
    in_range: tuple[int | list[int], int] | None = None,
    out_dtype: type = np.float32,
    bayer_pattern: TTensor | None = None,
    augmentation: bool = False,
) -> TTensor:
    """ Dequantize the tensor from integer type to floating type.

    Args:
        x: Input tensor.
        in_range: Range of the input tensor. If None, it will be inferred from the dtype of `x`.
        out_dtype: Output tensor dtype.
        bayer_pattern: Bayer pattern of the input tensor, required if x is bayer image and in_range is vector.
        augmentation: If True, the output will be augmented with random noise.
    """
    assert is_integer_dtype(x.dtype), "Input image must be integer type."
    assert is_floating_dtype(out_dtype), "Output data type must be floating type."

    if in_range is None:
        in_range_min = 0
        in_range_max: int = np.iinfo(as_np_dtype(x.dtype)).max
    else:
        in_range_min, in_range_max = in_range
        assert not isiterable(in_range_max), "vector type in_range.max is not supported."

    mid_dtype = np.float64 if out_dtype in [torch.float64, np.float64, np.float128] else np.float32
    x = as_type(x, mid_dtype)

    if augmentation:
        # add gaussian noise
        gaussian_noise = torch.randn_like(x) if isinstance(x, torch.Tensor) else np.random.randn(*x.shape)
        x += 0.4 * gaussian_noise
        # or, add uniform noise
        # uniform_noise = torch.rand_like(x) if isinstance(x, torch.Tensor) else np.random.rand(*x.shape)
        # x += uniform_noise - 0.5

    if isiterable(in_range_min) and len(in_range_min) > 1:
        if bayer_pattern is not None:
            assert x.shape[-1] != 1, "Input tensor must be :math:`(..., H, W)` shape if bayer pattern is specified."

            for i, j in np.ndindex(2, 2):
                c = bayer_pattern[i, j]
                x[..., i::2, j::2] = (x[..., i::2, j::2] - in_range_min[c]) / (in_range_max - in_range_min[c])
        else:
            channel_axis = -3 if isinstance(x, torch.Tensor) else -1
            assert x.shape[channel_axis] == len(in_range_min), \
                "in_range.min must have the same length as the number of channels"

            x = x.swapaxes(channel_axis, -1)
            x = (x - in_range_min) / (in_range_max - in_range_min)  # type: ignore
            x = x.swapaxes(-1, channel_axis)
    else:
        x = (x - in_range_min) / (in_range_max - in_range_min)  # type: ignore

    return as_type(x, out_dtype)


def quantize(
    x: TTensor, *,
    out_range: tuple[int, int] | None = None,
    out_dtype: type = np.uint8,
) -> TTensor:
    """ Quantize the tensor from floating type to integer type.

    Args:
        x: Input tensor.
        out_range: Range of the output tensor. If None, it will be inferred from `out_dtype`.
        out_dtype: Output tensor dtype.
    """
    assert is_floating_dtype(x.dtype), "Input image must be floating type."
    assert is_integer_dtype(out_dtype), "Output data type must be integer type."

    out_dtype = as_np_dtype(out_dtype)

    if out_range is None:
        out_range_min = 0
        out_range_max: int = np.iinfo(out_dtype).max
    else:
        out_range_min, out_range_max = out_range

    if x.dtype not in [torch.float32, torch.float64, np.float32, np.float64, np.float128]:
        x = as_type(x, np.float32)

    x = x * (out_range_max - out_range_min) + out_range_min
    x = x.clip(np.iinfo(out_dtype).min, np.iinfo(out_dtype).max).round()

    return as_type(x, out_dtype)


def as_f32image(x: TTensor, *, in_range: tuple[int, int] | None = None, augmentation: bool = False) -> TTensor:
    """ Convert tensor to [0, 1] float32 tensor. See `dequantize` for more details. """
    if is_floating_dtype(x.dtype):
        return as_type(x, np.float32)
    return dequantize(x, in_range=in_range, out_dtype=np.float32, augmentation=augmentation)


def as_u8image(x: TTensor, *, in_range: tuple[int, int] | None = None) -> TTensor:
    """ Convert tensor to [0, 255] uint8 tensor. See `quantize` for more details. """
    if is_integer_dtype(x.dtype) and x.dtype not in [torch.uint8, np.uint8]:
        return quantize(as_f32image(x, in_range=in_range), out_dtype=np.uint8)
    return quantize(x, out_dtype=np.uint8)
