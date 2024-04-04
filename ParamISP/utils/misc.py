from warnings import warn

import torch
import numpy as np


def assert_warn(condition: bool, message: str | Warning, category=None, stacklevel: int = 2, source=None):
    """ Assert condition and raise a warning. See `warnings.warn` for more details. """
    if not condition:
        warn(message, category=category, stacklevel=stacklevel, source=source)


def summary(x: torch.Tensor | np.ndarray) -> str:
    """ Summarize a tensor, without full contents. """
    if isinstance(x, torch.Tensor):
        return f"[{x.dtype} {tuple(x.shape)} {x.device}]"
    elif isinstance(x, np.ndarray):
        return f"[{x.dtype} {tuple(x.shape)}]"
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
