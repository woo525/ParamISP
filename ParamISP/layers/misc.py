from typing import Sequence

import torch
import torch.nn.functional as F


def replication_pad2d(x: torch.Tensor, pad: Sequence[int]) -> torch.Tensor:
    """ Replication padding for 2D tensor.

    Args:
        x: input tensor, (*, H, W)
        pad: padding size, (pad_left, pad_right, pad_top, pad_bottom)

    Returns:
        y: padded tensor, (*, H + pad_top + pad_bottom, W + pad_left + pad_right)
    """
    assert x.ndim >= 2, f"Expected at least 2-dim input, got {x.ndim}-dim"
    assert len(pad) == 4, f"Expected 4 values for padding, got {len(pad)}"
    PL, PR, PT, PB = pad

    x = F.pad(x, (PL, PR, PT, PB))
    # fmt: off
    x[...,    : PT,    : PL] = x[...,  PT  : PT+1,  PL  : PL+1]  # top-left
    x[...,    : PT,  PL:-PR] = x[...,  PT  : PT+1,  PL  :-PR  ]  # top
    x[...,    : PT, -PR:   ] = x[...,  PT  : PT+1, -PR-1:-PR  ]  # top-right
    x[...,  PT:-PB, -PR:   ] = x[...,  PT  :-PB  , -PR-1:-PR  ]  # right
    x[..., -PB:   , -PR:   ] = x[..., -PB-1:-PB  , -PR-1:-PR  ]  # bottom-right
    x[..., -PB:   ,  PL:-PR] = x[..., -PB-1:-PB  ,  PL  :-PR  ]  # bottom
    x[..., -PB:   ,    : PL] = x[..., -PB-1:-PB  ,  PL  : PL+1]  # bottom-left
    x[...,  PT:-PB,    : PL] = x[...,  PT  :-PB  ,  PL  : PL+1]  # left
    # fmt: on

    return x


def positional_encoding(x: torch.Tensor, n_freqs: int) -> torch.Tensor:
    """ Sine-cosine positional encoding.

    Reference:
        - Mildenhall, B., et al. (2020). Nerf: Representing scenes as neural radiance fields for view synthesis.
          European conference on computer vision, Springer.

    Args:
        x: points, (*, n_params)
        n_freqs: number of frequencies to use for each coordinate.

    Returns:
        y: encoded points, (*, n_params * (1 + 2*n_freqs))
    """
    freq_bands = 2.**torch.linspace((2-n_freqs)//2, n_freqs//2, n_freqs)  # modified heuristically

    embedding = [x]
    for freq in freq_bands:
        embedding.append(torch.sin(x * freq))
        embedding.append(torch.cos(x * freq))

    return torch.cat(embedding, dim=-1)


def coordinate_map(x: torch.Tensor, full_size: list[tuple[int, int]], top_left: list[tuple[int, int]]) -> torch.Tensor:
    """ Compute position map of each pixels.

    Args:
        x: input tensor, `(N, C, H, W)`.
        full_size: full size of each image, `[(H, W), ...]`.
        top_left: top-left position of each image, `[(y, x), ...]`.

    Returns:
        output tensor, `(N, 6, H, W)`.
    """
    assert x.ndim == 4, f"Expected batch image, got {x.shape}-dim"
    assert len(full_size) == len(top_left), \
        f"Expected same length of `full_size` and `top_left`, got {len(full_size)} and {len(top_left)}"

    N = len(full_size)
    H, W = x.shape[-2:]
    y_i, x_i = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    y_i = y_i.to(device=x.device, dtype=x.dtype)
    x_i = x_i.to(device=x.device, dtype=x.dtype)

    y_abs = torch.stack([y_i + top_left[b][0] - (full_size[b][0]//2) for b in range(N)])
    x_abs = torch.stack([x_i + top_left[b][1] - (full_size[b][1]//2) for b in range(N)])
    r_abs = torch.sqrt(y_abs**2 + x_abs**2)
    y_rel = torch.stack([y_abs[b] / full_size[b][0] for b in range(N)])
    x_rel = torch.stack([x_abs[b] / full_size[b][1] for b in range(N)])
    r_rel = torch.sqrt(y_rel**2 + x_rel**2)

    return torch.stack([y_abs, x_abs, r_abs, y_rel, x_rel, r_rel], dim=-3)
