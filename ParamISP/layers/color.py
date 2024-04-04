import torch
from kornia.enhance import AdjustGamma, adjust_gamma
from kornia.color import rgb_to_linear_rgb as srgb_to_linear, linear_rgb_to_rgb as linear_to_srgb


def apply_white_balance(x: torch.Tensor, white_balance: torch.Tensor) -> torch.Tensor:
    """ Apply white balance to the images.

    Args:
        x: Bayer image, `(*, C, H, W)`.
        white_balance: White balance, `(*, C)`.

    Returns:
        White balanced image, `(N, 3, H, W)`.
    """
    assert x.ndim >= 3, f"Expected color image, got {x.shape}-dim"
    assert white_balance.ndim >= 1, f"Expected white balance, got {white_balance.shape}"
    assert x.shape[-3] == white_balance.shape[-1], \
        f"mismatch between `x` and `white_balance` channels, got {x.shape[-3]} != {white_balance.shape[-1]}"

    x = x * white_balance.unsqueeze(-1).unsqueeze(-1)

    return x


def apply_color_matrix(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """ Apply color correction matrix to the RGB images.

    Inputs:
        x (torch.Tensor): RGB image, `(*, C_{in}, H, W)`.
        matrix (torch.Tensor): Color correction matrix, `(*, C_{out}, C_{in})`.

    Outputs:
        torch.Tensor: Color corrected image, `(*, C_{out}, H, W)`.
    """
    assert x.ndim >= 3, f"Expected color image, got {x.shape}-dim"
    assert matrix.ndim >= 2, f"Expected matrix, got {matrix.shape}"

    xC, xH, xW = x.shape[-3:]
    C_out, C_in = matrix.shape[-2:]
    assert xC == C_in, f"`x` and `matrix` should have the same number of channels, got {xC} != {C_in}"

    x_pixels = x.flatten(start_dim=-2)                  # (*, C_{in}, H*W)
    y_pixels = torch.bmm(matrix, x_pixels)              # (*, C_{out}, H*W)
    y = y_pixels.reshape(*x.shape[:-3], C_out, xH, xW)  # (*, C_{out}, H, W)

    return y


def apply_gamma_correction(x: torch.Tensor, gamma: torch.Tensor | float) -> torch.Tensor:
    """ Apply gamma correction to each pixels with given parameters, safe for negative values.

    Args:
        x: input tensor, `(*, C, H, W)`.
        gamma: parameters tensor, `(*, 1)`.

    Returns:
        output tensor, `(*, C, H, W)`.
    """
    assert x.ndim >= 3, f"Expected color image, got {x.shape}-dim"
    BDIM = x.shape[:-3]

    if isinstance(gamma, float):
        gamma = torch.tensor(gamma, device=x.device, dtype=x.dtype)
    elif gamma.numel() == 1:
        gamma = gamma.flatten()
    else:
        gamma = gamma.reshape(*BDIM, 1, 1, 1)

    eps = 1e-12
    y = x.sign() * x.abs().clip(eps).pow(gamma)   # (*, C, H, W)
    return y
