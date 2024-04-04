import torch
import torch.nn as nn
import torch.nn.functional as F

from .misc import replication_pad2d


def gather(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """ Gathers pixels from `x` according to `index` which follows bayer pattern.
    NOTE: Avoid using `torch.gather` since it does not supports deterministic mode.

    Args:
        x: Tensor of shape :math:`(N, C_{in}, H, W)`.
        index: Gather index with bayer pattern of shape :math:`(1|N, C_{out}, 2, 2)`.

    Returns:
        Gathered values of shape :math:`(N, C_{out}, H, W)`.
    """
    assert x.dim()     == 4, f"`x` must be 4-dimensional, got {x.dim()}"
    assert index.dim() == 4, f"`index` must be 4-dimensional, got {index.dim()}"

    xN, xC, xH, xW = x.shape
    iN, iC, iH, iW = index.shape
    C_in, C_out = xC, iC

    assert iN == xN or iN == 1, \
        f"Batch dimension of `index` must be `1` or equal to `x`, got {iN}"
    assert xH % 2 == 0 and xW % 2 == 0, \
        f"`x` must have even height and width, got {xH}x{xW}"
    assert iH == 2 and iW == 2, \
        f"`index` must be 2x2 size, got {iH}x{iW}"

    # x_down: (N, 4*C_in, H/2, W/2)
    x_down = F.pixel_unshuffle(x, 2)
    # x_quat: (N, 4, 1, C_in, H/2, W/2)
    x_quat = x_down.reshape((xN, -1, 4, xH//2, xW//2)).swapaxes(1, 2).unsqueeze(2)

    # i_down: (1|N, 4*C_out)
    i_down = F.pixel_unshuffle(index, 2).squeeze(-1).squeeze(-1)
    # i_quat: (1|N, 4, C_out, 1)
    i_quat = i_down.reshape((iN, -1, 4)).swapaxes(1, 2).unsqueeze(3)

    # mask_index: (N, 4, C_out, C_in)
    mask_index = torch.arange(C_in, dtype=index.dtype, device=index.device).repeat(xN, 4, C_out, 1)
    # mask: (N, 4, C_out, C_in)
    mask = (mask_index == i_quat)

    # x_gath: (N, 4, C_out, H/2, W/2)
    x_gath = (x_quat * mask.unsqueeze(-1).unsqueeze(-1)).sum(dim=3)

    # y_down: (N, 4*C_out, H/2, W/2)
    y_down = x_gath.swapaxes(1, 2).flatten(start_dim=1, end_dim=2)
    # y: (N, C_out, H, W)
    y = F.pixel_shuffle(y_down, 2)

    return y


def mosaic(x: torch.Tensor, bayer_pattern: torch.Tensor) -> torch.Tensor:
    """ Mosaicing RGB images to Bayer pattern. 

    Args:
        x: RGB image of shape :math:`(N, 3, H, W)`.
        bayer_pattern: Bayer pattern of `x` of shape :math:`(1|N, 1, 2, 2)`.

    Returns:
        Mosaicked image of shape :math:`(N, 1, H, W)`.
    """
    assert x.dim()             == 4, f"`x` must be 4-dimensional, got {x.dim()}"
    assert bayer_pattern.dim() == 4, f"`bayer_pattern` must be 4-dimensional, got {bayer_pattern.dim()}"

    xN, xC, xH, xW = x.shape
    bN, bC, bH, bW = bayer_pattern.shape

    assert xC == 3, f"Channel dimension of `x` must be 3, got {xC}"
    assert xH % 2 == 0 and xW % 2 == 0, f"`x` must have even height and width, got {xH}x{xW}"
    assert bN == xN or bN == 1, f"Batch dimension of `bayer_mask` must be `1` or equal to `x`, got {bN}"
    assert bC == 1, f"Channel dimension of `bayer_mask` must be 1, got {bC}"
    assert bH == 2 and bW == 2, f"Height and width dimension of `bayer_mask` must be 2, got {bH}x{bW}"

    y = gather(x, bayer_pattern)
    return y


class Demosaic(nn.Module):
    """ Demosaicing of bayer images using Malvar-He-Cutler method.

    References:
        - Malvar, Henrique S., Li-wei He, and Ross Cutler.
        "High-quality linear interpolation for demosaicing of Bayer-patterned color images."
        2004 IEEE International Conference on Acoustics, Speech, and Signal Processing. Vol. 3. IEEE, 2004.

    Inputs:
        x (torch.Tensor): Bayer image of shape :math:`(N, 1, H, W)`.
        bayer_pattern (torch.Tensor): Bayer pattern of `x` of shape :math:`(1|N, 1, 2, 2)`.

    Outputs:
        torch.Tensor: Demosaicked image of shape :math:`(N, 3, H, W)`.
    """

    def __init__(self):
        super().__init__()
        self.kernels: torch.Tensor
        self.register_buffer("kernels", torch.tensor([
            # + (G at R,B locations)
            [+0,  0, -2,  0,  0],
            [+0,  0,  4,  0,  0],
            [-2,  4,  8,  4, -2],
            [+0,  0,  4,  0,  0],
            [+0,  0, -2,  0,  0],
            # x (R at B and B at R)
            [+0,  0, -3,  0,  0],
            [+0,  4,  0,  4,  0],
            [-3,  0, 12,  0, -3],
            [+0,  4,  0,  4,  0],
            [+0,  0, -3,  0,  0],
            # - (R,B at G in R rows)
            [+0,  0,  1,  0,  0],
            [+0, -2,  0, -2,  0],
            [-2,  8, 10,  8, -2],
            [+0, -2,  0, -2,  0],
            [+0,  0,  1,  0,  0],
            # | (R,B at G in B rows)
            [0,  0, -2,  0,  0],
            [0, -2,  8, -2,  0],
            [1,  0, 10,  0,  1],
            [0, -2,  8, -2,  0],
            [0,  0, -2,  0,  0],
        ], dtype=torch.get_default_dtype()).view(4, 1, 5, 5) / 16.0)

        self.indices_rggb: torch.Tensor
        self.register_buffer("indices_rggb", torch.tensor([
            # R
            [4, 2],  # R G
            [3, 1],  # G B
            # G
            [0, 4],  # R G
            [4, 0],  # G B
            # B
            [1, 3],  # R G
            [2, 4],  # G B
        ]).view(1, 3, 2, 2))

    def _calc_index(self, bayer_pattern: torch.Tensor) -> torch.Tensor:
        """ Calculate required index. 

        Shape:
            - Input: :math:`(1, 1, 2, 2)`
            - Output: :math:`(1, 3, 2, 2)`
        """
        match tuple(bayer_pattern.flatten().tolist()):
            case [0, 1, 1, 2]: return self.indices_rggb
            case [1, 0, 2, 1]: return self.indices_rggb.roll(1, dims=-1)
            case [1, 2, 0, 1]: return self.indices_rggb.roll(1, dims=-2)
            case [2, 1, 1, 0]: return self.indices_rggb.roll(1, dims=-1).roll(1, dims=-2)
            case _: raise ValueError("Invalid bayer pattern")

    def forward(self, x: torch.Tensor, bayer_pattern: torch.Tensor) -> torch.Tensor:
        assert x.dim()             == 4, f"`x` must be 4-dimensional, got {x.dim()}"
        assert bayer_pattern.dim() == 4, f"`bayer_pattern` must be 4-dimensional, got {bayer_pattern.dim()}"

        xN, xC, xH, xW = x.shape
        bN, bC, bH, bW = bayer_pattern.shape

        assert xC == 1, f"`x` must have 1 channel, got {xC}"
        assert xH % 2 == 0 and xW % 2 == 0, f"`x` must have even height and width, got {xH}x{xW}"
        assert bN == xN or bN == 1, f"Batch dimension of `bayer_mask` must be `1` or equal to `x`, got {bN}"
        assert bC == 1, f"Channel dimension of `bayer_mask` must be 1, got {bC}"
        assert bH == 2 and bW == 2, f"Height and width dimension of `bayer_mask` must be 2, got {bH}x{bW}"

        
        # x_pad = F.pad(x, (2, 2, 2, 2), mode="replicate")
        x_pad = replication_pad2d(x, (2, 2, 2, 2))
        feats = F.conv2d(x_pad, self.kernels)
        feats = torch.cat((feats, x), dim=1)  # (N, 5, H, W)

        index = torch.cat([self._calc_index(pattern) for pattern in bayer_pattern], dim=0)  # (N, 3, 2, 2)

        y = gather(feats, index)
        return y


def apply_white_balance(x: torch.Tensor, bayer_pattern: torch.Tensor, white_balance: torch.Tensor, mosaic_flag=True) -> torch.Tensor:
    """ Apply white balance to the bayer images.

    Args:
        x: Bayer image of shape :math:`(N, 1, H, W)`.
        bayer_pattern: Bayer pattern of `x` of shape :math:`(1|N, 1, 2, 2)`.
        white_balance: White balance of shape :math:`(1|N, 3)`.

    Returns:
        White balanced image of shape :math:`(N, 1, H, W)`.
    """
    
    if mosaic_flag:
        assert x.dim()             == 4, f"`x` must be 4-dimensional, got {x.dim()}"
        assert bayer_pattern.dim() == 4, f"`bayer_pattern` must be 4-dimensional, got {bayer_pattern.dim()}"
        assert white_balance.dim() == 2, f"`white_balance` must be 2-dimensional, got {white_balance.dim()}"

    xN, xC, xH, xW = x.shape
    bN, bC, bH, bW = bayer_pattern.shape
    wN, wC = white_balance.shape

    if mosaic_flag:
        assert xC == 1, f"`x` must have 1 channel, got {xC}"
        assert xH % 2 == 0 and xW % 2 == 0, f"`x` must have even height and width, got {xH}x{xW}"
        assert bN == xN or bN == 1, f"Batch dimension of `bayer_mask` must be `1` or equal to `x`, got {bN}"
        assert bC == 1, f"Channel dimension of `bayer_mask` must be 1, got {bC}"
        assert bH == 2 and bW == 2, f"Height and width dimension of `bayer_mask` must be 2, got {bH}x{bW}"
        assert wN == xN or wN == 1, f"Batch dimension of `white_balance` must be `1` or equal to `x`, got {wN}"
        assert wC == 3, f"Channel dimension of `white_balance` must be 3, got {wC}"
    
    white_balance = white_balance / white_balance[:, 1:2]
    
    if mosaic_flag:
        x_rgb = x.repeat(1, wC, 1, 1) * white_balance.unsqueeze(-1).unsqueeze(-1)
        y = gather(x_rgb, bayer_pattern)
    else: # mosaic_flag == False
        x[:,0,:,:] = x[:,0,:,:] * white_balance[:,0].unsqueeze(-1).unsqueeze(-1) # R
        x[:,1,:,:] = x[:,1,:,:] * white_balance[:,1].unsqueeze(-1).unsqueeze(-1) # G
        x[:,2,:,:] = x[:,2,:,:] * white_balance[:,2].unsqueeze(-1).unsqueeze(-1) # B
        y = x

    return y
