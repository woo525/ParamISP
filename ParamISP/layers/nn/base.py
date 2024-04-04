from typing import Callable
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class Linear(nn.Linear):
    """ Simplified fully-connected layer."""

    def __init__(self, in_features: int, out_features: int, *,
                 bias: bool = True, init_weights: Callable = init.xavier_normal_,
                 device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        init_weights(self.weight)


class Conv2d(nn.Conv2d):
    """ Simplified 2D convolution layer. """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *,
                 stride: int = 1, padding: str | int = "preserve", dilation: int = 1, groups: int = 1,
                 bias: bool = True, padding_mode: str = "zeros",  # TODO: refine this type
                 init_weights: Callable = init.xavier_normal_, device=None, dtype=None):
        if padding == "preserve":
            assert stride == 1, "Preserve padding only works with stride=1"
            assert (dilation * (kernel_size - 1)) % 2 == 0, "Preserve padding only works with odd kernel sizes"
            padding = (dilation * (kernel_size - 1)) // 2
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation, groups=groups,
                         bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        init_weights(self.weight)


class HyperLinearEmbedder(nn.Module):
    """ Hypernetwork for embedding modulation parameters for a fully-connected layer.

    Args:
        n_params: The number of hyperparameters.
        n_layers: The number of target layers to be modulated.
        dim_w: The dimensionality of the weights, `(out_channels, in_channels)`.
        dim_b: The dimensionality of the bias, `(out_channels)`
        mid_channels: The number of channels in the hidden layers.
        init_weights: The weight initialization function.

    Inputs:
        h: hyperparameters, `(*, n_params)`.
        index: One-hot vector of the layer index of the target in the embedder.

    Outputs:
        w: The weight modulation parameters, `(*, out_channels//groups, in_channels, kernel_size, kernel_size)`.
        b: The bias modulation parameters, `(*, out_channels)`.
    """

    def __init__(self, n_params: int, n_layers: int, dim_w: torch.Size, dim_b: torch.Size | None,
                 mid_channels: int = 64, init_weights: Callable = lambda x: init.xavier_normal_(x, 2e-3)):
        super().__init__()
        self.n_params = n_params
        self.n_layers = n_layers
        self.dim_w = dim_w
        self.dim_b = dim_b
        self.mid_channels = mid_channels

        init_weights = lambda x: init.xavier_normal_(x, 2e-6)

        out_channels, in_channels = dim_w
        self.out_channels = out_channels
        self.in_channels = in_channels
        assert dim_b is None or dim_b == (out_channels,), "Invalid bias shape"

        self.fc_w_param = Linear(n_params, mid_channels * in_channels, init_weights=init_weights)
        self.fc_w_index = Linear(n_layers, mid_channels * in_channels, init_weights=init_weights)
        self.fc_w_embed = Linear(mid_channels, out_channels, init_weights=init_weights)
        if dim_b is not None:
            self.fc_b_param = Linear(n_params, out_channels, init_weights=init_weights)
            self.fc_b_index = Linear(n_layers, out_channels, init_weights=init_weights)

    def forward(self, h: torch.Tensor, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert h.shape[-1] == self.n_params, f"Expected {self.n_params} hyperparameters, got {h.shape[-1]}"
        assert index.shape == (self.n_layers,), f"Index shape should be ({self.n_layers},), got {index.shape}"

        w: torch.Tensor
        w = self.fc_w_param(h) + self.fc_w_index(index)                    # (*, C_in * C_mid)
        w = w.reshape(*h.shape[:-1], self.in_channels, self.mid_channels)  # (*, C_in, C_mid)
        w = self.fc_w_embed(w)                                             # (*, C_in, C_out)
        w = w.swapaxes(-2, -1)                                             # (*, C_out, C_in)

        b = None
        if self.dim_b is not None:
            b = self.fc_b_param(h) + self.fc_b_index(index)  # (*, C_out)

        return w, b


class HyperLinear(nn.Module):
    """ fully-connected layer with parameter modulation.

    Args:
        target: The target convolution layer to be modulated.
        embedder: The hypernetwork that embeds the modulation parameters.
        index: One-hot vector of the layer index of the target in the embedder.

    Inputs:
        x: input tensor, `(*, in_channels)`.
        h: hyperparameters, `(*, n_params)`.

    Outputs:
        output tensor, `(*, out_channels)`.
    """

    def __init__(self, target: Linear | nn.Linear, embedder: nn.Module, index: torch.Tensor):
        super().__init__()
        self.target = target
        self.embedder = embedder
        self.register_buffer("index", index)
        self.index: torch.Tensor

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        w: torch.Tensor
        b: torch.Tensor | None
        w, b = self.embedder(h, self.index)

        fc = self.target

        weights = fc.weight + w

        if fc.bias is None:
            assert b is None, "Target layer has no bias, but embedder produces one"
            bias = None
        else:
            bias = fc.bias
            if b is not None:
                bias = bias + b

        y = x  # (*BDIM, C_in)
        y = x.reshape(1, -1, 1, 1)  # (1, BDIM * C_in, 1, 1)
        weights = weights.reshape(-1, self.target.in_features, 1, 1)  # (BDIM * C_out, C_in, 1, 1)

        y = F.conv2d(y, weights, groups=x.shape[:-1].numel())  # (1, BDIM * C_out, 1, 1)

        y = y.reshape(*x.shape[:-1], self.target.out_features)  # (*BDIM, C_out)

        if bias is not None:
            y = y + bias

        return y

    @classmethod
    def build(cls, n_params: int, targets: list[Linear | nn.Linear],
              mid_channels: int = 64, init_weights: Callable = lambda x: init.xavier_normal_(x, 2e-3)):
        """ Build a list of hyper-modules for each target modules.

        Args:
            n_params: The number of hyperparameters.
            targets: The list of Conv2d.
            mid_channels: The number of channels in the hidden layers.
            init_weights: The weight initialization function.

        Returns:
            A list of HyperConv2d.
        """
        n_layers = len(targets)
        dim_w = targets[0].weight.shape
        dim_b = targets[0].bias.shape if targets[0].bias is not None else None

        assert all(t.weight.shape == dim_w for t in targets), "All target layers must have the same weight shape"
        if dim_b is not None:
            assert all(t.bias is not None and t.bias.shape == dim_b for t in targets), \
                "All target layers must have the same bias shape"
        else:
            assert all(t.bias is None for t in targets), "All target layers must have the same bias shape"

        embedder = HyperLinearEmbedder(n_params, n_layers, dim_w, dim_b, mid_channels, init_weights)

        indices = torch.eye(n_layers)
        return [cls(target, embedder, indices[i]) for i, target in enumerate(targets)]


class HyperConv2dEmbedder(nn.Module):
    """ Hypernetwork for embedding modulation parameters for a 2d convolution layer.

    Args:
        n_params: The number of hyperparameters.
        n_layers: The number of target layers to be modulated.
        dim_w: The dimensionality of the weights, `(out_channels//groups, in_channels, height, width)`.
        dim_b: The dimensionality of the bias, `(out_channels)`
        mid_channels: The number of channels in the hidden layers.
        init_weights: The weight initialization function.

    Inputs:
        h: hyperparameters, `(*, n_params)`.
        index: One-hot vector of the layer index of the target in the embedder.

    Outputs:
        w: The weight modulation parameters, `(*, out_channels//groups, in_channels, kernel_size, kernel_size)`.
        b: The bias modulation parameters, `(*, out_channels)`.
    """

    def __init__(self, n_params: int, n_layers: int, dim_w: torch.Size, dim_b: torch.Size | None,
                 mid_channels: int = 64, init_weights: Callable = lambda x: init.xavier_normal_(x, 2e-3)):
        super().__init__()
        self.n_params = n_params
        self.n_layers = n_layers
        self.dim_w = dim_w
        self.dim_b = dim_b
        self.mid_channels = mid_channels

        out_channels, in_channels, ksize_h, ksize_w = dim_w
        self.out_channels = out_channels
        self.in_channels = in_channels
        assert dim_b is None or dim_b == (out_channels,), "Invalid bias shape"

        self.fc_w_param = Linear(n_params, mid_channels * in_channels, init_weights=init_weights)
        self.fc_w_index = Linear(n_layers, mid_channels * in_channels, init_weights=init_weights)
        self.fc_w_embed = Linear(mid_channels, out_channels * ksize_h * ksize_w, init_weights=init_weights)
        if dim_b is not None:
            self.fc_b_param = Linear(n_params, out_channels, init_weights=init_weights)
            self.fc_b_index = Linear(n_layers, out_channels, init_weights=init_weights)

    def forward(self, h: torch.Tensor, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert h.shape[-1] == self.n_params, f"Expected {self.n_params} hyperparameters, got {h.shape[-1]}"
        assert index.shape == (self.n_layers,), f"Index shape should be ({self.n_layers},), got {index.shape}"

        w: torch.Tensor
        w = self.fc_w_param(h) + self.fc_w_index(index)                    # (*, C_in * C_mid)
        w = w.reshape(*h.shape[:-1], self.in_channels, self.mid_channels)  # (*, C_in, C_mid)
        w = self.fc_w_embed(w)                                             # (*, C_in, C_out * k * k)
        w = w.reshape(*w.shape[:-1], self.out_channels, *self.dim_w[-2:])  # (*, C_in, C_out, k, k)
        w = w.swapaxes(-4, -3)                                             # (*, C_out, C_in, k, k)

        b = None
        if self.dim_b is not None:
            b = self.fc_b_param(h) + self.fc_b_index(index)  # (*, C_out)

        return w, b


class HyperConv2d(nn.Module):
    """ 2d convolution layer with parameter modulation.

    Args:
        target: The target convolution layer to be modulated.
        embedder: The hypernetwork that embeds the modulation parameters.
        index: One-hot vector of the layer index of the target in the embedder.

    Inputs:
        x: input tensor, `(*, in_channels, height, width)`.
        h: hyperparameters, `(*, n_params)`.

    Outputs:
        output tensor, `(*, out_channels, out_height, out_width)`.
    """

    def __init__(self, target: Conv2d | nn.Conv2d, embedder: nn.Module, index: torch.Tensor):
        super().__init__()
        self.target = target
        self.embedder = embedder
        self.register_buffer("index", index)
        self.index: torch.Tensor

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        w: torch.Tensor
        b: torch.Tensor | None
        w, b = self.embedder(h, self.index)

        conv = self.target

        weights = conv.weight + w

        if conv.bias is None:
            assert b is None, "Target layer has no bias, but embedder produces one"
            bias = None
        else:
            bias = conv.bias
            if b is not None:
                bias = bias + b

        y = x  # (*BDIM, C_in, H, W)

        y = y.reshape(1, -1, *x.shape[-2:])          # (1, BDIM * C_in, H, W)
        weights = weights.flatten(end_dim=-4)        # (BDIM * C_out, C_in, k, k)
        groups = conv.groups * x.shape[:-3].numel()  # BDIM * groups

        if conv.padding_mode != "zeros":
            y = F.conv2d(F.pad(y, conv._reversed_padding_repeated_twice, mode=conv.padding_mode),
                         weights, stride=conv.stride, padding=0, dilation=conv.dilation, groups=groups)
        else:
            y = F.conv2d(y, weights, stride=conv.stride, padding=conv.padding,
                         dilation=conv.dilation, groups=groups)

        y = y.reshape(*x.shape[:-3], conv.out_channels, *y.shape[-2:])  # (BDIM, C_out, H', W')

        if bias is not None:
            y = y + bias.unsqueeze(-1).unsqueeze(-1)

        return y

    @classmethod
    def build(cls, n_params: int, targets: list[Conv2d | nn.Conv2d],
              mid_channels: int = 64, init_weights: Callable = lambda x: init.xavier_normal_(x, 2e-3)):
        """ Build a list of hyper-modules for each target modules.

        Args:
            n_params: The number of hyperparameters.
            targets: The list of Conv2d.
            mid_channels: The number of channels in the hidden layers.
            init_weights: The weight initialization function.

        Returns:
            A list of HyperConv2d.
        """
        n_layers = len(targets)
        dim_w = targets[0].weight.shape
        dim_b = targets[0].bias.shape if targets[0].bias is not None else None

        assert all(t.weight.shape == dim_w for t in targets), "All target layers must have the same weight shape"
        if dim_b is not None:
            assert all(t.bias is not None and t.bias.shape == dim_b for t in targets), \
                "All target layers must have the same bias shape"
        else:
            assert all(t.bias is None for t in targets), "All target layers must have the same bias shape"

        embedder = HyperConv2dEmbedder(n_params, n_layers, dim_w, dim_b, mid_channels, init_weights)

        indices = torch.eye(n_layers)
        return [cls(target, embedder, indices[i]) for i, target in enumerate(targets)]


class ChAttn2d(nn.Module):
    """ Channel Attention Module.

    Reference:
        - Woo, S., et al. (2018). "CBAM: Convolutional Block Attention Module." ECCV.

    Args:
        n_channels: number of feature channels.
        reduction: scale of feature channels reduction: `hidden_channels = n_channels // reduction`.
        groups: number of groups for convolutional layers.
        bias: whether to use bias in convolutional layers.
        init_weights: function to initialize weights.
    """

    def __init__(self, n_channels: int, reduction: int = 4, *,
                 groups: int = 1, bias: bool = True, init_weights: Callable = init.xavier_normal_):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = Conv2d(n_channels, n_channels//reduction, 1, groups=groups, bias=bias, init_weights=init_weights)
        self.conv2 = Conv2d(n_channels//reduction, n_channels, 1, groups=groups, bias=bias, init_weights=init_weights)
        self.act = nn.ELU(inplace=True)

    def getattn(self, x: torch.Tensor) -> torch.Tensor:
        avg_y = self.conv2(self.act(self.conv1(self.avgpool(x))))
        max_y = self.conv2(self.act(self.conv1(self.maxpool(x))))
        return avg_y + max_y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.getattn(x))


class HyperChAttn2dEmbedder(nn.Module):
    """ Hypernetwork for embedding modulation parameters for a Channel Attention Module.

    Args:
        n_params: The number of hyperparameters.
        n_layers: The number of target layers to be modulated.
        n_channels: The number of feature channels in the target layers.
        init_weights: function to initialize weights.

    Inputs:
        h: hyperparameters, `(*, n_params)`.
        index: One-hot vector of the layer index of the target in the embedder.

    Outputs:
        w: The weight modulation parameters, `(*, n_channels, 1, 1)`.
    """

    def __init__(self, n_params: int, n_layers: int, n_channels: int, init_weights: Callable = lambda x: init.xavier_normal_(x, 2e-3)):
        super().__init__()
        self.n_params = n_params
        self.n_layers = n_layers
        self.n_channels = n_channels

        self.fc_param = Linear(n_params, n_channels, init_weights=init_weights)
        self.fc_index = Linear(n_layers, n_channels, init_weights=init_weights)

    def forward(self, h: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        assert h.shape[-1] == self.n_params, f"Expected {self.n_params} hyperparameters, got {h.shape[-1]}"
        assert index.shape == (self.n_layers,), f"Index shape should be ({self.n_layers},), got {index.shape}"

        w: torch.Tensor
        w = self.fc_param(h) + self.fc_index(index)  # (*, n_channels)
        w = w.unsqueeze(-1).unsqueeze(-1)            # (*, n_channels, 1, 1)

        return w


class HyperChAttn2d(nn.Module):
    """ Channel Attention Module with parameter modulation.

    Args:
        target: The target channel attention module to be modulated.
        embedder: The hypernetwork that embeds the modulation parameters.
        index: One-hot vector of the layer index of the target in the embedder.

    Inputs:
        x: input tensor, `(*, n_channels, height, width)`.
        h: hyperparameters, `(*, n_params)`.

    Outputs:
        output tensor, `(*, n_channels, height, width)`.
    """

    def __init__(self, target: ChAttn2d, embedder: nn.Module, index: torch.Tensor):
        super().__init__()
        self.target = target
        self.embedder = embedder
        self.register_buffer("index", index)
        self.index: torch.Tensor

    def getattn(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        attn = self.target
        avg_y = attn.conv2(attn.act(attn.conv1(attn.avgpool(x))))
        max_y = attn.conv2(attn.act(attn.conv1(attn.maxpool(x))))
        mod_y = attn.conv2(attn.act(attn.conv1(self.embedder(h, self.index))))
        return avg_y + max_y + mod_y

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        assert x.shape[-3] == self.target.conv1.in_channels, \
            f"Input channels should be {self.target.conv1.in_channels}, got {x.shape[-3]}"

        return x * torch.sigmoid(self.getattn(x, h))

    @classmethod
    def build(cls, n_params: int, targets: list[ChAttn2d], init_weights: Callable = lambda x: init.xavier_normal_(x, 2e-3)):
        """ Build a list of hyper-modules for each target modules.

        Args:
            n_params: The number of hyperparameters.
            targets: The list of ChAttn2d.
            init_weights: The weight initialization function.

        Returns:
            A list of HyperChAttn2d.
        """
        n_layers = len(targets)
        n_channels = targets[0].conv1.in_channels

        assert all(t.conv1.in_channels == n_channels for t in targets), "All target layers must have the same channel size"

        embedder = HyperChAttn2dEmbedder(n_params, n_layers, n_channels, init_weights)

        indices = torch.eye(n_layers)
        return [cls(target, embedder, indices[i]) for i, target in enumerate(targets)]


class SpAttn2d(nn.Module):
    """ Spatial Attention Module.

    Reference:
        - Woo, S., et al. (2018). "CBAM: Convolutional Block Attention Module." ECCV.

    Args:
        kernel_size: kernel size of convolution layers.
        dilation: dilation of convolution layers.
        bias: whether to use bias in convolutional layers.
        padding_mode: padding mode of convolution layers.
        init_weights: function to initialize weights.
    """

    def __init__(self, kernel_size: int = 7, *,
                 dilation: int = 1, bias: bool = True, padding_mode: str = "zeros",
                 init_weights: Callable = init.xavier_normal_):
        super().__init__()
        self.conv = Conv2d(2, 1, kernel_size, dilation=dilation, bias=bias, padding_mode=padding_mode,
                           init_weights=init_weights)

    def getattn(self, x: torch.Tensor) -> torch.Tensor:
        avg_y = torch.mean(x, dim=-3, keepdim=True)
        max_y = torch.max(x,  dim=-3, keepdim=True)[0]
        y = torch.cat([avg_y, max_y], dim=-3)
        return self.conv(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.getattn(x))


class HyperSpAttn2dEmbedder(nn.Module):
    """ Hypernetwork for embedding modulation parameters for a Spatial Attention Module.

    Args:
        n_params: The number of hyperparameters.
        n_layers: The number of target layers to be modulated.
        kernel_size: The kernel size of the target spatial attention module.
        init_weights: The weight initialization function.

    Inputs:
        h: hyperparameters, `(*, n_params)`.
        index: One-hot vector of the layer index of the target in the embedder.

    Outputs:
        w: The weight modulation parameters, `(*, 1, 2, kernel_size, kernel_size)`.
        b: The bias modulation parameters, `(*, 1)`.
    """

    def __init__(self, n_params: int, n_layers: int, kernel_size: int, bias: bool = True,
                 init_weights: Callable = lambda x: init.xavier_normal_(x, 2e-3)):
        super().__init__()
        self.n_params = n_params
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.bias = bias

        self.fc_w_param = Linear(n_params, 2 * kernel_size * kernel_size, init_weights=init_weights)
        self.fc_w_index = Linear(n_layers, 2 * kernel_size * kernel_size, init_weights=init_weights)
        if self.bias:
            self.fc_b_param = Linear(n_params, 1, init_weights=init_weights)
            self.fc_b_index = Linear(n_layers, 1, init_weights=init_weights)

    def forward(self, h: torch.Tensor, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert h.shape[-1] == self.n_params, f"Expected {self.n_params} hyperparameters, got {h.shape[-1]}"
        assert index.shape == (self.n_layers,), f"Index shape should be ({self.n_layers},), got {index.shape}"

        w: torch.Tensor
        w = self.fc_w_param(h) + self.fc_w_index(index)                         # (*, 2 * kernel_size * kernel_size)
        w = w.reshape(*h.shape[:-1], 1, 2, self.kernel_size, self.kernel_size)  # (*, 1, 2, ke

        b = None
        if self.bias:
            b = self.fc_b_param(h) + self.fc_b_index(index)  # (*, 1)

        return w, b


class HyperSpAttn2d(nn.Module):
    """ Spatial Attention Module with parameter modulation.

    Args:
        target: The target spatial attention module to be modulated.
        embedder: The hypernetwork that embeds the modulation parameters.
        index: One-hot vector of the layer index of the target in the embedder.

    Inputs:
        x: input tensor, `(*, n_channels, height, width)`.
        h: hyperparameters, `(*, n_params)`.

    Outputs:
        output tensor, `(*, n_channels, height, width)`.
    """

    def __init__(self, target: SpAttn2d, embedder: nn.Module, index: torch.Tensor):
        super().__init__()
        self.target = target
        self.embedder = embedder
        self.register_buffer("index", index)
        self.index: torch.Tensor
        self.hconv = HyperConv2d(target.conv, embedder, index)

    def getattn(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        avg_y = torch.mean(x, dim=-3, keepdim=True)
        max_y = torch.max(x,  dim=-3, keepdim=True)[0]
        y = torch.cat([avg_y, max_y], dim=-3)
        return self.hconv(y, h)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.getattn(x, h))

    @classmethod
    def build(cls, n_params: int, targets: list[SpAttn2d], init_weights: Callable = lambda x: init.xavier_normal_(x, 2e-3)):
        """ Build a list of hyper-modules for each target modules.

        Args:
            n_params: The number of hyperparameters.
            targets: The list of SpAttn2d.
            init_weights: The weight initialization function.

        Returns:
            A list of HyperSpAttn2d.
        """
        n_layers = len(targets)
        kernel_size = targets[0].conv.kernel_size[0]
        bias = targets[0].conv.bias is not None

        assert all(t.conv.kernel_size[0] == kernel_size for t in targets), \
            "All targets should have the same kernel size"
        if bias:
            assert all(t.conv.bias is not None for t in targets), "All target layers must have the same bias size"
        else:
            assert all(t.conv.bias is None for t in targets), "All target layers must have the same bias size"

        embedder = HyperSpAttn2dEmbedder(n_params, n_layers, kernel_size, bias, init_weights)

        indices = torch.eye(n_layers)
        return [HyperSpAttn2d(target, embedder, indices[i]) for i, target in enumerate(targets)]
