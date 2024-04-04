import io

import torch
import torchvision.transforms.functional as TF
import torchvision.utils
import numpy as np

import PIL.Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as mpFigure
from matplotlib.axes import Axes as mpAxes
from matplotlib.patches import Rectangle as mpRectangle
from toolz import isiterable

from .convert import as_numpy_image, as_torch_tensor, as_u8image, as_f32image, TTensor


def _enumerate_nullable(iterable):
    return enumerate(iterable if iterable is not None else [])


def display(img: torch.Tensor | np.ndarray):
    """ Display image using IPython; use only in Jupyter environments.

    Args:
        img: image to display.
    """
    from IPython.display import display
    display(PIL.Image.fromarray(as_u8image(as_numpy_image(img))))


def draw_text(
    text: str,
    pad: int = 2,
    h: int | None = None,
    w: int | None = None,
    *,
    dpi: int = 300,
    fontsize: int = 16,
    color: str = "white",
    bgcolor: str = "none",
    textfont: str = "STIXGeneral",
    mathfont: str = "stix",
) -> np.ndarray:
    """ Draw text on a transparent background with prefered size.

    Args:
        text: text to draw.
        h: height of the output image, auto if None.
        w: width of the output image, auto if None.
        dpi: dpi when drawing the text.
        fontsize: fontsize when drawing the text.
        color: color of the text.
    """
    fig = mpFigure(facecolor=bgcolor, dpi=dpi)

    with plt.rc_context({"mathtext.fontset": mathfont}):
        fig.text(0, 0, text, fontname=textfont, fontsize=fontsize, color=color)

    with io.BytesIO() as buffer:
        fig.savefig(buffer, dpi=dpi, format="png", bbox_inches="tight", pad_inches=0)
        buffer.seek(0)
        img = PIL.Image.open(buffer)
        img = as_numpy_image(img)  # type: ignore

    bg_pixel = img[0, 0, :]

    H, W = img.shape[:2]
    if h is not None and w is None:
        w = round(W * (h-2*pad) / H + 2*pad)
    elif h is None and w is not None:
        h = round(H * (w-2*pad) / W + 2*pad)
    elif h is not None and w is not None:
        if h/H > w/W:
            h = round(H * (w-2*pad)/W + 2*pad)
        else:
            w = round(W * (h-2*pad)/H + 2*pad)

    if h is not None and w is not None:
        img = PIL.Image.fromarray(img)
        img = img.resize((w-2*pad, h-2*pad), PIL.Image.BICUBIC)
        img = as_numpy_image(img)  # type: ignore

    # add padding
    img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")  # type: ignore
    img[:pad, :] = img[-pad:, :] = img[:, :pad] = img[:, -pad:] = bg_pixel

    return img


def draw_text_on_image(
    img: TTensor,
    text: str,
    hratio: float = 0.1,
    pad: int = 2,
    *,
    color: str = "#ffffff",
    bgcolor: str = "#00000099",
    textfont: str = "STIXGeneral",
    mathfont: str = "stix",
) -> TTensor:
    """ Draw text on image.

    Args:
        img: image to draw text on.
        text: text to draw.
        height: height of the text.
        color: color of the text.
        bgcolor: background color of the text.
        textfont: font of the text.
        mathfont: font of the math text.

    Returns:
        image with text drawn on, as float32 or uint8 image.
    """
    if was_tensor := isinstance(img, torch.Tensor):
        device = img.device
        img = as_numpy_image(img)  # type: ignore

    dtype = img.dtype
    img = as_f32image(img)

    H, W = img.shape[:2]
    h = round(H * hratio)

    text_img = draw_text(text, h=h, w=W, fontsize=h//2, pad=pad,
                         color=color, bgcolor=bgcolor, textfont=textfont, mathfont=mathfont)
    text_img = as_f32image(text_img)
    h, w = text_img.shape[:2]

    # add text_img right-bottom corner of img
    img[-h:, -w:, :3] = (text_img[:, :, 3:] * text_img[:, :, :3] +  # type: ignore
                         (1 - text_img[:, :, 3:]) * img[-h:, -w:, :3])

    if dtype == np.uint8:
        img = as_u8image(img)

    if was_tensor:
        img = as_torch_tensor(img).to(device)  # type: ignore

    return img


def rawgrid(images, *, padding: int = 2, transpose: bool = False) -> torch.Tensor:
    """ Plot images as grid with exact pixel matching.

    Args:
        images: images to plot; can be a list or tensor of images.
        padding: padding between images.
        plot: Display outputs using IPython; use only in Jupyter environments.
        transpose: transpose grid axis.
    """
    assert isiterable(images) and len(images) > 0 and all(map(isiterable, images)), \
        "`images` must be a 2D list of images"
    assert any(len(row) > 0 for row in images), \
        "images must contains at least one image"

    nrows = len(images)
    ncols = max(len(row) for row in images)
    if transpose:
        nrows, ncols = ncols, nrows

    imgs = [torch.zeros(3, 1, 1, dtype=torch.uint8)
            for _ in range(ncols)
            for _ in range(nrows)]

    max_h, max_w = 1, 1
    for R, row in _enumerate_nullable(images):
        for C, img in _enumerate_nullable(row):
            img = as_u8image(as_torch_tensor(img))
            assert img.ndim == 3, f"image must be a 3D tensor, received {img.ndim}"
            if img.shape[0] == 1:
                img = img.expand(3, *img.shape[1:])

            h, w = img.shape[-3: -1]
            max_h = max(max_h, h)
            max_w = max(max_w, w)
            imgs[C * ncols + R if transpose else R * ncols + C] = img

    for i, img in enumerate(imgs):
        h, w = img.shape[-3: -1]
        imgs[i] = TF.pad(img, [0, 0, max_w - w, max_h - h], padding_mode="constant", fill=0)

    results = torchvision.utils.make_grid(imgs, nrow=ncols, padding=padding)

    return results


def grid(images, *, figsize=None, transpose: bool = False,
         suptitle=None, supxlabel=None, supylabel=None, suptitlesize=None, suplabelsize=None,
         titles=None, xlabels=None, ylabels=None, titlesize=None, labelsize=None,
         xtick=False, ytick=False, border=True) -> tuple[mpFigure, list[list[mpAxes]]]:
    """ Plot images as grid.

    Args:
        images: PIL.Image, numpy.ndarray, or torch.Tensor. 
        :math:`(1, C, H, W)`, :math:`(C, H, W)`, :math:`(H, W, C)` or :math:`(H, W)`.
        suptitle: title of entire figure. `supxlabel` and `supylabel` has same rules.
        titles: title of each subplot. `xlabels` and `ylabels` has same rules.
        xtick: enable axis ticks. `ytick` has same rules.
        border: show borders around of suplots.
        transpose: transpose grid axis.
    """
    assert not (not border and (xtick or ytick)), "`xtick` and `ytick` cannot be True when `border` is False."
    assert isiterable(images) and len(images) > 0 and all(map(isiterable, images)), "images must be a 2D list of images"
    assert any(len(row) > 0 for row in images), "images must contains at least one image"

    nrows = len(images)
    ncols = max(len(row) for row in images)
    if transpose:
        nrows, ncols = ncols, nrows

    fig, ax = plt.subplots(nrows, ncols, squeeze=False,
                           dpi=300, figsize=figsize,
                           gridspec_kw={"wspace": 0.01, "hspace": 0.01},
                           constrained_layout=True)
    fig.tight_layout()
    patch: mpRectangle = fig.patch  # type: ignore
    patch.set_facecolor("white")

    for ax_row in ax:  # type: ignore
        for ax_elem in ax_row:
            if not border:
                ax_elem.axis("off")
            else:
                if not xtick:
                    ax_elem.set_xticks([])
                if not ytick:
                    ax_elem.set_yticks([])

    if suptitle:
        fig.suptitle(suptitle, fontsize=suptitlesize)
    if supxlabel:
        fig.supxlabel(supxlabel, fontsize=suplabelsize)
    if supylabel:
        fig.supylabel(supylabel, fontsize=suplabelsize)

    elems: list[list[dict[str, np.ndarray | None]]] = [[
        {"img": None, "title": None, "xlabel": None, "ylabel": None}
        for _ in range(ncols)]
        for _ in range(nrows)]

    for R, row in _enumerate_nullable(images):
        for C, img in _enumerate_nullable(row):
            elem = elems[C][R] if transpose else elems[R][C]
            elem["img"] = as_f32image(as_numpy_image(img)).squeeze()

    for R, row in _enumerate_nullable(titles):
        for C, title in _enumerate_nullable(row):
            elem = elems[R][C]
            elem["title"] = title

    for R, row in _enumerate_nullable(xlabels):
        for C, xlabel in _enumerate_nullable(row):
            elem = elems[R][C]
            elem["xlabel"] = xlabel

    for R, row in _enumerate_nullable(ylabels):
        for C, ylabel in _enumerate_nullable(row):
            elem = elems[R][C]
            elem["ylabel"] = ylabel

    for R, row in enumerate(elems):
        for C, elem in enumerate(row):
            if elem["img"] is None:
                ax[R][C].set_visible(False)  # type: ignore
            else:
                ax[R][C].imshow(elem["img"])  # type: ignore
                if elem["title"] is not None:
                    ax[R][C].set_title(elem["title"], fontsize=titlesize)  # type: ignore
                if elem["xlabel"] is not None:
                    ax[R][C].set_xlabel(elem["xlabel"], fontsize=labelsize)  # type: ignore
                if elem["ylabel"] is not None:
                    ax[R][C].set_ylabel(elem["ylabel"], fontsize=labelsize)  # type: ignore

    return fig, ax
