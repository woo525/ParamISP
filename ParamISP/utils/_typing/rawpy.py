""" NOTE: Typing only definitions for rawpy module """
from typing import Union, Optional
from enum import Enum
from collections import namedtuple
import sys
import os

import numpy as np
import rawpy


ImageSizes = namedtuple("ImageSizes", [
    "raw_height", "raw_width", "height", "width",
    "top_margin", "left_margin", "iheight", "iwidth",
    "pixel_aspect", "flip",
])


class RawType(Enum):
    """ RAW image type. """

    Flat = 0
    """ Bayer type or black and white """

    Stack = 1
    """ Foveon type or sRAW/mRAW files or RawSpeed decoding """


class ThumbFormat(Enum):
    """ Thumbnail/preview image type. """

    JPEG = 1
    """ JPEG image as bytes object. """

    BITMAP = 2
    """ RGB image as ndarray object. """


Thumbnail = namedtuple("Thumbnail", ["format", "data"])


class RawPy(rawpy.RawPy):  # type: ignore
    """ Load RAW images, work on their data, and create a postprocessed (demosaiced) image.

    All operations are implemented using numpy arrays.

    NOTE: Typing only definitions for rawpy.Rawpy
    """

    def close(self):
        """
        Release all resources and close the RAW image.

        Consider using context managers for the same effect:

        .. code-block:: python

            with rawpy.imread("image.nef") as raw:
              # work with raw object

        """
        return super().close()

    @property
    def raw_type(self) -> RawType:
        """ Return the RAW type. """
        return super().raw_type

    @property
    def raw_image(self) -> np.ndarray:
        """ View of RAW image. Includes margin.

        For Bayer images, a 2D ndarray is returned.
        For Foveon and other RGB-type images, a 3D ndarray is returned.
        Note that there may be 4 color channels, where the 4th channel can be blank (zeros).

        Modifying the returned array directly influences the result of
        calling :meth:`~rawpy.RawPy.postprocess`.

        WARNING: The returned numpy array can only be accessed while this RawPy instance
            is not closed yet, that is, within a :code:`with` block or before calling :meth:`~rawpy.RawPy.close`.
            If you need to work on the array after closing the RawPy instance,
            make sure to create a copy of it with :code:`raw_image = raw.raw_image.copy()`.

        :rtype: ndarray of shape (h,w[,c])
        """
        return super().raw_image

    @property
    def raw_image_visible(self) -> np.ndarray:
        """ Like raw_image but without margin.

        :rtype: ndarray of shape (hv,wv[,c])
        """
        return super().raw_image_visible

    def raw_value(self, row: int, column: int) -> int:
        """ Return RAW value at given position relative to the full RAW image.
            Only usable for flat RAW images (see :attr:`~rawpy.RawPy.raw_type` property).
        """
        return super().raw_value(row, column)

    def raw_value_visible(self, row: int, column: int) -> int:
        """ Return RAW value at given position relative to visible area of image.
            Only usable for flat RAW images (see :attr:`~rawpy.RawPy.raw_type` property).
        """
        return super().raw_value_visible(row, column)

    @property
    def sizes(self) -> ImageSizes:
        """ Return a :class:`rawpy.ImageSizes` instance with size information of
            the RAW image and postprocessed image.
        """
        return super().sizes

    @property
    def num_colors(self) -> int:
        """ Number of colors.
            Note that e.g. for RGBG this can be 3 or 4, depending on the camera model,
            as some use two different greens.
        """
        return super().num_colors

    @property
    def color_desc(self) -> str:
        """ String description of colors numbered from 0 to 3 (RGBG,RGBE,GMCY, or GBTG).
            Note that same letters may not refer strictly to the same color.
            There are cameras with two different greens for example.
        """
        return super().color_desc

    def raw_color(self, row: int, column: int) -> int:
        """ Return color index for the given coordinates relative to the full RAW size.
            Only usable for flat RAW images (see raw_type property).
        """
        return super().raw_color(row, column)

    @property
    def raw_colors(self) -> np.ndarray:
        """ An array of color indices for each pixel in the RAW image.
        Equivalent to calling raw_color(y,x) for each pixel.
        Only usable for flat RAW images (see raw_type property).

        :rtype: ndarray of shape (h,w)
        """
        return super().raw_colors

    @property
    def raw_colors_visible(self) -> np.ndarray:
        """ Like raw_colors but without margin.

        :rtype: ndarray of shape (hv,wv)
        """
        return super().raw_colors_visible

    @property
    def raw_pattern(self) -> Optional[np.ndarray]:
        """ The smallest possible Bayer pattern of this image.

        :rtype: ndarray, or None if not a flat RAW image
        """
        return super().raw_pattern

    @property
    def camera_whitebalance(self) -> list[int]:
        """ White balance coefficients (as shot). Either read from file or calculated.

        :rtype: list of length 4
        """
        return super().camera_whitebalance

    @property
    def daylight_whitebalance(self) -> list[int]:
        """ White balance coefficients for daylight (daylight balance).
        Either read from file, or calculated on the basis of file data,
        or taken from hardcoded constants.

        :rtype: list of length 4
        """
        return super().daylight_whitebalance

    @property
    def black_level_per_channel(self) -> list[int]:
        """ Per-channel black level correction.

        :rtype: list of length 4
        """
        return super().black_level_per_channel

    @property
    def white_level(self) -> int:
        """ Level at which the raw pixel value is considered to be saturated. """
        return super().white_level

    @property
    def camera_white_level_per_channel(self) -> Optional[list[int]]:
        """ Per-channel saturation levels read from raw file metadata, if it exists. Otherwise None.

        :rtype: list of length 4, or None if metadata missing
        """
        return super().camera_white_level_per_channel

    @property
    def color_matrix(self) -> np.ndarray:
        """ Color matrix, read from file for some cameras, calculated for others.

        :rtype: ndarray of shape (3,4)
        """
        return super().color_matrix

    @property
    def rgb_xyz_matrix(self) -> np.ndarray:
        """ Camera RGB - XYZ conversion matrix.
        This matrix is constant (different for different models).
        Last row is zero for RGB cameras and non-zero for different color models (CMYG and so on).

        :rtype: ndarray of shape (4,3)
        """
        return super().rgb_xyz_matrix

    @property
    def tone_curve(self) -> np.ndarray:
        """ Camera tone curve, read from file for Nikon, Sony and some other cameras.

        :rtype: ndarray of length 65536
        """
        return super().tone_curve

    def extract_thumb(self) -> Thumbnail:
        """
        Extracts and returns the thumbnail/preview image (whichever is bigger)
        of the opened RAW image as :class:`rawpy.Thumbnail` object.
        For JPEG thumbnails, data is a bytes object and can be written as-is to file.
        For bitmap thumbnails, data is an ndarray of shape (h,w,c).
        If no image exists or the format is unsupported, an exception is raised.

        .. code-block:: python

            with rawpy.imread("image.nef") as raw:
              try:
                thumb = raw.extract_thumb()
              except rawpy.LibRawNoThumbnailError:
                print("no thumbnail found")
              except rawpy.LibRawUnsupportedThumbnailError:
                print("unsupported thumbnail")
              else:
                if thumb.format == rawpy.ThumbFormat.JPEG:
                  with open("thumb.jpg", "wb") as f:
                    f.write(thumb.data)
                elif thumb.format == rawpy.ThumbFormat.BITMAP:
                  imageio.imsave("thumb.tiff", thumb.data)

        :rtype: :class:`rawpy.Thumbnail`
        """
        return super().extract_thumb()

    def postprocess(self, params: "Params | None" = None, **kw) -> np.ndarray:
        """
        Postprocess the currently loaded RAW image and return the
        new resulting image as numpy array.

        :param rawpy.Params params:
            The parameters to use for postprocessing.
        :param **kw:
            Alternative way to provide postprocessing parameters.
            The keywords are used to construct a :class:`rawpy.Params` instance.
            If keywords are given, then `params` must be omitted.
        :rtype: ndarray of shape (h,w,c)
        """
        return super().postprocess(params, **kw)


class DemosaicAlgorithm(Enum):
    """ Identifiers for demosaic algorithms.

    NOTE: Typing only definitions for rawpy.DemosaicAlgorithm
    """
    LINEAR = 0
    VNG = 1
    PPG = 2
    AHD = 3
    DCB = 4
    # 5-9 only usable if demosaic pack GPL2 available
    MODIFIED_AHD = 5
    AFD = 6
    VCD = 7
    VCD_MODIFIED_AHD = 8
    LMMSE = 9
    # 10 only usable if demosaic pack GPL3 available
    AMAZE = 10
    # 11-12 only usable for LibRaw >= 0.16
    DHT = 11
    AAHD = 12

    @property
    def isSupported(self) -> bool:
        """
        Return True if the demosaic algorithm is supported, False if it is not,
        and None if the support status is unknown. The latter is returned if
        LibRaw < 0.15.4 is used or if it was compiled without cmake.

        The necessary information is read from the libraw_config.h header which
        is only written with cmake builds >= 0.15.4.
        """
        raise NotImplementedError

    def checkSupported(self) -> bool:
        """ Like :attr:`isSupported` but raises an exception for the `False` case. """
        raise NotImplementedError


class FBDDNoiseReductionMode(Enum):
    """ FBDD noise reduction modes. """
    Off = 0
    Light = 1
    Full = 2


class ColorSpace(Enum):
    """ Color spaces. """
    raw = 0
    sRGB = 1
    Adobe = 2
    Wide = 3
    ProPhoto = 4
    XYZ = 5


class HighlightMode(Enum):
    """ Highlight modes. """
    Clip = 0
    Ignore = 1
    Blend = 2
    ReconstructDefault = 5

    @classmethod
    def Reconstruct(self, level) -> int:
        """ :param int level: 3 to 9, low numbers favor whites, high numbers favor colors """
        raise NotImplementedError


class Params(rawpy.Params):  # type: ignore
    """ A class that handles postprocessing parameters.

    NOTE: Typing only definitions for rawpy.Params
    """

    def __init__(self, demosaic_algorithm: DemosaicAlgorithm | None = None, half_size: bool = False,
                 four_color_rgb: bool = False, dcb_iterations: int = 0, dcb_enhance: bool = False,
                 fbdd_noise_reduction: FBDDNoiseReductionMode = FBDDNoiseReductionMode.Off,
                 noise_thr: float | None = None, median_filter_passes: int = 0,
                 use_camera_wb: bool = False, use_auto_wb: bool = False, user_wb: list[int] | None = None,
                 output_color: ColorSpace = ColorSpace.sRGB, output_bps: int = 8,
                 user_flip: int | None = None, user_black: int | None = None, user_sat: int | None = None,
                 no_auto_bright: bool = False, auto_bright_thr: float | None = None, adjust_maximum_thr: float = 0.75,
                 bright: float = 1.0, highlight_mode: HighlightMode = HighlightMode.Clip,
                 exp_shift: float | None = None, exp_preserve_highlights: float = 0.0, no_auto_scale: bool = False,
                 gamma: tuple[float, float] | None = None, chromatic_aberration: tuple[float, float] | None = None,
                 bad_pixels_path: str | None = None):
        """
        If use_camera_wb and use_auto_wb are False and user_wb is None, then
        daylight white balance correction is used.
        If both use_camera_wb and use_auto_wb are True, then use_auto_wb has priority.

        :param rawpy.DemosaicAlgorithm demosaic_algorithm: default is AHD
        :param bool half_size: outputs image in half size by reducing each 2x2 block to one pixel
                               instead of interpolating
        :param bool four_color_rgb: whether to use separate interpolations for two green channels
        :param int dcb_iterations: number of DCB correction passes, requires DCB demosaicing algorithm
        :param bool dcb_enhance: DCB interpolation with enhanced interpolated colors
        :param rawpy.FBDDNoiseReductionMode fbdd_noise_reduction: controls FBDD noise reduction before demosaicing
        :param float noise_thr: threshold for wavelet denoising (default disabled)
        :param int median_filter_passes: number of median filter passes after demosaicing to reduce color artifacts
        :param bool use_camera_wb: whether to use the as-shot white balance values
        :param bool use_auto_wb: whether to try automatically calculating the white balance
        :param list user_wb: list of length 4 with white balance multipliers for each color
        :param rawpy.ColorSpace output_color: output color space
        :param int output_bps: 8 or 16
        :param int user_flip: 0=none, 3=180, 5=90CCW, 6=90CW,
                              default is to use image orientation from the RAW image if available
        :param int user_black: custom black level
        :param int user_sat: saturation adjustment (custom white level)
        :param bool no_auto_scale: Whether to disable pixel value scaling
        :param bool no_auto_bright: whether to disable automatic increase of brightness
        :param float auto_bright_thr: ratio of clipped pixels when automatic brighness increase is used
                                      (see `no_auto_bright`). Default is 0.01 (1%).
        :param float adjust_maximum_thr: see libraw docs
        :param float bright: brightness scaling
        :param highlight_mode: highlight mode
        :type highlight_mode: :class:`rawpy.HighlightMode` | int
        :param float exp_shift: exposure shift in linear scale.
                          Usable range from 0.25 (2-stop darken) to 8.0 (3-stop lighter).
        :param float exp_preserve_highlights: preserve highlights when lightening the image with `exp_shift`.
                          From 0.0 to 1.0 (full preservation).
        :param tuple gamma: pair (power,slope), default is (2.222, 4.5) for rec. BT.709
        :param tuple chromatic_aberration: pair (red_scale, blue_scale), default is (1,1),
                                           corrects chromatic aberration by scaling the red and blue channels
        :param str bad_pixels_path: path to dcraw bad pixels file. Each bad pixel will be corrected using
                                    the mean of the neighbor pixels. See the :mod:`rawpy.enhance` module
                                    for alternative repair algorithms, e.g. using the median.
        """
        super().__init__(demosaic_algorithm, half_size, four_color_rgb, dcb_iterations, dcb_enhance,
                         fbdd_noise_reduction, noise_thr, median_filter_passes, use_camera_wb, use_auto_wb,
                         user_wb, output_color, output_bps, user_flip, user_black, user_sat, no_auto_bright,
                         auto_bright_thr, adjust_maximum_thr, bright, highlight_mode, exp_shift,
                         exp_preserve_highlights, no_auto_scale, gamma, chromatic_aberration, bad_pixels_path)
