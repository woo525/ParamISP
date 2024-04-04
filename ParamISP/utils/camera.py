import enum

import numpy as np
import numpy.typing as npt


class BayerPatternFlag(enum.IntFlag):
    """ Valid bayer pattern representation. """
    RGGB = 0b00
    GBRG = 0b01
    GRBG = 0b10
    BGGR = 0b11

    @classmethod
    def from_str(cls, s: str) -> "BayerPatternFlag":
        """ Convert a string to a BayerPatternFlag. """
        return cls[s.upper()]

    @classmethod
    def from_numpy(cls, bayer_pattern: npt.NDArray[np.int_]) -> "BayerPatternFlag":
        """ Convert the bayer pattern matrix to a flag. """
        bayer_pattern = np.asarray(bayer_pattern)
        assert bayer_pattern.shape == (2, 2), f"Invalid bayer pattern: {bayer_pattern}"

        match bayer_pattern.flatten().tolist():
            case [0, 1, 1, 2]: return cls.RGGB
            case [2, 1, 1, 0]: return cls.BGGR
            case [1, 0, 2, 1]: return cls.GRBG
            case [1, 2, 0, 1]: return cls.GBRG
            case _: raise ValueError(f"Invalid bayer pattern: {bayer_pattern}")

    @classmethod
    def numpy(cls, bayer_flag: "BayerPatternFlag") -> npt.NDArray[np.int_]:
        """ Convert the bayer pattern flag to a matrix. """
        match bayer_flag:
            case cls.RGGB: return np.array([[0, 1], [1, 2]])
            case cls.BGGR: return np.array([[1, 2], [0, 1]])
            case cls.GRBG: return np.array([[1, 0], [2, 1]])
            case cls.GBRG: return np.array([[2, 1], [1, 0]])
            case _: raise ValueError(f"Invalid bayer flag: {bayer_flag}")

    @classmethod
    def str_to_numpy(cls, s: str) -> npt.NDArray[np.int_]:
        """ Convert a string to a bayer pattern matrix. """
        return cls.numpy(cls.from_str(s))


def is_valid_bayer_pattern(bayer_pattern: npt.NDArray[np.int_]) -> bool:
    """ Check if the bayer pattern is valid. """
    try:
        BayerPatternFlag.from_numpy(bayer_pattern)
        return True
    except ValueError:
        return False


XYZ_TO_SRGB_D65_MATRIX: npt.NDArray[np.float64] = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.969266,  1.8760108,  0.0415560],
    [0.0556434, -0.2040259,  1.0572252],
])
""" XYZ -> sRGB color matrix."""


def normalize_whitebalance(white_balance: npt.NDArray[np.float_], dtype: type = np.float32) -> npt.NDArray[np.float_]:
    """ Normalize the white balance. 

    Args:
        white_balance: White balance value.
        dtype: Output array dtype, must be float32 or float64.

    Returns:
        Normalized white balance.
    """
    white_balance = white_balance / white_balance[..., 1:2]
    return white_balance.astype(dtype)


def normalize_colormatrix(color_matrix: npt.NDArray[np.float_], dtype: type = np.float32) -> npt.NDArray[np.float_]:
    """ Normalize the cameraRGB -> sRGB color matrix.
    See https://ninedegreesbelow.com/files/dcraw-c-code-annotated-code.html#E3 for more details. 

    Args:
        color_matrix: CameraRGB -> sRGB color matrix.
        dtype: Output matrix dtype, must be float32 or float64.

    Returns:
        Normalized cameraRGB -> sRGB color matrix.
    """
    color_matrix = color_matrix / np.mean(color_matrix, axis=-1, keepdims=True)
    return color_matrix.astype(dtype)


def compute_colormatrix(camera_matrix: npt.NDArray[np.float_], dtype: type = np.float32) -> npt.NDArray[np.float_]:
    """ Compute the cameraRGB -> sRGB color matrix from XYZ -> cameraRGB matrix. 

    Args:
        camera_matrix: XYZ -> cameraRGB color matrix.
        dtype: Output matrix dtype, must be float32 or float64.

    Returns:
        cameraRGB -> sRGB color matrix.
    """
    color_matrix = XYZ_TO_SRGB_D65_MATRIX @ np.linalg.inv(camera_matrix)
    return normalize_colormatrix(color_matrix, dtype)
