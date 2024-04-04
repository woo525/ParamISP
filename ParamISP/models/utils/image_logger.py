from typing import Any, Dict, Optional, Union
import os
from argparse import Namespace
import logging

import torch
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.logger import _add_prefix, _convert_params
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn

import utils.io

log = logging.getLogger(__name__)


class ImageWriter:
    """ Image Writer for CSVLogger.

    Args:
        log_dir: Directory for the experiment logs.
    """

    NAME_HPARAMS_FILE = "hparams.yaml"

    def __init__(self, log_dir: str) -> None:
        self.hparams = {}
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log_hparams(self, params: Dict[str, Any]) -> None:
        """Record hparams."""
        self.hparams.update(params)

    def log_image(self, tiff: torch.Tensor, name: str):
        suffix = name.split(".")[-1]
        match suffix:
            case "tif" | "tiff":
                utils.io.savetiff(tiff, os.path.join(self.log_dir, name), u16=True)
            case "png" | "jpg":
                utils.io.saveimg(tiff, os.path.join(self.log_dir, name))
            case _:
                raise ValueError(f"Unknown suffix {suffix}")


class ImageLogger(LightningLoggerBase):
    """ Log image. """

    def __init__(self, save_dir: str, name: Optional[str] = "lightning_logs", version: Optional[Union[int, str]] = None):
        super().__init__()
        self._save_dir = save_dir
        self._name = name or ""
        self._version = version
        self._experiment = None

    @property
    def root_dir(self) -> str:
        return os.path.join(self.save_dir, self.name)  # type: ignore

    @property
    def log_dir(self) -> str:
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        return log_dir

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    @property
    @rank_zero_experiment
    def experiment(self) -> ImageWriter:
        r"""

        Actual ExperimentWriter object. To use ExperimentWriter features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment:
            return self._experiment

        os.makedirs(self.root_dir, exist_ok=True)
        self._experiment = ImageWriter(log_dir=self.log_dir)
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        self.experiment.log_hparams(params)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        pass

    @rank_zero_only
    def save(self) -> None:
        pass

    @rank_zero_only
    def finalize(self, status: str) -> None:
        pass

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> int:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version  # type: ignore

    def _get_next_version(self):
        root_dir = self.root_dir

        if not os.path.isdir(root_dir):
            log.warning("Missing logger folder: %s", root_dir)
            return 0

        existing_versions = []
        for d in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("version_"):
                existing_versions.append(int(d.split("_")[1]))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1
