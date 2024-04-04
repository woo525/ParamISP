from pathlib import Path
import argparse
from datetime import datetime

import pytorch_lightning as pl
import torch

import utils.path
import utils.camera
import data.modules


class Namespace(argparse.Namespace):
    def __init__(self, args=None, clone=False, /, **kwargs):
        super().__init__()
        if args is not None:
            if clone:
                self.__dict__.update(args.__dict__)
            else:
                self.__dict__ = args.__dict__
        for name in kwargs:
            setattr(self, name, kwargs[name])

    @classmethod
    def parse_args(cls, args: argparse.Namespace): ...

    @classmethod
    def help(cls, desc: str, default=None):
        if default is not None:
            desc += f" (default: {default})"
        return desc

    @classmethod
    def abspath(cls, *path) -> str:
        return Path(*path).absolute().as_posix()

    @classmethod
    def posixpath(cls, *path) -> str:
        return Path(*path).as_posix()

    @classmethod
    def strs(cls, expr: str) -> list[str]:
        return expr.split(",")

    @classmethod
    def indices(cls, expr: str) -> list[int]:
        indices = []
        for index in expr.split(","):
            try:
                if "-" in index:
                    start, end = index.split("-")
                    indices.extend(range(int(start), int(end)+1))
                else:
                    indices.append(int(index))
            except ValueError:
                raise ValueError(f"Invalid index: {index}")

        indices = sorted(set(indices))
        return indices


class Parser(argparse.ArgumentParser):
    def __init__(self, prog=None, usage=None, description=None, epilog=None, namespaces: list[type] = []):
        super().__init__(prog, usage, description, epilog)
        self.registered_namespaces = namespaces

    def parse_args(self, args=None, namespace=None) -> Namespace:
        args = super().parse_args(args, namespace or Namespace())

        def _get_bases(cls):
            bases = []
            for base in cls.__bases__:
                bases.extend(_get_bases(base))
            bases.append(cls)
            return bases

        targets = []
        for cls in self.registered_namespaces:
            targets.extend(_get_bases(cls))

        targets = list(dict.fromkeys(targets))  # remove duplicates

        for cls in targets:
            if callable(parse_args := getattr(cls, "parse_args", None)):
                parse_args(args)

        return args


class CameraModels(Namespace):
    camera_models: list[data.modules.CameraModel]

    @classmethod
    def add_to(
        cls, parser: argparse.ArgumentParser,
        default: str | None = ",".join(data.modules.EVERY_CAMERA_MODEL),
    ):
        parser.add_argument("--camera", metavar="[NAME,...]", type=cls.strs, dest="camera_models",
                            default=default, required=default is None,
                            help=cls.help("Camera models to use, seperated by commas", default))


class CameraModel(Namespace):
    camera_model: data.modules.CameraModel

    @classmethod
    def add_to(
        cls, parser: argparse.ArgumentParser,
        default: data.modules.CameraModel | None = None,
    ):
        parser.add_argument("--camera", metavar="NAME", choices=data.modules.EVERY_CAMERA_MODEL, dest="camera_model",
                            default=default, required=default is None,
                            help=cls.help("Camera model to use", default))


class CropType(Namespace):
    crop_type: data.modules.CropType

    @classmethod
    def add_to(
        cls, parser: argparse.ArgumentParser,
        default: data.modules.CropType = "full",
    ):
        parser.add_argument("--crop-type", metavar="TYPE", choices=data.modules.EVERY_CROP_TYPE, default=default,
                            help=cls.help("Crop type to use", default))


class CropSize(Namespace):
    crop_size: int

    @classmethod
    def add_to(
        cls, parser: argparse.ArgumentParser,
        default: int = 512,
    ):
        parser.add_argument("--crop-size", metavar="N", type=int, default=default,
                            help=cls.help("Crop size of patches, ignored if `crop_type` is 'full'", default))


class BayerPattern(Namespace):
    bayer_pattern: str | None

    @classmethod
    def add_to(
        cls, parser: argparse.ArgumentParser,
        default: str | None = "RGGB",
    ):
        parser.add_argument("--bayer-pattern", metavar="CCCC", choices=["RGGB", "GRBG", "GBRG", "BGGR"], default=default,
                            help=cls.help("Bayer pattern to use", default))


class UseExtraMetadata(Namespace):
    use_extra_metadata: bool

    @classmethod
    def add_to(
        cls, parser: argparse.ArgumentParser,
        default: bool = False,
    ):
        if default:
            parser.add_argument("--no-extra", action="store_false", dest="use_extra_metadata",
                                help=cls.help("Ignore extra data"))
        else:
            parser.add_argument("--use-extra", action="store_true", dest="use_extra_metadata",
                                help=cls.help("Use extra data"))


class NumSamples(Namespace):
    num_samples: int

    @classmethod
    def add_to(
        cls, parser: argparse.ArgumentParser,
        default: int = 1024,
    ):
        parser.add_argument("-l", "--num-samples", metavar="N", type=int, default=default,
                            help=cls.help("Sampling size for each training epoch", default))


class SampleIndices(Namespace):
    sample_indices: list[int]

    @classmethod
    def add_to(
        cls, parser: argparse.ArgumentParser,
    ):
        parser.add_argument("--idx", metavar="[N,...]", type=cls.indices, default=[], dest="sample_indices",
                            help=cls.help("Sample indices to use, seperated by commas or a range, e.g. 1,2,3 or 1-3"))


class BatchSize(Namespace):
    batch_size: int

    @classmethod
    def add_to(
        cls, parser: argparse.ArgumentParser,
        default: int = 4,
    ):
        parser.add_argument("-b", "--batch-size", metavar="N", type=int, default=default,
                            help=cls.help("Batch size for training", default))


class NumWorkers(Namespace):
    num_workers: int

    @classmethod
    def add_to(
        cls, parser: argparse.ArgumentParser,
        default: int = 4,
    ):
        parser.add_argument("-j", "--num-workers", metavar="N", type=int, default=default,
                            help=cls.help("Number of workers for data loading", default))


class DDPMode(Namespace):
    ddp: bool
    strategy: str | None
    replace_sampler_ddp: bool | None

    @classmethod
    def add_to(
        cls, parser: argparse.ArgumentParser,
        default: bool = True,
    ):
        if default:
            parser.add_argument("--no-ddp", action="store_false", dest="ddp",
                                help=cls.help("Disable DDP mode"))
        else:
            parser.add_argument("--ddp", action="store_true",
                                help=cls.help("Enable DDP mode"))

    @classmethod
    def parse_args(cls, args: Namespace):
        if hasattr(args, "ddp") and args.ddp:
            args.strategy = "ddp"
            args.replace_sampler_ddp = False


class Seed(Namespace):
    seed: int | None
    deterministic: bool | None

    @classmethod
    def add_to(
        cls, parser: argparse.ArgumentParser,
        default: int | None = 42,
    ):
        parser.add_argument("--seed", metavar="N", type=int, default=default,
                            help=cls.help("Random seed", default))

    @classmethod
    def parse_args(cls, args: Namespace):
        if hasattr(args, "seed") and args.seed is not None:
            pl.seed_everything(args.seed)
            # args.deterministic = True


class MaxEpochs(Namespace):
    max_epochs: int

    @classmethod
    def add_to(
        cls, parser: argparse.ArgumentParser,
        default: int = -1,
    ):
        parser.add_argument("--max-epochs", metavar="N", type=int, default=default,
                            help=cls.help("Maximum number of training epochs", default))


class AdamConfig(Namespace):
    lr: float
    weight_decay: float

    @classmethod
    def add_to(
        cls, parser: argparse.ArgumentParser,
        lr: float = 1e-4,
        wd: float = 1e-12,
    ):
        parser.add_argument("--lr", metavar="F", type=float, default=lr,
                            help=cls.help("Learning rate", lr))
        parser.add_argument("--wd", metavar="F", type=float, default=wd, dest="weight_decay",
                            help=cls.help("Weight decay", wd))


class StepLRConfig(Namespace):
    lr_gamma: float
    lr_step: int

    @classmethod
    def add_to(
        cls, parser: argparse.ArgumentParser,
        gamma: float = 0.8,
        step:  int   = 10,
    ):
        parser.add_argument("--lr-gamma", metavar="F", type=float, default=gamma,
                            help=cls.help("Learning rate decay factor", gamma))
        parser.add_argument("--lr-step", metavar="N", type=int, default=step,
                            help=cls.help("Learning rate decay step", step))


class RunsPath(Namespace):
    runs_name: str
    runs_root: str

    @classmethod
    def add_to(
        cls, parser: argparse.ArgumentParser,
        name: str | None = None,
        root: str = (utils.path.projroot/"runs").as_posix(),
    ):
        parser.add_argument("-o", "--runs-name", metavar="NAME", default=name, required=name is None,
                            help=cls.help("Name of the output runs folder", name))
        parser.add_argument("-O", "--runs-root", metavar="PATH", type=cls.abspath, default=root,
                            help=cls.help("Parent path of the runs directory", root))

    @property
    def runs_path(self) -> Path:
        return Path(self.runs_root, self.runs_name)


class ResumePath(Namespace):
    resume_name: str | None

    @classmethod
    def add_to(cls, parser: argparse.ArgumentParser):
        parser.add_argument("--resume", metavar="NAME", dest="resume_name",
                            help=cls.help("Resume training from a checkpoint"))

    @property
    def resume_path(self) -> str | None:
        if not hasattr(self, "resume_name") or self.resume_name is None:
            return None
        runs_path = RunsPath(self).runs_path
        return self.posixpath(runs_path, f"checkpoints/{self.resume_name}.ckpt")


class CheckpointPaths(Namespace):
    ckpt_names: list[str]

    @classmethod
    def add_to(
        cls, parser: argparse.ArgumentParser,
        default: list[str] = ["best-*"],
    ):
        parser.add_argument("--ckpt", metavar="[NAME,...]", type=cls.strs, default=default, dest="ckpt_names",
                            help=cls.help("Checkpoint name to load"))

    @classmethod
    def parse_args(cls, args: Namespace):
        if hasattr(args, "ckpt_names"):

            runs_path = RunsPath(args).runs_path
            checkpoints_path: Path = Path(runs_path, "checkpoints")
            ckpt_names: list[str] = []
            for ckpt_name in args.ckpt_names:
                if (checkpoints_path/f"{ckpt_name}.ckpt").exists():
                    ckpt_names.append(ckpt_name)
                elif len(ckpt_paths := list(checkpoints_path.glob(f"{ckpt_name}.ckpt"))) > 0:
                    ckpt_names.extend([ckpt_path.stem for ckpt_path in ckpt_paths])
                elif "weights" in ckpt_name: # using official weights
                    ckpt_names.append(ckpt_name)
                else:
                    raise FileNotFoundError(f"Checkpoint '{ckpt_name}' not found")
            ckpt_names = sorted(set(ckpt_names), reverse=True)
            args.ckpt_names = ckpt_names

    @property
    def ckpt_paths(self) -> list[Path]:
        runs_path = RunsPath(self).runs_path
        for name in self.ckpt_names:
            if "weights" in name: # using official weights
                return [Path("/workspace", name)]
            else:
                return [Path(runs_path, f"checkpoints/{name}.ckpt")]

class SaveConfig(Namespace):
    save_best: int
    save_freq: int
    early_stop: int

    @classmethod
    def add_to(
        cls, parser: argparse.ArgumentParser,
        best: int = 1,
        freq: int = 100,
        early_stop: int = 100,
    ):
        parser.add_argument("--save-best", metavar="N", type=int, default=best,
                            help=cls.help("Number of best checkpoints to save", best))
        parser.add_argument("--save-freq", metavar="N", type=int, default=freq,
                            help=cls.help("Frequency of saving checkpoints", freq))
        parser.add_argument("--early-stop", metavar="N", type=int, default=early_stop,
                            help=cls.help("Number of epochs to wait before early stopping", early_stop))


class PretrainedPath(Namespace):
    pretrained_name: str | None

    @classmethod
    def add_to(cls, parser: argparse.ArgumentParser):
        parser.add_argument("--pretrained", metavar="NAME", dest="pretrained_name",
                            help=cls.help("Pretrained model name"))

    @classmethod
    def parse_args(cls, args: Namespace):
        if hasattr(args, "pretrained_name") and args.pretrained_name is not None:
            splits = args.pretrained_name.split("/")
            if len(splits) <= 0:
                args.pretrained_name = ""
            elif len(splits) == 1:
                runs_name = RunsPath(args).runs_name
                args.pretrained_name = "/".join([runs_name, "checkpoints", *splits])
            elif splits[0] == "checkpoints":
                runs_name = RunsPath(args).runs_name
                args.pretrained_name = "/".join([runs_name, *splits])
            elif any(tmp in "checkpoints" for tmp in splits): #$
                args.pretrained_name = "/".join([*splits])
            elif splits[1] != "checkpoints":
                args.pretrained_name = "/".join([splits[0], "checkpoints", *splits[1:]])

    @property
    def pretrained_path(self) -> Path | None:
        if not hasattr(self, "pretrained_name") or not self.pretrained_name:
            return None
        runs_root = RunsPath(self).runs_root
        return Path(runs_root, f"{self.pretrained_name}.ckpt")


class PreviousStageWeightPath(Namespace):
    prev_stage_weight_name: str

    @classmethod
    def add_to(cls, parser: argparse.ArgumentParser):
        parser.add_argument("--prev-stage", metavar="NAME", dest="prev_stage_weight_name",
                            help=cls.help("Checkpoint name to load as previous stage weights"))

    @classmethod
    def parse_args(cls, args: Namespace):
        if hasattr(args, "prev_stage_weight_name") and args.prev_stage_weight_name is not None:
            splits = args.prev_stage_weight_name.split("/")
            if len(splits) <= 0:
                args.prev_stage_weight_name = ""
            elif len(splits) == 1:
                runs_name = RunsPath(args).runs_name
                args.prev_stage_weight_name = "/".join([runs_name, "checkpoints", *splits])
            elif splits[0] == "checkpoints":
                runs_name = RunsPath(args).runs_name
                args.prev_stage_weight_name = "/".join([runs_name, *splits])
            elif splits[1] != "checkpoints":
                args.prev_stage_weight_name = "/".join([splits[0], "checkpoints", *splits[1:]])

    @property
    def prev_stage_weight_path(self) -> Path | None:
        if not hasattr(self, "prev_stage_weight_name") or not self.prev_stage_weight_name:
            return None
        runs_root = RunsPath(self).runs_root
        return Path(runs_root, f"{self.prev_stage_weight_name}.ckpt")


class Profiler(Namespace):
    profiler: str | None

    @classmethod
    def add_to(
        cls, parser: argparse.ArgumentParser,
        default: str | None = None,
    ):
        parser.add_argument("--profiler", metavar="NAME", choices=["simple", "advanced", "pytorch"], default=default,
                            help=cls.help("Enable profiler", default))


class ProcessArgs(Namespace):
    timestamp: str
    gpus: int

    @classmethod
    def parse_args(cls, args: Namespace):
        if not hasattr(args, "timestamp") or args.timestamp is None:
            args.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        args.gpus = torch.cuda.device_count()


class Mode(Namespace):
    mode: str

    @classmethod
    def add_subparsers(cls, parser: argparse.ArgumentParser):
        return parser.add_subparsers(dest="mode", required=True, help="Running mode")
