from pathlib import Path


projroot = Path(__file__).parent.parent


def purify(*path: str | Path, absolute: bool = False) -> str:
    """ Purify a path as posix string.

    Args:
        path: target path; will be joined if multiple paths are given.
        absolute: if True, return an absolute path.
    """
    path = tuple(filter(lambda x: x, path))
    if len(path) == 0:
        return ""

    path_: Path = Path(*path)
    if absolute:
        path_ = path_.absolute()
    return path_.as_posix()
