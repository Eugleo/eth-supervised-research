import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch as t


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_version(path: Path, version: Optional[int] = None) -> tuple[int, Path]:
    with open(path / ".version") as f:
        latest_version = int(f.read().strip())
        if version is not None and version < latest_version:
            return version, path / f"v{version}"
        return latest_version, path / f"v{latest_version}"


def next_version(path: Path) -> tuple[int, Path]:
    previous_version = 0 if not (path / ".version").exists() else get_version(path)[0]
    version = previous_version + 1
    with open(path / ".version", "w") as f:
        f.write(str(version))
    dir = path / f"v{version}"
    dir.mkdir(parents=True, exist_ok=True)
    return version, dir
