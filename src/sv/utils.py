import os
import random
from pathlib import Path

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


def current_version(path: Path) -> tuple[int, Path]:
    with open(path / ".version") as f:
        version = int(f.read().strip())
        return version, path / f"v{version}"


def next_version(path: Path) -> tuple[int, Path]:
    previous_version = (
        0 if not (path / ".version").exists() else current_version(path)[0]
    )
    version = previous_version + 1
    with open(path / ".version", "w") as f:
        f.write(str(version))
    dir = path / f"v{version}"
    dir.mkdir(parents=True, exist_ok=True)
    return version, dir
