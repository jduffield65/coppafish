import os
import psutil
import subprocess
import numpy as np
from pathlib import PurePath


def get_software_verison() -> str:
    """
    Get coppafish's version tag written in _version.py

    Returns:
        str: software version.
    """
    with open(PurePath(os.path.dirname(os.path.realpath(__file__))).parent.joinpath("_version.py"), "r") as f:
        version_tag = f.read().split("'")[1]
    return version_tag


def get_git_revision_hash() -> str:
    """
    Get the latest git commit full hash.

    Returns:
        str: commit hash.
    """
    return (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PurePath(os.path.dirname(os.path.realpath(__file__))).parent
        )
        .decode("ascii")
        .strip()
    )


def get_available_memory() -> float:
    """
    Get system's available memory at the time of calling this function.

    Returns:
        float: available memory in GB.
    """
    return psutil.virtual_memory().available / 1e9


def get_core_count() -> int:
    """
    Get the number of threads available for multiprocessing tasks on the system.

    Returns:
        int: number of available threads.
    """
    n_threads = psutil.cpu_count(logical=True)
    if n_threads is None:
        n_threads = 1
    else:
        n_threads -= 2
    n_threads = np.clip(n_threads, 1, 999, dtype=int)

    return int(n_threads)
