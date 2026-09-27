"""Run a function in a fresh ``spawn``-ed Python process.

Why: PyTorch (neuralforecast, Chronos) and LightGBM each load their own OpenMP runtime
(``torch/lib/libomp.dylib`` vs the ``libomp`` LightGBM links against). With both in one process on
macOS we observed segfaults (torch first) and deadlocks (LightGBM first); ``OMP_NUM_THREADS`` and
``KMP_DUPLICATE_LIB_OK`` did not fix it. PyTorch-based tiers therefore run in a child process and
import torch only there, so the calling process (backtest runner, test suite) never loads it.
"""

import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any


def run_isolated(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn(*args, **kwargs)`` in a fresh spawned process and return its (pickled) result.

    ``fn`` must be importable at module level (spawn re-imports it by name).
    """
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        return pool.submit(fn, *args, **kwargs).result()
