"""Run a function in a fresh Python subprocess that imports only that function's module.

Why: PyTorch (neuralforecast, Chronos) and LightGBM each load their own OpenMP runtime
(``torch/lib/libomp.dylib`` vs the ``libomp`` LightGBM links against). With both in one process on
macOS we observed segfaults (torch first) and deadlocks (LightGBM first); ``OMP_NUM_THREADS`` and
``KMP_DUPLICATE_LIB_OK`` did not fix it. PyTorch-based tiers therefore run in a child process and
import torch only there, so the calling process (backtest runner, test suite) never loads it.

``multiprocessing``'s ``spawn`` is **not** enough: it re-imports the calling script's top-level
imports in the child, so a script that imports the LightGBM tier would still load LightGBM next to
torch there. Instead the child is started as ``python -m forecasting_platform.isolation``, and the
call (function reference + arguments) and its result travel through temporary pickle files.
"""

import importlib
import os
import pickle
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

# The child must import this package even if the editable-install .pth is skipped (macOS can flag
# it hidden, which Python 3.13 ignores), so put src/ on its PYTHONPATH explicitly.
_SRC_DIR = str(Path(__file__).resolve().parents[1])


def run_isolated(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn(*args, **kwargs)`` in a fresh subprocess and return its (pickled) result.

    ``fn`` must be a module-level function. The child's stderr is passed through; a non-zero exit
    (including a crash) raises ``RuntimeError``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        call_path, result_path = Path(tmp) / "call.pkl", Path(tmp) / "result.pkl"
        with call_path.open("wb") as fh:
            pickle.dump((fn.__module__, fn.__qualname__, args, kwargs), fh)
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(filter(None, [_SRC_DIR, os.environ.get("PYTHONPATH")])),
        }
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "forecasting_platform.isolation",
                str(call_path),
                str(result_path),
            ],
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Isolated call {fn.__module__}.{fn.__qualname__} failed "
                f"(exit code {proc.returncode}); see its stderr above."
            )
        with result_path.open("rb") as fh:
            return pickle.load(fh)


def _main(call_path: str, result_path: str) -> None:
    with open(call_path, "rb") as fh:
        module_name, qualname, args, kwargs = pickle.load(fh)
    fn = importlib.import_module(module_name)
    for part in qualname.split("."):
        fn = getattr(fn, part)
    result = fn(*args, **kwargs)
    with open(result_path, "wb") as fh:
        pickle.dump(result, fh)


if __name__ == "__main__":
    _main(sys.argv[1], sys.argv[2])
