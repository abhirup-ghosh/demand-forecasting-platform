"""Tests for the subprocess isolation helper."""

import operator
import os

import pandas as pd
import pytest

from forecasting_platform.isolation import run_isolated


def test_returns_result_including_dataframes() -> None:
    assert run_isolated(operator.add, 2, 3) == 5
    df = pd.DataFrame({"a": [1, 2]})
    out = run_isolated(pd.concat, [df, df], ignore_index=True)
    pd.testing.assert_frame_equal(out, pd.DataFrame({"a": [1, 2, 1, 2]}))


def test_child_crash_raises_instead_of_hanging() -> None:
    with pytest.raises(RuntimeError, match=r"abort.*exit code"):
        run_isolated(os.abort)
