"""Shared pytest configuration: headless backend, rcParams isolation, sample data."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # must run before pyplot is imported anywhere

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mplmasterpro import datasets


@pytest.fixture(autouse=True)
def _isolate_matplotlib_state():
    """Restore rcParams and close every figure after each test."""
    matplotlib.rcParams["figure.max_open_warning"] = 0
    with matplotlib.rc_context():
        yield
    plt.close("all")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def sales_df() -> pd.DataFrame:
    """The bundled sales dataset, regenerated in memory (no disk dependency)."""
    return datasets.generate_sales_data()


@pytest.fixture
def monthly_df(sales_df: pd.DataFrame) -> pd.DataFrame:
    return sales_df.groupby("Month", as_index=False)[["Units Sold", "Revenue"]].sum()
