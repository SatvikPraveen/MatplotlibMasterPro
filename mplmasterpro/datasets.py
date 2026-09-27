"""
Deterministic synthetic datasets used across the notebooks, scripts and tests.

All generators are seeded and use the legacy :class:`numpy.random.RandomState`
stream so the CSV files committed under ``datasets/`` can be regenerated
byte-for-byte (see ``tests/test_datasets.py``). Run as a module to rebuild them::

    python -m mplmasterpro.datasets --out datasets
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "DATASETS",
    "DEFAULT_DATA_DIR",
    "generate_all",
    "generate_covid_cases",
    "generate_sales_data",
    "generate_stock_prices",
    "generate_weather_data",
    "load_dataset",
]

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"


def generate_sales_data(seed: int = 42) -> pd.DataFrame:
    """Monthly product-level unit sales and revenue for four products (48 rows)."""
    rng = np.random.RandomState(seed)
    months = pd.date_range(start="2023-01-01", periods=12, freq="ME")
    products = ["Laptop", "Tablet", "Smartphone", "Monitor"]
    rows = []
    for month in months:
        for product in products:
            units = rng.randint(50, 200)
            revenue = units * rng.randint(300, 1500)
            rows.append([month.strftime("%Y-%m"), product, units, revenue])
    return pd.DataFrame(rows, columns=["Month", "Product", "Units Sold", "Revenue"])


def generate_covid_cases(seed: int = 0) -> pd.DataFrame:
    """Cumulative daily case counts for four US states over 100 days (400 rows)."""
    rng = np.random.RandomState(seed)
    days = pd.date_range(start="2020-03-01", periods=100)
    states = ["California", "Texas", "New York", "Florida"]
    rows = []
    for state in states:
        base = rng.randint(50, 100)
        cases = base + rng.poisson(lam=100, size=100).cumsum()
        for i, date in enumerate(days):
            rows.append([date.strftime("%Y-%m-%d"), state, cases[i]])
    return pd.DataFrame(rows, columns=["Date", "State", "Cases"])


def generate_stock_prices(seed: int = 1) -> pd.DataFrame:
    """Daily OHLCV prices for three tickers over 60 trading days (180 rows)."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-01", periods=60)
    tickers = ["AAPL", "GOOG", "TSLA"]
    rows = []
    for ticker in tickers:
        price = rng.uniform(100, 500)
        for date in dates:
            open_price = price + rng.uniform(-5, 5)
            close_price = open_price + rng.uniform(-5, 5)
            high = max(open_price, close_price) + rng.uniform(0, 5)
            low = min(open_price, close_price) - rng.uniform(0, 5)
            volume = rng.randint(1_000_000, 5_000_000)
            rows.append(
                [
                    date.strftime("%Y-%m-%d"),
                    ticker,
                    round(open_price, 2),
                    round(close_price, 2),
                    round(high, 2),
                    round(low, 2),
                    volume,
                ]
            )
            price = close_price
    return pd.DataFrame(rows, columns=["Date", "Stock", "Open", "Close", "High", "Low", "Volume"])


def generate_weather_data(seed: int = 10) -> pd.DataFrame:
    """Daily temperature and humidity for four cities over 30 days (120 rows)."""
    rng = np.random.RandomState(seed)
    cities = ["New York", "San Francisco", "Austin", "Chicago"]
    dates = pd.date_range(start="2023-06-01", periods=30)
    rows = []
    for city in cities:
        temp = rng.uniform(15, 35, size=30)
        humidity = rng.uniform(40, 90, size=30)
        for i, date in enumerate(dates):
            rows.append([date.strftime("%Y-%m-%d"), city, round(temp[i], 1), round(humidity[i], 1)])
    return pd.DataFrame(rows, columns=["Date", "City", "Temperature (C)", "Humidity (%)"])


DATASETS: dict[str, tuple[Callable[[], pd.DataFrame], str, list[str]]] = {
    # name: (generator, description, date columns to parse on load)
    "sales_data": (generate_sales_data, "Monthly product-wise units sold and revenue", []),
    "covid_cases": (generate_covid_cases, "Cumulative COVID-19 case counts by US state", ["Date"]),
    "stock_prices": (generate_stock_prices, "Daily OHLCV prices for AAPL, GOOG and TSLA", ["Date"]),
    "weather_data": (generate_weather_data, "Daily city-level temperature and humidity", ["Date"]),
}


def generate_all(out_dir: str | Path = DEFAULT_DATA_DIR, *, verbose: bool = True) -> list[Path]:
    """Write every dataset to ``out_dir`` as CSV and return the written paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (generator, _, _) in DATASETS.items():
        path = out / f"{name}.csv"
        generator().to_csv(path, index=False)
        written.append(path)
        if verbose:
            print(f"wrote {path}")
    return written


def load_dataset(
    name: str,
    data_dir: str | Path | None = None,
    *,
    parse_dates: bool = True,
    regenerate_if_missing: bool = True,
) -> pd.DataFrame:
    """Load one of the bundled datasets by name (e.g. ``"sales_data"``).

    Parameters
    ----------
    name
        Dataset key from :data:`DATASETS` (with or without the ``.csv`` suffix).
    data_dir
        Directory containing the CSV files. Defaults to the repository ``datasets/`` folder.
    parse_dates
        Parse the dataset's date columns into ``datetime64``.
    regenerate_if_missing
        If the CSV is absent, build it in memory from the seeded generator instead of failing.
    """
    key = name.removesuffix(".csv")
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(DATASETS)}")
    generator, _, date_cols = DATASETS[key]
    path = Path(data_dir or DEFAULT_DATA_DIR) / f"{key}.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=date_cols if parse_dates else None)
    elif regenerate_if_missing:
        df = generator()
        if parse_dates:
            for col in date_cols:
                df[col] = pd.to_datetime(df[col])
    else:
        raise FileNotFoundError(path)
    return df


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the MatplotlibMasterPro datasets.")
    parser.add_argument("--out", default=str(DEFAULT_DATA_DIR), help="Output directory for CSVs.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file output.")
    args = parser.parse_args(argv)
    generate_all(args.out, verbose=not args.quiet)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
