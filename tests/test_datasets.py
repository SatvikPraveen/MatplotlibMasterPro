"""Reproducibility tests: the seeded generators must match the committed CSV files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from mplmasterpro import datasets

REPO_DATA = Path(__file__).resolve().parent.parent / "datasets"


@pytest.mark.parametrize("name", sorted(datasets.DATASETS))
def test_generator_matches_committed_csv(name: str) -> None:
    """Regenerating a dataset must reproduce the committed CSV byte-for-byte after a CSV round-trip."""
    generator = datasets.DATASETS[name][0]
    generated = generator()
    on_disk = pd.read_csv(REPO_DATA / f"{name}.csv")
    pd.testing.assert_frame_equal(generated.astype(str), on_disk.astype(str))


@pytest.mark.parametrize("name", sorted(datasets.DATASETS))
def test_generators_are_deterministic(name: str) -> None:
    generator = datasets.DATASETS[name][0]
    pd.testing.assert_frame_equal(generator(), generator())


def test_load_dataset_parses_dates() -> None:
    df = datasets.load_dataset("covid_cases")
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    assert set(df["State"]) == {"California", "Texas", "New York", "Florida"}


def test_load_dataset_accepts_suffix_and_unknown_name() -> None:
    assert datasets.load_dataset("sales_data.csv").shape == (48, 4)
    with pytest.raises(KeyError):
        datasets.load_dataset("nope")


def test_load_dataset_regenerates_when_missing(tmp_path: Path) -> None:
    df = datasets.load_dataset("weather_data", data_dir=tmp_path)
    assert len(df) == 120
    with pytest.raises(FileNotFoundError):
        datasets.load_dataset("weather_data", data_dir=tmp_path, regenerate_if_missing=False)


def test_generate_all_and_cli(tmp_path: Path) -> None:
    written = datasets.generate_all(tmp_path, verbose=False)
    assert {p.name for p in written} == {f"{n}.csv" for n in datasets.DATASETS}
    out = tmp_path / "cli"
    result = subprocess.run(
        [sys.executable, "-m", "mplmasterpro.datasets", "--out", str(out), "--quiet"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_DATA.parent,
    )
    assert result.returncode == 0
    assert sorted(p.name for p in out.glob("*.csv")) == sorted(f"{n}.csv" for n in datasets.DATASETS)
