#!/usr/bin/env python3
"""
Execute every notebook headlessly and report which ones fail.

This is the reproducibility gate used by CI: each notebook is run top to
bottom with a fresh kernel (Agg backend) and any raised exception fails the
run. Executed copies are written to ``build/notebooks/`` by default so the
committed notebooks are never modified unless ``--inplace`` is given.

Usage::

    python scripts/run_notebooks.py                 # all notebooks
    python scripts/run_notebooks.py 17 21 --inplace # refresh outputs of two notebooks
    python scripts/run_notebooks.py --timeout 900 --keep-going
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = ROOT / "notebooks"


def select_notebooks(patterns: list[str]) -> list[Path]:
    all_nbs = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not patterns:
        return all_nbs
    chosen = []
    for nb in all_nbs:
        if any(nb.name.startswith(p) or p in nb.name for p in patterns):
            chosen.append(nb)
    return chosen


def run_one(path: Path, *, timeout: int, out_dir: Path | None, inplace: bool) -> tuple[bool, str, float]:
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    start = time.perf_counter()
    try:
        client.execute()
    except Exception as exc:  # nbclient raises CellExecutionError with the traceback
        message = str(exc).strip().splitlines()[-1] if str(exc).strip() else repr(exc)
        return False, message[:400], time.perf_counter() - start
    if inplace:
        nbformat.write(nb, path)
    elif out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, out_dir / path.name)
    return True, "", time.perf_counter() - start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "patterns", nargs="*", help="Notebook number prefixes or name fragments (default: all)."
    )
    parser.add_argument("--timeout", type=int, default=600, help="Per-cell timeout in seconds.")
    parser.add_argument(
        "--out-dir", default=str(ROOT / "build" / "notebooks"), help="Where executed copies go."
    )
    parser.add_argument("--inplace", action="store_true", help="Overwrite the source notebooks with outputs.")
    parser.add_argument("--keep-going", action="store_true", help="Run every notebook even after a failure.")
    args = parser.parse_args(argv)

    # The kernel inherits this environment. Force the inline backend (headless, but it
    # still captures figures as PNG outputs); a global MPLBACKEND=Agg would drop them.
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    notebooks = select_notebooks(args.patterns)
    if not notebooks:
        print("No notebooks matched.", file=sys.stderr)
        return 2

    failures: list[tuple[str, str]] = []
    for nb in notebooks:
        ok, message, seconds = run_one(
            nb, timeout=args.timeout, out_dir=Path(args.out_dir), inplace=args.inplace
        )
        status = "OK  " if ok else "FAIL"
        print(f"[{status}] {nb.name:<35} {seconds:6.1f}s {'' if ok else message}", flush=True)
        if not ok:
            failures.append((nb.name, message))
            if not args.keep_going:
                break

    print(f"\n{len(notebooks) - len(failures)}/{len(notebooks)} notebooks executed successfully.")
    for name, message in failures:
        print(f"  - {name}: {message}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
