# Examples

Minimal, copy-paste-ready scripts (10–50 lines each). They run from the repository root and write their output next to the file or into `exports/`.

| Example | Lines | Shows |
| --- | --- | --- |
| `quick_start.py` | ~20 | The first plot: figure, axes, labels, save |
| `publication_figure.py` | ~35 | `apply_publication_theme()` and a 600 dpi PDF |
| `batch_process.py` | ~55 | Loop over every CSV in `datasets/` and plot each |
| `custom_theme_example.py` | ~45 | Switching between dark, minimal, corporate and colour-blind themes |
| `animation_example.py` | ~30 | A `FuncAnimation` in a few lines |

```bash
python examples/quick_start.py
```

For the research-oriented helpers (confidence bands, raincloud plots, journal sizing, CVD checks) see the "Sixty-second tour" in the top-level README and `docs/API.md`.

All examples are executed in CI, so they are guaranteed to run against the current package.
