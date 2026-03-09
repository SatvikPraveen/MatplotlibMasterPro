# 🧪 Tests

This folder contains **unit tests** for the utility modules, demonstrating professional testing practices.

## 🎯 Test Coverage

### `test_plot_utils.py`
Tests for plotting utility functions in `utils/plot_utils.py`:
- Line plots
- Bar charts
- Scatter plots
- Histograms
- Pie charts
- Multi-line plots
- Grouped bar plots

### `test_theme_utils.py`
Tests for theme utilities in `utils/theme_utils.py`:
- Theme application
- Color palette generation
- rcParams modification
- Theme reset functionality

---

## 🚀 Running Tests

### Run all tests:
```bash
python -m pytest tests/ -v
```

### Run specific test file:
```bash
python -m pytest tests/test_plot_utils.py -v
python -m pytest tests/test_theme_utils.py -v
```

### Run with coverage:
```bash
python -m pytest tests/ --cov=utils --cov-report=html
```

---

## 📦 Requirements

Install testing dependencies:
```bash
pip install pytest pytest-cov
```

Or if using the dev requirements:
```bash
pip install -r requirements_dev.txt
```

---

## 💡 Testing Philosophy

These tests demonstrate:
- ✅ **Function validation** - Ensure utilities work as expected
- ✅ **Error handling** - Test edge cases and invalid inputs
- ✅ **Figure creation** - Verify plots are generated correctly
- ✅ **Professional practices** - Industry-standard testing patterns

---

## 🎓 Learning Value

The tests serve as:
- **Documentation** - Show how to use each utility function
- **Examples** - Demonstrate proper function calls
- **Quality assurance** - Catch bugs before they reach production
- **Portfolio material** - Show testing knowledge to employers
