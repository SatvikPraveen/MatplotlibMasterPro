# 🔧 Troubleshooting Guide

Common issues and solutions when working with MatplotlibMasterPro.

---

## 🖥️ Display & Backend Issues

### **Problem: Plots Not Showing**

**Symptoms:**
- Code runs without error but no plot appears
- `plt.show()` doesn't display anything

**Solutions:**

1. **Check your backend:**
   ```python
   import matplotlib
   print(matplotlib.get_backend())
   ```

2. **Set appropriate backend:**
   ```python
   # For Jupyter notebooks
   %matplotlib inline
   
   # For interactive plots in Jupyter
   %matplotlib notebook
   # or
   %matplotlib widget
   
   # For standalone scripts (GUI)
   import matplotlib
   matplotlib.use('TkAgg')  # or 'Qt5Agg', 'MacOSX'
   ```

3. **In JupyterLab, ensure ipympl is installed:**
   ```bash
   pip install ipympl
   jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib
   ```

### **Problem: "RuntimeError: main thread is not in main loop"**

**Solution:**
```python
# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

### **Problem: Blank/Empty Plots in Saved Files**

**Solution:**
- Call `plt.savefig()` **before** `plt.show()`
- Use `plt.close()` to free memory between plots
```python
plt.plot(x, y)
plt.savefig('myplot.png')  # Save first
plt.show()  # Then show
plt.close()  # Clean up
```

---

## 🎨 Font & Text Issues

### **Problem: "findfont: Font family not found"**

**Solutions:**

1. **Clear matplotlib cache:**
   ```bash
   rm -rf ~/.matplotlib/
   rm -rf ~/.cache/matplotlib/
   ```

2. **Rebuild font cache:**
   ```python
   import matplotlib.font_manager
   matplotlib.font_manager._rebuild()
   ```

3. **Use available fonts:**
   ```python
   import matplotlib.font_manager as fm
   fonts = sorted([f.name for f in fm.fontManager.ttflist])
   print(fonts[:20])  # List first 20 available fonts
   ```

### **Problem: LaTeX Rendering Errors**

**Solution:**
```python
# Disable LaTeX if not needed
plt.rcParams['text.usetex'] = False

# Or install LaTeX system-wide
# On macOS: brew install --cask mactex
# On Ubuntu: sudo apt-get install texlive-full
```

---

## 💾 Installation & Import Issues

### **Problem: "ModuleNotFoundError: No module named 'matplotlib'"**

**Solutions:**

1. **Ensure virtual environment is activated:**
   ```bash
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

2. **Install matplotlib:**
   ```bash
   pip install matplotlib
   ```

3. **Check Python version:**
   ```bash
   python --version  # Should be 3.9+
   ```

### **Problem: "ImportError: cannot import name 'Axes3D'"**

**Solution:**
```python
# Correct import for 3D plots
from mpl_toolkits.mplot3d import Axes3D  # Correct
# OR
from mpl_toolkits import mplot3d  # Also works
```

### **Problem: Jupyter Kernel Dies When Running Notebooks**

**Solutions:**

1. **Increase memory limit (if running in container):**
   ```bash
   jupyter notebook --NotebookApp.max_buffer_size=1000000000
   ```

2. **Restart kernel and clear outputs:**
   - Kernel → Restart & Clear Output

3. **Reduce data size or plot resolution:**
   ```python
   plt.rcParams['figure.dpi'] = 72  # Lower DPI
   ```

---

## 📊 Plot Quality & Display Issues

### **Problem: Plots Look Pixelated or Low Quality**

**Solution:**
```python
# Increase DPI
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# Or when saving
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
```

### **Problem: Labels Cut Off or Overlapping**

**Solutions:**

1. **Use tight_layout:**
   ```python
   plt.tight_layout()
   ```

2. **Adjust subplot parameters:**
   ```python
   plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
   ```

3. **When saving:**
   ```python
   plt.savefig('plot.png', bbox_inches='tight')
   ```

4. **Rotate labels:**
   ```python
   plt.xticks(rotation=45, ha='right')
   ```

### **Problem: Legend Outside Plot Area**

**Solution:**
```python
# Place legend outside plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Or shrink plot to accommodate legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
```

---

## 🐌 Performance Issues

### **Problem: Plotting is Very Slow**

**Solutions:**

1. **Reduce number of points:**
   ```python
   # Downsample data
   x_reduced = x[::10]  # Every 10th point
   y_reduced = y[::10]
   ```

2. **Use faster line drawing:**
   ```python
   plt.rcParams['path.simplify'] = True
   plt.rcParams['path.simplify_threshold'] = 1.0
   ```

3. **Turn off interactive mode:**
   ```python
   plt.ioff()  # Turn off interactive mode
   # ... create multiple plots ...
   plt.show()  # Show all at once
   ```

4. **Use Agg backend for batch processing:**
   ```python
   import matplotlib
   matplotlib.use('Agg')  # No GUI, faster
   ```

### **Problem: Memory Error with Many Plots**

**Solutions:**

1. **Close figures after use:**
   ```python
   plt.close('all')  # Close all figures
   plt.close(fig)    # Close specific figure
   ```

2. **Use context manager:**
   ```python
   with plt.rc_context():
       # Your plotting code
       pass
   # Settings automatically reset
   ```

---

## 🎨 Style & Theme Issues

### **Problem: "Style 'seaborn' not found"**

**Solution:**
```python
# Seaborn styles were renamed in recent versions
plt.style.use('seaborn-v0_8-darkgrid')  # Use versioned name
# Instead of: plt.style.use('seaborn-darkgrid')

# List available styles
print(plt.style.available)
```

### **Problem: Theme Changes Don't Apply**

**Solution:**
```python
# Reset to defaults first
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

# Then apply your theme
from utils.theme_utils import apply_dark_theme
apply_dark_theme()
```

---

## 📁 File & Path Issues

### **Problem: "FileNotFoundError" When Loading Data**

**Solutions:**

1. **Use absolute paths:**
   ```python
   from pathlib import Path
   PROJECT_ROOT = Path(__file__).resolve().parent.parent
   data_path = PROJECT_ROOT / "datasets" / "sales_data.csv"
   df = pd.read_csv(data_path)
   ```

2. **Check current working directory:**
   ```python
   import os
   print(os.getcwd())
   ```

### **Problem: Saved Plots Have Wrong Format**

**Solution:**
```python
# Explicitly specify format
plt.savefig('plot.png', format='png')
plt.savefig('plot.pdf', format='pdf')
plt.savefig('plot.svg', format='svg')
```

---

## 🔄 Animation Issues

### **Problem: "MovieWriter not available"**

**Solutions:**

1. **Install FFmpeg:**
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # Windows (use Chocolatey)
   choco install ffmpeg
   ```

2. **Verify installation:**
   ```python
   import matplotlib.animation as animation
   print(animation.writers.list())
   ```

3. **Use Pillow writer (no FFmpeg required):**
   ```python
   anim.save('animation.gif', writer='pillow', fps=30)
   ```

---

## 🐍 Jupyter-Specific Issues

### **Problem: Kernel Keeps Restarting**

**Solutions:**

1. **Check memory usage**
2. **Reduce plot complexity**
3. **Clear output cells regularly**
4. **Restart Jupyter server**

### **Problem: Interactive Widgets Not Working**

**Solutions:**

1. **Install ipywidgets:**
   ```bash
   pip install ipywidgets
   jupyter nbextension enable --py widgetsnbextension
   ```

2. **For JupyterLab:**
   ```bash
   jupyter labextension install @jupyter-widgets/jupyterlab-manager
   ```

---

## ❓ Still Having Issues?

### **Get Help:**

1. **Check matplotlib documentation:** https://matplotlib.org/stable/users/faq.html
2. **Search Stack Overflow:** https://stackoverflow.com/questions/tagged/matplotlib
3. **GitHub Issues:** https://github.com/matplotlib/matplotlib/issues
4. **Our GitHub Discussions:** https://github.com/SatvikPraveen/MatplotlibMasterPro/discussions

### **Provide This Information When Asking for Help:**

```python
import sys
import matplotlib
import numpy as np
import pandas as pd

print(f"Python: {sys.version}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Backend: {matplotlib.get_backend()}")
```

---

**Remember:** Most issues can be solved by:
1. ✅ Restarting the kernel
2. ✅ Checking virtual environment activation
3. ✅ Using `plt.tight_layout()`
4. ✅ Calling `plt.savefig()` before `plt.show()`
5. ✅ Clearing the matplotlib cache

Happy plotting! 📊
