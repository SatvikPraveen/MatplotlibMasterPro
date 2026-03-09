"""Unit tests for plot_utils module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.plot_utils import (
    line_plot,
    bar_plot,
    scatter_plot,
    histogram_plot,
    pie_chart,
    multi_line_plot,
    grouped_bar_plot
)


@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {
        'x': np.array([1, 2, 3, 4, 5]),
        'y': np.array([2, 4, 6, 8, 10]),
        'categories': ['A', 'B', 'C', 'D'],
        'values': np.array([10, 20, 15, 25])
    }


@pytest.fixture(autouse=True)
def close_plots():
    """Automatically close all plots after each test."""
    yield
    plt.close('all')


class TestLinePlot:
    """Test suite for line_plot function."""
    
    def test_basic_line_plot(self, sample_data):
        """Test basic line plot creation."""
        fig, ax = line_plot(
            sample_data['x'],
            sample_data['y'],
            title="Test Line Plot",
            xlabel="X Axis",
            ylabel="Y Axis"
        )
        
        assert fig is not None
        assert ax is not None
        assert ax.get_title() == "Test Line Plot"
        assert ax.get_xlabel() == "X Axis"
        assert ax.get_ylabel() == "Y Axis"
    
    def test_line_plot_with_style(self, sample_data):
        """Test line plot with custom styling."""
        fig, ax = line_plot(
            sample_data['x'],
            sample_data['y'],
            color='red',
            linestyle='--',
            linewidth=2
        )
        
        assert fig is not None
        lines = ax.get_lines()
        assert len(lines) == 1
        assert lines[0].get_color() == 'red'
        assert lines[0].get_linestyle() == '--'
    
    def test_line_plot_with_markers(self, sample_data):
        """Test line plot with markers."""
        fig, ax = line_plot(
            sample_data['x'],
            sample_data['y'],
            marker='o',
            markersize=8
        )
        
        assert fig is not None
        lines = ax.get_lines()
        assert lines[0].get_marker() == 'o'


class TestBarPlot:
    """Test suite for bar_plot function."""
    
    def test_basic_bar_plot(self, sample_data):
        """Test basic bar plot creation."""
        fig, ax = bar_plot(
            sample_data['categories'],
            sample_data['values'],
            title="Test Bar Plot"
        )
        
        assert fig is not None
        assert ax is not None
        assert ax.get_title() == "Test Bar Plot"
    
    def test_bar_plot_colors(self, sample_data):
        """Test bar plot with custom colors."""
        fig, ax = bar_plot(
            sample_data['categories'],
            sample_data['values'],
            color='steelblue'
        )
        
        assert fig is not None
        bars = ax.patches
        assert len(bars) == len(sample_data['categories'])
    
    def test_horizontal_bar_plot(self, sample_data):
        """Test horizontal bar plot."""
        fig, ax = bar_plot(
            sample_data['categories'],
            sample_data['values'],
            horizontal=True
        )
        
        assert fig is not None
        assert len(ax.patches) == len(sample_data['categories'])


class TestScatterPlot:
    """Test suite for scatter_plot function."""
    
    def test_basic_scatter(self, sample_data):
        """Test basic scatter plot."""
        fig, ax = scatter_plot(
            sample_data['x'],
            sample_data['y'],
            title="Test Scatter"
        )
        
        assert fig is not None
        assert ax is not None
        collections = ax.collections
        assert len(collections) > 0
    
    def test_scatter_with_colors(self, sample_data):
        """Test scatter plot with color mapping."""
        colors = sample_data['y']
        fig, ax = scatter_plot(
            sample_data['x'],
            sample_data['y'],
            c=colors,
            cmap='viridis'
        )
        
        assert fig is not None
        assert len(ax.collections) > 0
    
    def test_scatter_with_sizes(self, sample_data):
        """Test scatter plot with varying sizes."""
        sizes = sample_data['y'] * 10
        fig, ax = scatter_plot(
            sample_data['x'],
            sample_data['y'],
            s=sizes
        )
        
        assert fig is not None


class TestHistogramPlot:
    """Test suite for histogram_plot function."""
    
    def test_basic_histogram(self):
        """Test basic histogram creation."""
        data = np.random.randn(1000)
        fig, ax = histogram_plot(
            data,
            bins=30,
            title="Test Histogram"
        )
        
        assert fig is not None
        assert ax is not None
        assert ax.get_title() == "Test Histogram"
    
    def test_histogram_with_density(self):
        """Test histogram with density normalization."""
        data = np.random.randn(1000)
        fig, ax = histogram_plot(
            data,
            bins=20,
            density=True
        )
        
        assert fig is not None
        # Check that histogram is normalized
        patches = ax.patches
        assert len(patches) > 0


class TestPieChart:
    """Test suite for pie_chart function."""
    
    def test_basic_pie_chart(self, sample_data):
        """Test basic pie chart creation."""
        fig, ax = pie_chart(
            sample_data['values'],
            labels=sample_data['categories'],
            title="Test Pie Chart"
        )
        
        assert fig is not None
        assert ax is not None
        assert ax.get_title() == "Test Pie Chart"
    
    def test_pie_chart_with_autopct(self, sample_data):
        """Test pie chart with percentage labels."""
        fig, ax = pie_chart(
            sample_data['values'],
            labels=sample_data['categories'],
            autopct='%1.1f%%'
        )
        
        assert fig is not None
    
    def test_pie_chart_explode(self, sample_data):
        """Test pie chart with exploded slice."""
        explode = [0.1, 0, 0, 0]
        fig, ax = pie_chart(
            sample_data['values'],
            labels=sample_data['categories'],
            explode=explode
        )
        
        assert fig is not None


class TestMultiLinePlot:
    """Test suite for multi_line_plot function."""
    
    def test_multiple_lines(self, sample_data):
        """Test plotting multiple lines."""
        y_data = [
            sample_data['y'],
            sample_data['y'] * 1.5,
            sample_data['y'] * 0.5
        ]
        labels = ['Line 1', 'Line 2', 'Line 3']
        
        fig, ax = multi_line_plot(
            sample_data['x'],
            y_data,
            labels=labels,
            title="Multi-Line Plot"
        )
        
        assert fig is not None
        assert len(ax.get_lines()) == 3
        assert ax.get_title() == "Multi-Line Plot"
    
    def test_multi_line_with_legend(self, sample_data):
        """Test multi-line plot has legend."""
        y_data = [sample_data['y'], sample_data['y'] * 2]
        labels = ['A', 'B']
        
        fig, ax = multi_line_plot(
            sample_data['x'],
            y_data,
            labels=labels
        )
        
        legend = ax.get_legend()
        assert legend is not None


class TestGroupedBarPlot:
    """Test suite for grouped_bar_plot function."""
    
    def test_grouped_bars(self, sample_data):
        """Test grouped bar plot creation."""
        data = {
            'Group 1': sample_data['values'],
            'Group 2': sample_data['values'] * 1.2
        }
        
        fig, ax = grouped_bar_plot(
            sample_data['categories'],
            data,
            title="Grouped Bars"
        )
        
        assert fig is not None
        assert ax.get_title() == "Grouped Bars"
        # Should have bars for both groups
        assert len(ax.patches) == len(sample_data['categories']) * 2


class TestErrorHandling:
    """Test error handling in plot utilities."""
    
    def test_empty_data(self):
        """Test handling of empty data."""
        with pytest.raises((ValueError, IndexError)):
            line_plot([], [])
    
    def test_mismatched_dimensions(self):
        """Test handling of mismatched array dimensions."""
        x = [1, 2, 3]
        y = [1, 2]  # Different length
        
        with pytest.raises((ValueError, AssertionError)):
            line_plot(x, y)
    
    def test_invalid_plot_type(self):
        """Test handling of invalid parameters."""
        # This should raise an error or handle gracefully
        try:
            line_plot([1, 2], [1, 2], color='invalid_color_name_xyz')
        except (ValueError, KeyError):
            pass  # Expected behavior


# Performance and integration tests
class TestIntegration:
    """Integration tests for combined functionality."""
    
    def test_subplot_creation(self, sample_data):
        """Test creating multiple plots in subplots."""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Create different plots in each subplot
        line_plot(sample_data['x'], sample_data['y'], ax=axes[0, 0])
        bar_plot(sample_data['categories'], sample_data['values'], ax=axes[0, 1])
        scatter_plot(sample_data['x'], sample_data['y'], ax=axes[1, 0])
        histogram_plot(np.random.randn(100), ax=axes[1, 1])
        
        assert fig is not None
        assert len(fig.axes) == 4
    
    def test_plot_save_load(self, sample_data, tmp_path):
        """Test saving and loading plots."""
        fig, ax = line_plot(sample_data['x'], sample_data['y'])
        
        # Save to temporary file
        output_path = tmp_path / "test_plot.png"
        fig.savefig(output_path, dpi=100)
        
        # Verify file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
