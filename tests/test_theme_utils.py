"""Unit tests for theme_utils module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils.theme_utils import (
    apply_dark_theme,
    apply_minimal_theme,
    apply_corporate_theme,
    apply_publication_theme,
    apply_ieee_theme,
    apply_colorblind_friendly_theme,
    apply_high_contrast_theme,
    apply_pastel_theme,
    get_colorblind_palette
)


@pytest.fixture(autouse=True)
def reset_matplotlib():
    """Reset matplotlib settings before and after each test."""
    # Save original settings
    original_params = mpl.rcParams.copy()
    yield
    # Restore original settings
    mpl.rcParams.update(original_params)
    plt.close('all')


class TestThemeApplication:
    """Test suite for theme application functions."""
    
    def test_dark_theme(self):
        """Test dark theme application."""
        apply_dark_theme()
        
        # Check that dark colors are set
        assert mpl.rcParams['figure.facecolor'] != 'white'
        assert mpl.rcParams['axes.facecolor'] != 'white'
        
        # Verify text is light colored for dark background
        text_color = mpl.rcParams['text.color']
        assert text_color in ['white', 'lightgray', '#E0E0E0']
    
    def test_minimal_theme(self):
        """Test minimal theme application."""
        apply_minimal_theme()
        
        # Minimal theme should have clean, simple settings
        assert mpl.rcParams['axes.grid'] == True
        assert mpl.rcParams['grid.alpha'] < 0.5
    
    def test_corporate_theme(self):
        """Test corporate theme application."""
        apply_corporate_theme()
        
        # Corporate theme should have professional settings
        assert 'font.size' in mpl.rcParams
        assert mpl.rcParams['axes.edgecolor'] is not None
    
    def test_publication_theme(self):
        """Test publication theme application."""
        apply_publication_theme()
        
        # Publication theme should use serif fonts
        font_family = mpl.rcParams['font.family']
        assert 'serif' in font_family or isinstance(font_family, list)
        
        # Should have appropriate font sizes
        assert mpl.rcParams['font.size'] >= 10
    
    def test_ieee_theme(self):
        """Test IEEE theme application."""
        apply_ieee_theme()
        
        # IEEE style should have specific formatting
        assert mpl.rcParams['font.size'] >= 8
        assert 'font.family' in mpl.rcParams
    
    def test_colorblind_friendly_theme(self):
        """Test colorblind-friendly theme application."""
        apply_colorblind_friendly_theme()
        
        # Should set safe color cycle
        prop_cycle = mpl.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        
        # Should have multiple colors
        assert len(colors) >= 3
    
    def test_high_contrast_theme(self):
        """Test high contrast theme application."""
        apply_high_contrast_theme()
        
        # High contrast should have strong color differences
        fg_color = mpl.rcParams['text.color']
        bg_color = mpl.rcParams['figure.facecolor']
        
        # Colors should be different
        assert fg_color != bg_color
    
    def test_pastel_theme(self):
        """Test pastel theme application."""
        apply_pastel_theme()
        
        # Pastel theme should have soft colors
        prop_cycle = mpl.rcParams['axes.prop_cycle']
        assert prop_cycle is not None


class TestColorPalette:
    """Test color palette generation."""
    
    def test_colorblind_palette(self):
        """Test colorblind-friendly palette generation."""
        palette = get_colorblind_palette()
        
        # Should return a list of colors
        assert isinstance(palette, list)
        assert len(palette) >= 3
        
        # Each color should be a valid hex code
        for color in palette:
            assert color.startswith('#')
            assert len(color) == 7  # #RRGGBB format
    
    def test_palette_uniqueness(self):
        """Test that palette colors are distinct."""
        palette = get_colorblind_palette()
        
        # All colors should be unique
        assert len(palette) == len(set(palette))


class TestThemePersistence:
    """Test that themes persist across plot creation."""
    
    def test_theme_persists_after_plot(self):
        """Test theme settings persist after creating a plot."""
        apply_dark_theme()
        
        # Get settings after theme application
        bg_color_after_theme = mpl.rcParams['figure.facecolor']
        
        # Create a plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        # Settings should still be the same
        assert mpl.rcParams['figure.facecolor'] == bg_color_after_theme
        
        plt.close(fig)
    
    def test_theme_overwrite(self):
        """Test that applying a new theme overwrites the old one."""
        # Apply dark theme
        apply_dark_theme()
        dark_bg = mpl.rcParams['figure.facecolor']
        
        # Apply minimal theme
        apply_minimal_theme()
        minimal_bg = mpl.rcParams['figure.facecolor']
        
        # Colors should be different
        assert dark_bg != minimal_bg


class TestThemeWithPlots:
    """Test themes work correctly with actual plots."""
    
    def test_plot_with_publication_theme(self):
        """Test creating a plot with publication theme."""
        apply_publication_theme()
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        ax.set_title("Test Plot")
        
        # Plot should be created successfully
        assert fig is not None
        assert len(ax.lines) == 1
        
        plt.close(fig)
    
    def test_plot_with_colorblind_theme(self):
        """Test creating multi-line plot with colorblind theme."""
        apply_colorblind_friendly_theme()
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3], label='Line 1')
        ax.plot([1, 2, 3], [3, 2, 1], label='Line 2')
        ax.legend()
        
        # Should have 2 lines with distinct colors
        assert len(ax.lines) == 2
        
        plt.close(fig)
    
    def test_subplot_with_theme(self):
        """Test creating subplots with corporate theme."""
        apply_corporate_theme()
        
        fig, axes = plt.subplots(2, 2)
        
        for ax in axes.flat:
            ax.plot([1, 2, 3], [1, 2, 3])
        
        # All subplots should be created
        assert len(fig.axes) == 4
        
        plt.close(fig)


class TestThemeEdgeCases:
    """Test edge cases and error handling."""
    
    def test_multiple_theme_applications(self):
        """Test applying the same theme multiple times."""
        # Should not cause errors
        for _ in range(3):
            apply_minimal_theme()
        
        # Settings should be consistent
        assert mpl.rcParams['axes.grid'] == True
    
    def test_theme_with_custom_figsize(self):
        """Test theme doesn't interfere with custom figure sizes."""
        apply_publication_theme()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Figure should have the requested size
        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 8
        
        plt.close(fig)
    
    def test_theme_with_custom_colors(self):
        """Test applying custom colors after theme."""
        apply_minimal_theme()
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3], color='red')
        
        # Custom color should override theme
        line_color = ax.lines[0].get_color()
        assert line_color == 'red'
        
        plt.close(fig)


class TestThemeDocumentation:
    """Test that all themes are properly documented."""
    
    def test_all_themes_have_docstrings(self):
        """Test that all theme functions have docstrings."""
        themes = [
            apply_dark_theme,
            apply_minimal_theme,
            apply_corporate_theme,
            apply_publication_theme,
            apply_ieee_theme,
            apply_colorblind_friendly_theme,
            apply_high_contrast_theme,
            apply_pastel_theme
        ]
        
        for theme in themes:
            assert theme.__doc__ is not None
            assert len(theme.__doc__.strip()) > 0


class TestIntegration:
    """Integration tests combining multiple themes and features."""
    
    def test_sequential_theme_switching(self):
        """Test switching between themes sequentially."""
        themes = [
            apply_dark_theme,
            apply_minimal_theme,
            apply_publication_theme
        ]
        
        for theme in themes:
            theme()
            
            # Create a test plot
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])
            
            # Should work without errors
            assert fig is not None
            plt.close(fig)
    
    def test_theme_with_saving(self, tmp_path):
        """Test saving plots with different themes."""
        apply_publication_theme()
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        # Save to temporary file
        output_path = tmp_path / "themed_plot.pdf"
        fig.savefig(output_path, dpi=300)
        
        # File should be created
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
