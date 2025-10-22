"""
Report and Output Utilities for Analysis Module

Provides common utilities for:
- Console output formatting
- Template rendering
- Directory management
- Status reporting
"""

from pathlib import Path
from typing import Dict, Any, Optional


def print_separator(char: str = '=', width: int = 70):
    """
    Print a separator line

    Args:
        char: Character to use for separator
        width: Width of separator line
    """
    print(char * width)


def print_section_header(title: str, width: int = 70, char: str = '='):
    """
    Print a boxed section header

    Args:
        title: Title text
        width: Total width
        char: Border character
    """
    print()
    print(char * width)
    print(title)
    print(char * width)


def print_status(message: str, success: bool = True, indent: int = 0):
    """
    Print a status message with success/failure indicator

    Args:
        message: Message to print
        success: If True, show ✓, else show ✗
        indent: Number of spaces to indent
    """
    symbol = "✓" if success else "✗"
    spaces = " " * indent
    print(f"{spaces}{symbol} {message}")


def create_output_dir(path: Path, verbose: bool = True) -> Path:
    """
    Create output directory with logging

    Args:
        path: Directory path to create
        verbose: If True, print creation message

    Returns:
        Path object for created directory
    """
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)

    if verbose:
        print(f"Output directory: {path.absolute()}")

    return path


def render_template(template_path: Path, **variables) -> str:
    """
    Simple template rendering using string formatting

    Args:
        template_path: Path to template file
        **variables: Variables to substitute in template

    Returns:
        Rendered template string
    """
    template_path = Path(template_path)

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, 'r') as f:
        template = f.read()

    # Use format to substitute variables
    return template.format(**variables)


def get_dimensionality_text(is_2d: bool) -> str:
    """
    Get dimensionality description text

    Args:
        is_2d: Whether data is 2D

    Returns:
        Dimensionality string
    """
    return '2D' if is_2d else '1D'


def get_function_fitting_details(is_2d: bool) -> str:
    """
    Get function fitting details section based on dimensionality

    Args:
        is_2d: Whether data is 2D

    Returns:
        Markdown-formatted details string
    """
    if is_2d:
        return """
For 2D functions:
- **Surface plots** showing true function vs NN prediction
- **Contour plots** for easier comparison
- **MSE calculations** displayed on each plot

Each visualization shows all model types (MLP, SIREN, KAN, KAN with pruning) side-by-side.
"""
    else:
        return """
For 1D functions:
- **Line plots** showing true function vs NN output
- **Point-by-point comparisons** across the domain
- **MSE calculations** for each model

Each visualization compares all model types (MLP, SIREN, KAN, KAN with pruning).
"""


def get_heatmap_section(is_2d: bool) -> str:
    """
    Get heatmap analysis section if applicable

    Args:
        is_2d: Whether data is 2D

    Returns:
        Markdown-formatted heatmap section or empty string
    """
    if not is_2d:
        return ""

    return """
### 3. Heatmap Analysis (2D Only)

Location: `03_heatmap_analysis/`

Detailed heatmap analysis for 2D functions:
- **Comparison heatmaps** - Side-by-side views of true function, prediction, and error
- **Error analysis** - Absolute error, signed error, and relative error maps
- **Cross-section plots** - 1D slices at fixed x₁ and x₂ values
- **Error quantile maps** - Identifying high-error regions
- **Error statistics** - Quantitative breakdown by region

Files generated per function and model:
- `heatmap_<N>_<function>_<model>.png` - Detailed comparison heatmaps
- `cross_section_<N>_<function>.png` - Cross-section comparisons
- `error_quantile_<N>_<function>_<model>.png` - Error quantile analysis
"""


def get_detailed_analysis_extra(is_2d: bool) -> str:
    """
    Get extra detailed analysis steps for 2D data

    Args:
        is_2d: Whether data is 2D

    Returns:
        Markdown-formatted extra steps or empty string
    """
    if not is_2d:
        return ""

    return """
4. Review **heatmap analysis** to identify problematic regions in the domain
5. Check **cross-sections** to understand behavior along specific dimensions
6. Use **error quantile maps** to find where models struggle most
"""


def format_models_dir(models_dir: Optional[str]) -> str:
    """
    Format models directory path for display

    Args:
        models_dir: Models directory path or None

    Returns:
        Formatted string
    """
    return models_dir if models_dir else 'Not provided'


# Template directory path
def get_template_dir() -> Path:
    """
    Get the templates directory path

    Returns:
        Path to templates directory
    """
    return Path(__file__).parent / 'templates'


def load_analysis_summary_template() -> Path:
    """Get path to analysis summary template"""
    return get_template_dir() / 'analysis_summary.md.template'


def load_thesis_report_template() -> Path:
    """Get path to thesis report template"""
    return get_template_dir() / 'thesis_report.md.template'


__all__ = [
    'print_separator',
    'print_section_header',
    'print_status',
    'create_output_dir',
    'render_template',
    'get_dimensionality_text',
    'get_function_fitting_details',
    'get_heatmap_section',
    'get_detailed_analysis_extra',
    'format_models_dir',
    'get_template_dir',
    'load_analysis_summary_template',
    'load_thesis_report_template',
]
