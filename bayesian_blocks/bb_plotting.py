# bb_plotting.py
# MIT License
# (c) 2025


from __future__ import annotations

from typing import Optional, Sequence, Union, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from bayesian_blocks import BBResult


class BBPlotter:
    """Advanced plotting class for Bayesian Blocks results."""

    def __init__(self, result: BBResult, data: Optional[np.ndarray] = None):
        self.result = result
        self.data = data
        self.style_config = {
            "block_color": "#2E86AB",
            "change_color": "#A23B72",
            "data_color": "#F18F01",
            "confidence_alpha": 0.3,
            "grid_alpha": 0.3,
        }

    def plot_blocks(
        self,
        ax: Optional[Axes] = None,
        show_data: bool = True,
        show_changepoints: bool = True,
        show_confidence: bool = False,
        style: str = "default",
    ) -> Axes:
        """
        Enhanced block plotting with multiple visualization options.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axes to plot on. If None, creates new figure.
        show_data : bool, default True
            Whether to show underlying data points/histogram.
        show_changepoints : bool, default True
            Whether to mark changepoint locations.
        show_confidence : bool, default False
            Whether to show confidence bands (if available).
        style : str, default 'default'
            Plotting style ('default', 'minimal', 'publication').
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        # Apply style
        self._apply_style(ax, style)

        edges = self.result.edges
        values = self.result.block_value

        # Plot block structure
        for i in range(len(values)):
            left, right = edges[i], edges[i + 1]

            # Main block line
            ax.plot(
                [left, right],
                [values[i], values[i]],
                linewidth=3,
                color=self.style_config["block_color"],
                solid_capstyle="butt",
            )

            # Vertical connectors at changepoints
            if i < len(values) - 1:
                next_val = values[i + 1]
                ax.plot(
                    [right, right],
                    [values[i], next_val],
                    linewidth=2,
                    color=self.style_config["block_color"],
                    alpha=0.7,
                    linestyle="--",
                )

        # Show underlying data if provided
        if show_data and self.data is not None:
            self._plot_underlying_data(ax)

        # Mark changepoints
        if show_changepoints and len(self.result.change_points) > 0:
            for cp in self.result.change_points:
                if cp < len(edges) - 1:  # Valid changepoint
                    ax.axvline(
                        edges[cp],
                        color=self.style_config["change_color"],
                        linestyle=":",
                        alpha=0.8,
                        linewidth=2,
                        label="Changepoint"
                        if cp == self.result.change_points[0]
                        else "",
                    )

        # Confidence bands (if implemented)
        if show_confidence:
            self._plot_confidence_bands(ax)

        # Formatting
        ax.set_ylabel("Value")
        ax.grid(True, alpha=self.style_config["grid_alpha"])

        if show_changepoints and len(self.result.change_points) > 0:
            ax.legend()

        return ax

    def plot_diagnostics(self, figsize: Tuple[int, int] = (15, 10)) -> Figure:
        """
        Create comprehensive diagnostic plots.

        Returns
        -------
        Figure with multiple diagnostic subplots.
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Main blocks plot
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_blocks(ax1, show_confidence=True)
        ax1.set_title("Bayesian Blocks Segmentation")

        # Block size distribution
        ax2 = fig.add_subplot(gs[1, 0])
        block_sizes = np.diff(self.result.edges)
        ax2.hist(
            block_sizes,
            bins=min(15, len(block_sizes)),
            alpha=0.7,
            color=self.style_config["block_color"],
        )
        ax2.set_xlabel("Block Size")
        ax2.set_ylabel("Count")
        ax2.set_title("Block Size Distribution")

        # Block values distribution
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(
            self.result.block_value,
            bins=min(15, len(self.result.block_value)),
            alpha=0.7,
            color=self.style_config["data_color"],
        )
        ax3.set_xlabel("Block Value")
        ax3.set_ylabel("Count")
        ax3.set_title("Block Value Distribution")

        # Information criteria (if available)
        ax4 = fig.add_subplot(gs[1, 2])
        if hasattr(self.result, "aic") and hasattr(self.result, "bic"):
            criteria = ["AIC", "BIC", "Log-Likelihood"]
            values = [self.result.aic, self.result.bic, -self.result.log_likelihood]
            bars = ax4.bar(criteria, values, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
            ax4.set_title("Model Selection Criteria")
            ax4.tick_params(axis="x", rotation=45)
        else:
            ax4.text(
                0.5,
                0.5,
                "Criteria not available",
                transform=ax4.transAxes,
                ha="center",
                va="center",
            )
            ax4.set_title("Model Selection Criteria")

        # Residuals (if data provided)
        if self.data is not None:
            ax5 = fig.add_subplot(gs[2, :2])
            self._plot_residuals(ax5)
        else:
            ax5 = fig.add_subplot(gs[2, :2])
            ax5.text(
                0.5,
                0.5,
                "Residuals require original data",
                transform=ax5.transAxes,
                ha="center",
                va="center",
            )
            ax5.set_title("Residuals Analysis")

        # Summary statistics
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_summary_stats(ax6)

        return fig

    def plot_interactive(self) -> Optional[go.Figure]:
        """
        Create interactive Plotly visualization.

        Returns
        -------
        Plotly Figure object for interactive exploration.
        """
        if not HAS_PLOTLY:
            print("Plotly not available. Install with: pip install plotly")
            return None

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Blocks", "Block Sizes", "Block Values", "Summary"),
            specs=[[{"colspan": 2}, None], [{}, {}]],
        )

        # Main blocks plot
        edges = self.result.edges
        values = self.result.block_value

        # Create step plot manually
        x_step, y_step = [], []
        for i in range(len(values)):
            x_step.extend([edges[i], edges[i + 1]])
            y_step.extend([values[i], values[i]])

        fig.add_trace(
            go.Scatter(
                x=x_step,
                y=y_step,
                mode="lines",
                name="Blocks",
                line=dict(color="blue", width=3),
            ),
            row=1,
            col=1,
        )

        # Add changepoint markers
        if len(self.result.change_points) > 0:
            cp_x = [edges[cp] for cp in self.result.change_points if cp < len(edges)]
            cp_y = [
                values[min(cp, len(values) - 1)]
                for cp in self.result.change_points
                if cp < len(edges)
            ]

            fig.add_trace(
                go.Scatter(
                    x=cp_x,
                    y=cp_y,
                    mode="markers",
                    name="Changepoints",
                    marker=dict(color="red", size=10, symbol="diamond"),
                ),
                row=1,
                col=1,
            )

        # Block size histogram
        block_sizes = np.diff(edges)
        fig.add_trace(
            go.Histogram(x=block_sizes, name="Block Sizes", nbinsx=15), row=2, col=1
        )

        # Block values histogram
        fig.add_trace(
            go.Histogram(x=values, name="Block Values", nbinsx=15), row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title="Interactive Bayesian Blocks Analysis", showlegend=True, height=700
        )

        return fig

    def _apply_style(self, ax: Axes, style: str):
        """Apply visual style to axes."""
        if style == "publication":
            # Clean, publication-ready style
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(which="both", direction="out")
        elif style == "minimal":
            # Minimal style with light colors
            self.style_config.update(
                {
                    "block_color": "#666666",
                    "change_color": "#CCCCCC",
                    "data_color": "#999999",
                }
            )

    def _plot_underlying_data(self, ax: Axes):
        """Plot the original data points or histogram."""
        # This would depend on the data type and format
        # Implementation would vary based on whether data is events, counts, etc.
        if len(self.data.shape) == 1:
            if np.all(np.isin(self.data, [0, 1])):  # Binary data
                ax.scatter(
                    range(len(self.data)),
                    self.data,
                    alpha=0.6,
                    s=20,
                    color=self.style_config["data_color"],
                )
            else:  # Continuous or count data
                ax.plot(
                    self.data,
                    alpha=0.7,
                    linewidth=1,
                    color=self.style_config["data_color"],
                    marker="o",
                    markersize=2,
                )

    def _plot_confidence_bands(self, ax: Axes):
        """Plot confidence bands around blocks (placeholder for future implementation)."""
        # Would require bootstrap or analytical confidence intervals
        pass

    def _plot_residuals(self, ax: Axes):
        """Plot residuals analysis."""
        # Generate fitted values from blocks
        if hasattr(self, "_fitted_values"):
            residuals = self.data - self._fitted_values
            ax.scatter(self._fitted_values, residuals, alpha=0.6)
            ax.axhline(0, color="red", linestyle="--", alpha=0.8)
            ax.set_xlabel("Fitted Values")
            ax.set_ylabel("Residuals")
            ax.set_title("Residuals vs Fitted")

    def _plot_summary_stats(self, ax: Axes):
        """Plot key summary statistics as text."""
        stats_text = f"""
        Blocks: {len(self.result.block_value)}
        Changepoints: {len(self.result.change_points)}
        """

        if hasattr(self.result, "aic"):
            stats_text += f"AIC: {self.result.aic:.2f}\n"
        if hasattr(self.result, "bic"):
            stats_text += f"BIC: {self.result.bic:.2f}\n"

        ax.text(
            0.1,
            0.9,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=11,
            fontfamily="monospace",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Summary Statistics")


def plot_comparison(
    results: List[BBResult],
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 8),
) -> Figure:
    """
    Compare multiple Bayesian Blocks results side by side.

    Parameters
    ----------
    results : List[BBResult]
        List of results to compare.
    labels : List[str], optional
        Labels for each result. If None, uses generic labels.
    figsize : tuple
        Figure size.

    Returns
    -------
    Figure with comparison plots.
    """
    if labels is None:
        labels = [f"Result {i + 1}" for i in range(len(results))]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Bayesian Blocks Comparison", fontsize=16)

    colors = plt.cm.Set3(np.linspace(0, 1, len(results)))

    # Plot all results on same axes
    ax = axes[0, 0]
    for i, (result, label, color) in enumerate(zip(results, labels, colors)):
        edges = result.edges
        values = result.block_value

        # Offset slightly for visibility
        offset = i * 0.02 * (values.max() - values.min())

        for j in range(len(values)):
            ax.plot(
                [edges[j], edges[j + 1]],
                [values[j] + offset, values[j] + offset],
                linewidth=2,
                color=color,
                label=label if j == 0 else "",
            )

    ax.set_title("Overlay Comparison")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Number of blocks comparison
    ax = axes[0, 1]
    n_blocks = [len(r.block_value) for r in results]
    bars = ax.bar(labels, n_blocks, color=colors, alpha=0.7)
    ax.set_title("Number of Blocks")
    ax.set_ylabel("Count")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add value labels on bars
    for bar, n in zip(bars, n_blocks):
        height = bar.get_height()
        ax.annotate(
            f"{n}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    # Block size distributions
    ax = axes[1, 0]
    for i, (result, label, color) in enumerate(zip(results, labels, colors)):
        block_sizes = np.diff(result.edges)
        ax.hist(block_sizes, alpha=0.6, label=label, color=color, bins=15, density=True)
    ax.set_title("Block Size Distributions")
    ax.set_xlabel("Block Size")
    ax.set_ylabel("Density")
    ax.legend()

    # Model criteria comparison (if available)
    ax = axes[1, 1]
    has_criteria = all(hasattr(r, "aic") and hasattr(r, "bic") for r in results)

    if has_criteria:
        aic_values = [r.aic for r in results]
        bic_values = [r.bic for r in results]

        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width / 2, aic_values, width, label="AIC", alpha=0.7)
        ax.bar(x + width / 2, bic_values, width, label="BIC", alpha=0.7)

        ax.set_xlabel("Results")
        ax.set_ylabel("Criterion Value")
        ax.set_title("Model Selection Criteria")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "Model criteria not available",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_title("Model Selection Criteria")

    plt.tight_layout()
    return fig


def plot_sensitivity_analysis(
    data: np.ndarray,
    p0_values: Sequence[float],
    algorithm_func: callable,
    figsize: Tuple[int, int] = (12, 8),
) -> Figure:
    """
    Analyze sensitivity to the p0 parameter.

    Parameters
    ----------
    data : array-like
        Input data for analysis.
    p0_values : sequence of float
        Range of p0 values to test.
    algorithm_func : callable
        Function that takes (data, p0) and returns BBResult.
    figsize : tuple
        Figure size.

    Returns
    -------
    Figure showing sensitivity analysis.
    """
    results = []
    for p0 in p0_values:
        try:
            result = algorithm_func(data, p0=p0)
            results.append(result)
        except Exception as e:
            print(f"Warning: Failed for p0={p0}: {e}")
            results.append(None)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Sensitivity Analysis: p0 Parameter", fontsize=16)

    # Number of blocks vs p0
    ax = axes[0, 0]
    n_blocks = [len(r.block_value) if r else 0 for r in results]
    ax.plot(p0_values, n_blocks, "o-", linewidth=2, markersize=6)
    ax.set_xlabel("p0 (false positive rate)")
    ax.set_ylabel("Number of Blocks")
    ax.set_title("Blocks vs p0")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    # Model criteria vs p0 (if available)
    ax = axes[0, 1]
    valid_results = [r for r in results if r and hasattr(r, "aic")]

    if valid_results:
        valid_p0 = [
            p0_values[i] for i, r in enumerate(results) if r and hasattr(r, "aic")
        ]
        aic_values = [r.aic for r in valid_results]
        bic_values = [r.bic for r in valid_results]

        ax.plot(valid_p0, aic_values, "o-", label="AIC", linewidth=2)
        ax.plot(valid_p0, bic_values, "s-", label="BIC", linewidth=2)
        ax.set_xlabel("p0")
        ax.set_ylabel("Criterion Value")
        ax.set_title("Model Criteria vs p0")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
    else:
        ax.text(
            0.5,
            0.5,
            "Model criteria not available",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_title("Model Criteria vs p0")

    # Show a few representative segmentations
    ax = axes[1, :]
    ax = plt.subplot(2, 1, 2)  # Use full width for this plot

    # Select a few representative p0 values
    n_show = min(4, len(results))
    indices = np.linspace(0, len(results) - 1, n_show, dtype=int)

    colors = plt.cm.viridis(np.linspace(0, 1, n_show))

    for i, (idx, color) in enumerate(zip(indices, colors)):
        if results[idx] is None:
            continue

        result = results[idx]
        p0 = p0_values[idx]

        # Offset for visibility
        offset = (
            i
            * 0.05
            * (
                np.max([r.block_value.max() for r in results if r])
                - np.min([r.block_value.min() for r in results if r])
            )
        )

        edges = result.edges
        values = result.block_value + offset

        for j in range(len(values)):
            ax.plot(
                [edges[j], edges[j + 1]],
                [values[j], values[j]],
                linewidth=2,
                color=color,
                label=f"p0={p0:.3f}" if j == 0 else "",
            )

    ax.set_title("Representative Segmentations")
    ax.set_xlabel("Position")
    ax.set_ylabel("Value (offset)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_blocks_time(
    *,
    t_min: float,
    t_max: float,
    result: BBResult,
    ax: Optional[Axes] = None,
    title: Optional[str] = "Bayesian Blocks (rate over time)",
) -> Axes:
    """
    Step plot of blockwise rate (Poisson) over time.
    
    Parameters
    ----------
    t_min, t_max : float
        Time range for plot
    result : BBResult
        Result from bayesian_blocks_events
    ax : Axes, optional
        Matplotlib axes to plot on
    title : str, optional
        Plot title
        
    Returns
    -------
    Axes
        The matplotlib axes object
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    
    edges = result.edges
    vals = result.block_value
    
    # Plot horizontal lines for each block
    for i in range(len(vals)):
        ax.hlines(vals[i], edges[i], edges[i + 1], linewidth=2, color='blue')
        ax.vlines(edges[i], 0, vals[i], linestyles="dotted", linewidth=0.8, color='gray')
    
    # Final vertical line
    ax.vlines(edges[-1], 0, vals[-1], linestyles="dotted", linewidth=0.8, color='gray')
    
    ax.set_xlim(t_min, t_max)
    ax.set_ylabel("rate")
    ax.set_xlabel("time")
    
    if title:
        ax.set_title(title)
    
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def plot_blocks_index(
    *,
    N: int,
    result: BBResult,
    ax: Optional[Axes] = None,
    title: Optional[str] = "Bayesian Blocks (per-index)",
    ylabel: str = "value",
) -> Axes:
    """
    Step plot of block values over integer index (counts/bernoulli settings).
    
    Parameters
    ----------
    N : int
        Total number of data points
    result : BBResult
        Result from bayesian_blocks_counts or bayesian_blocks_bernoulli
    ax : Axes, optional
        Matplotlib axes to plot on
    title : str, optional
        Plot title
    ylabel : str, default "value"
        Y-axis label
        
    Returns
    -------
    Axes
        The matplotlib axes object
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    
    edges = result.edges.astype(int)
    vals = result.block_value
    
    # Plot horizontal lines for each block
    for i in range(len(vals)):
        ax.hlines(vals[i], edges[i], edges[i + 1], linewidth=2, color='blue')
        ax.vlines(
            edges[i],
            min(vals.min(), 0),
            max(vals.max(), vals[i]),
            linestyles="dotted",
            linewidth=0.8,
            color='gray'
        )
    
    # Final vertical line
    ax.vlines(
        edges[-1],
        min(vals.min(), 0),
        max(vals.max(), 0),
        linestyles="dotted",
        linewidth=0.8,
        color='gray'
    )
    
    ax.set_xlim(0, N)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("index")
    
    if title:
        ax.set_title(title)
        
    ax.grid(True, axis="y", alpha=0.3)
    return ax


# Add these functions to the BBPlotter class as well for consistency
def _add_compatibility_methods():
    """Add backward compatibility methods to BBPlotter class."""
    
    def plot_time_compat(self, t_min: float, t_max: float, **kwargs) -> Axes:
        """Compatibility method for time-based plots."""
        return plot_blocks_time(
            t_min=t_min, 
            t_max=t_max, 
            result=self.result,
            **kwargs
        )
    
    def plot_index_compat(self, N: int, **kwargs) -> Axes:
        """Compatibility method for index-based plots.""" 
        return plot_blocks_index(
            N=N,
            result=self.result,
            **kwargs
        )
    
    # Add methods to BBPlotter class
    BBPlotter.plot_time = plot_time_compat
    BBPlotter.plot_index = plot_index_compat


# Call this when module is imported
_add_compatibility_methods()