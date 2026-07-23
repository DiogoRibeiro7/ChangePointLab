from __future__ import annotations

from .plotting_helpers import (
    plot_changepoint_posterior_mass,
    plot_pointwise_bands,
    plot_posterior_num_segments,
)
from .edivisive_plotting import plot_scree_edivisive, plot_segments_1d
from .kcp_plotting import plot_segments_1d as plot_segments_1d_kcp, plot_model_scree

__all__ = [
    "plot_changepoint_posterior_mass",
    "plot_pointwise_bands",
    "plot_posterior_num_segments",
    "plot_scree_edivisive",
    # KCP plotting
    "plot_segments_1d",
    "plot_model_scree",
]

# expose KCP's segment plot under the generic name
plot_segments_1d = plot_segments_1d_kcp
