import matplotlib.pyplot as plt
import numpy as np

from kcp.kcp_plotting import plot_segments_1d


def test_kcp_plot_segments_smoke():
    x = np.random.default_rng(0).normal(size=20)
    edges = np.array([0, 10, 20])
    ax = plot_segments_1d(x, edges)
    plt.close(ax.figure)
