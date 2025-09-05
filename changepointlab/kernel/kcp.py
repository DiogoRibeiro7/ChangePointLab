from warnings import warn

from changepoint_lab.algorithms.kernel.kcp import *  # noqa: F401,F403

warn(
    "`changepointlab.kernel.kcp` is deprecated. "
    "Use `changepoint_lab.algorithms.kernel.kcp`.",
    DeprecationWarning,
    stacklevel=2,
)
