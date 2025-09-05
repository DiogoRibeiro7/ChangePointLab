from warnings import warn

from changepoint_lab.algorithms.optimization.pelt import *  # noqa: F401,F403

warn(
    "`changepointlab.optimization.pelt` is deprecated. "
    "Use `changepoint_lab.algorithms.optimization.pelt`.",
    DeprecationWarning,
    stacklevel=2,
)
