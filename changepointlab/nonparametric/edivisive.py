from warnings import warn

from changepoint_lab.algorithms.nonparametric.edivisive import *  # noqa: F401,F403

warn(
    "`changepointlab.nonparametric.edivisive` is deprecated. "
    "Use `changepoint_lab.algorithms.nonparametric.edivisive`.",
    DeprecationWarning,
    stacklevel=2,
)
