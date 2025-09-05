from warnings import warn

from changepoint_lab.algorithms.bayesian.bocpd import *  # noqa: F401,F403

warn(
    "`changepointlab.bayesian.bocpd` is deprecated. "
    "Use `changepoint_lab.algorithms.bayesian.bocpd`.",
    DeprecationWarning,
    stacklevel=2,
)
