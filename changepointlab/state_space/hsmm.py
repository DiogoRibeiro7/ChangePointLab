from warnings import warn

from changepoint_lab.algorithms.state_space.hsmm import *  # noqa: F401,F403

warn(
    "`changepointlab.state_space.hsmm` is deprecated. "
    "Use `changepoint_lab.algorithms.state_space.hsmm`.",
    DeprecationWarning,
    stacklevel=2,
)
