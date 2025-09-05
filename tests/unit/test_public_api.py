import importlib
import warnings


def test_top_level_reexports():
    from changepoint_lab import (
        BOCPD,
        EDivisive,
        HSMM,
        KernelCPD,
        PELT,
        ChangePointResult,
    )
    from changepoint_lab.algorithms.bayesian.bocpd import BOCPD as _BOCPD
    from changepoint_lab.algorithms.nonparametric.edivisive import (
        EDivisive as _EDivisive,
    )
    from changepoint_lab.algorithms.state_space.hsmm import HSMM as _HSMM
    from changepoint_lab.algorithms.kernel.kcp import KernelCPD as _KernelCPD
    from changepoint_lab.algorithms.optimization.pelt import PELT as _PELT
    from changepoint_lab.core.datatypes import ChangePointResult as _CPR

    assert BOCPD.__name__ == _BOCPD.__name__
    assert EDivisive.__name__ == _EDivisive.__name__
    assert HSMM.__name__ == _HSMM.__name__
    assert KernelCPD.__name__ == _KernelCPD.__name__
    assert PELT.__name__ == _PELT.__name__
    assert ChangePointResult.__name__ == _CPR.__name__


def test_compat_deprecated_imports():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        from changepoint_lab import pelt  # noqa: F401
    assert any(issubclass(warn.category, DeprecationWarning) for warn in w)
