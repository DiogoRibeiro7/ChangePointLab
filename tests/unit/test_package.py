import importlib
import subprocess
import sys


# Verify that top-level imports expose the expected symbols

def test_main_imports():
    pkg = importlib.import_module("changepoint_lab")
    names = [
        "BOCPD",
        "BOCPDConfig",
        "BOCPDResult",
        "ConstantHazard",
        "BoostedBoundaryHazard",
        "ScheduledHazard",
        "PELT",
        "EDivisive",
        "HSMM",
        "KernelCPD",
        "SlicedPoissonCPD",
        "WithinPeriodCPD",
        "SDHMM",
        "SDHMMMixVI",
    ]
    for name in names:
        assert hasattr(pkg, name)


# Ensure that the major classes can be instantiated with minimal arguments

def test_class_instantiation():
    import numpy as np
    from changepoint_lab.algorithms.bayesian.bocpd import (
        BOCPD,
        BOCPDConfig,
        BoostedBoundaryHazard,
        ConstantHazard,
    )
    from changepoint_lab.algorithms.optimization.pelt import (
        NormalMeanKnownVar,
        NormalMeanVarUnknown,
    )
    from changepoint_lab.algorithms.point_process import SlicedPoissonCPD, SlicedPoissonConfig
    from changepoint_lab.algorithms.state_space.hsmm import (
        HSMM,
        HSMMConfig,
        HSMMParams,
        PoissonDur,
    )
    from changepoint_lab.algorithms.state_space.sdhmm import SDHMM, SDHMMConfig
    from changepoint_lab.algorithms.state_space.sdhmm_mix_vi import (
        SDHMMMixVI,
        SDHMMMixVIConfig,
    )
    from changepoint_lab.algorithms.bayesian.within_period import (
        ModelPrior,
        RJConfig,
        WithinPeriodCPD,
    )

    # BOCPD related classes
    hazard = ConstantHazard(mean_run_length=10)
    BOCPD(hazard, BOCPDConfig(max_run_length=5))
    BoostedBoundaryHazard(hazard, period=10, boundary_indices={0})

    # Within-period CPD
    prior = ModelPrior(N=8, l=2)
    WithinPeriodCPD(prior)
    RJConfig()

    # PELT cost classes
    NormalMeanKnownVar(sigma2=1.0)
    NormalMeanVarUnknown()

    SlicedPoissonCPD(SlicedPoissonConfig(period=1.0, n_basis=1, degree=0))

    # HSMM
    hsmm_cfg = HSMMConfig(K=1, Dmax=5)
    hsmm_params = HSMMParams(
        pi=np.array([1.0]),
        A=np.array([[1.0]]),
        duration=("poisson", PoissonDur(lam=np.array([1.0]))),
    )
    HSMM(hsmm_cfg, hsmm_params)

    # SDHMM
    SDHMM(SDHMMConfig(K=1))
    SDHMMMixVI(SDHMMMixVIConfig(K=1, M=1))


# Ensure CLI entry points are available as Python modules

def test_cli_entry_points():
    modules = [
        "changepoint_lab.cli.bocpd_cli",
        "changepoint_lab.algorithms.bayesian.within_period.cli",
        "changepoint_lab.cli.cpd_cli",
    ]
    for mod in modules:
        result = subprocess.run(
            [sys.executable, "-m", mod, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert result.returncode == 0
