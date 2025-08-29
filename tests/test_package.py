import importlib
import subprocess
import sys


# Verify that top-level imports expose the expected symbols

def test_main_imports():
    pkg = importlib.import_module("changepoint_toolkit")
    names = [
        "BOCPD",
        "BOCPDConfig",
        "BOCPDResult",
        "ConstantHazard",
        "BoostedBoundaryHazard",
        "WithinPeriodCPD",
        "ModelPrior",
        "RJConfig",
        "gram_rbf",
        "kcp_penalized",
        "kcp_select_bic",
        "edivisive",
        "pelt",
        "NormalMeanKnownVar",
        "NormalMeanVarUnknown",
        "HSMM",
        "HSMMConfig",
        "SDHMM",
        "SDHMMConfig",
    ]
    for name in names:
        assert hasattr(pkg, name)


# Ensure that the major classes can be instantiated with minimal arguments

def test_class_instantiation():
    import numpy as np
    from changepoint_toolkit import (
        BOCPD,
        BOCPDConfig,
        ConstantHazard,
        BoostedBoundaryHazard,
        WithinPeriodCPD,
        ModelPrior,
        RJConfig,
        NormalMeanKnownVar,
        NormalMeanVarUnknown,
        HSMM,
        HSMMConfig,
        SDHMM,
        SDHMMConfig,
    )
    from hsmm import HSMMParams, PoissonDur

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


# Ensure CLI entry points are available as Python modules

def test_cli_entry_points():
    modules = [
        "bocpd.bocpd_cli",
        "within_period.cli",
        "toolkit.cpd_cli",
    ]
    for mod in modules:
        result = subprocess.run(
            [sys.executable, "-m", mod, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert result.returncode == 0
