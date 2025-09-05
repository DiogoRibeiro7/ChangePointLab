import warnings


def test_top_level_exports():
    import changepoint_lab as cpl
    assert hasattr(cpl, "PELT")
    assert hasattr(cpl, "BOCPD")
    assert hasattr(cpl, "EDivisive")
    assert hasattr(cpl, "HSMM")
    assert hasattr(cpl, "KernelCPD")


def test_deprecated_imports_warn():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from changepoint_lab import pelt as _  # noqa: F401
        assert any(issubclass(ww.category, DeprecationWarning) for ww in w)


def test_legacy_module_shim_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from changepointlab.optimization import pelt as _  # noqa: F401
        assert any(issubclass(ww.category, DeprecationWarning) for ww in w)
