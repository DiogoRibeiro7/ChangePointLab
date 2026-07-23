try:
    import matplotlib
except ModuleNotFoundError:
    pass
else:
    matplotlib.use("Agg")


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark integration tests")
