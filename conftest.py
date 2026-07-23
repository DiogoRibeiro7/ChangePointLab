import sys
from pathlib import Path

try:
    import matplotlib
except ModuleNotFoundError:
    pass
else:
    matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark integration tests")
