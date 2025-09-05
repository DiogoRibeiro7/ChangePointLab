import sys
from pathlib import Path
import importlib

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Allow `import changepoint_lab` when running tests directly from the repo root.
pkg_name = ROOT.name
pkg = importlib.import_module(pkg_name)
sys.modules.setdefault("changepoint_lab", pkg)
algo_mod = importlib.import_module(f"{pkg_name}.algorithms")
core_mod = importlib.import_module(f"{pkg_name}.core")
sys.modules.setdefault("changepoint_lab.algorithms", algo_mod)
sys.modules.setdefault("changepoint_lab.core", core_mod)


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark integration tests")
