import json
import os
import subprocess
import sys
import numpy as np
import xml.etree.ElementTree as ET
import pytest
from pathlib import Path

STUB_CODE = """import matplotlib.pyplot as plt

def plot_blocks(edges, block_value, data, ax=None, mode='counts'):
    if ax is None:
        _, ax = plt.subplots()
    ax.step(range(len(data)), data, where='post')
    return ax
"""


def _run_cli(csv_path: Path, out_dir: Path, tmp_path: Path, extra_args=None):
    extra_args = extra_args or []
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{tmp_path}:{env.get('PYTHONPATH', '')}"
    cmd = [
        sys.executable,
        "-m",
        "toolkit.cpd_cli",
        "--input",
        str(csv_path),
        "--output",
        str(out_dir),
        "bayesian-blocks",
    ] + list(extra_args)
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


@pytest.mark.integration
def test_cli_edges(tmp_path: Path):
    (tmp_path / "bb_plotting.py").write_text(STUB_CODE)
    csv_path = tmp_path / "counts.csv"
    csv_path.write_text("count\n1\n2\n3\n4\n5\n6\n")
    out_dir = tmp_path / "out"
    res = _run_cli(csv_path, out_dir, tmp_path)
    assert res.returncode == 0, res.stderr
    data = np.load(out_dir / "bayesian-blocks_results.npz")
    edges = data["edges"]
    assert edges.ndim == 1 and edges.size >= 2
    diffs = np.diff(edges)
    assert np.all(diffs > 0)
    meta = json.load(open(out_dir / "bayesian-blocks_metadata.json"))
    assert meta["method"] == "bayesian_blocks"


@pytest.mark.integration
def test_cli_random_profile(tmp_path: Path):
    (tmp_path / "bb_plotting.py").write_text(STUB_CODE)
    csv_path = tmp_path / "rand.csv"
    rng = np.random.default_rng(0)
    counts = rng.poisson(1.0, size=20)
    csv_path.write_text("count\n" + "\n".join(str(int(c)) for c in counts) + "\n")
    out_dir = tmp_path / "out"
    profile_path = tmp_path / "run.cprof"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{tmp_path}:{env.get('PYTHONPATH', '')}"
    cmd = [
        sys.executable,
        "-m",
        "cProfile",
        "-o",
        str(profile_path),
        "-m",
        "toolkit.cpd_cli",
        "--input",
        str(csv_path),
        "--output",
        str(out_dir),
        "bayesian-blocks",
    ]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert profile_path.exists() and profile_path.stat().st_size > 0


@pytest.mark.integration
def test_cli_export_formats(tmp_path: Path):
    (tmp_path / "bb_plotting.py").write_text(STUB_CODE)
    csv_path = tmp_path / "counts.csv"
    csv_path.write_text("count\n0\n1\n0\n1\n0\n1\n")
    out_dir = tmp_path / "out"
    res = _run_cli(csv_path, out_dir, tmp_path)
    assert res.returncode == 0, res.stderr
    meta = json.load(open(out_dir / "bayesian-blocks_metadata.json"))
    json_path = tmp_path / "export.json"
    with json_path.open("w") as f:
        json.dump(meta, f)
    edges = np.load(out_dir / "bayesian-blocks_results.npz")["edges"]
    graphml_path = tmp_path / "export.graphml"
    gml = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
    graph = ET.SubElement(gml, "graph", edgedefault="undirected")
    for i in range(len(edges)):
        ET.SubElement(graph, "node", id=str(i))
    for i in range(len(edges) - 1):
        ET.SubElement(graph, "edge", source=str(i), target=str(i + 1))
    ET.ElementTree(gml).write(graphml_path)
    with json_path.open() as f:
        json.load(f)
    ET.parse(graphml_path)
    assert json_path.stat().st_size > 0
    assert graphml_path.stat().st_size > 0
