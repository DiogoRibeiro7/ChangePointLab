import json
import subprocess
import sys
import numpy as np
import xml.etree.ElementTree as ET
import pytest
from pathlib import Path

def _run_cli(csv_path: Path, out_dir: Path, tmp_path: Path, extra_args=None):
    extra_args = extra_args or []
    cmd = [
        sys.executable,
        "-m",
        "changepoint_lab.cli.cpd_cli",
        "--input",
        str(csv_path),
        "--output",
        str(out_dir),
        "edivisive",
        "--columns",
        "count",
        "--min-size",
        "2",
    ] + list(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.mark.integration
def test_cli_edges(tmp_path: Path):
    csv_path = tmp_path / "counts.csv"
    csv_path.write_text("count\n1\n2\n3\n4\n5\n6\n")
    out_dir = tmp_path / "out"
    res = _run_cli(csv_path, out_dir, tmp_path)
    assert res.returncode == 0, res.stderr
    data = np.load(out_dir / "edivisive_results.npz")
    cps = data["change_points"]
    assert cps.ndim == 1
    with (out_dir / "edivisive_metadata.json").open() as f:
        meta = json.load(f)
    assert meta["method"] == "edivisive"


@pytest.mark.integration
def test_cli_random_profile(tmp_path: Path):
    csv_path = tmp_path / "rand.csv"
    rng = np.random.default_rng(0)
    counts = rng.poisson(1.0, size=20)
    csv_path.write_text("count\n" + "\n".join(str(int(c)) for c in counts) + "\n")
    out_dir = tmp_path / "out"
    profile_path = tmp_path / "run.cprof"
    cmd = [
        sys.executable,
        "-m",
        "cProfile",
        "-o",
        str(profile_path),
        "-m",
        "changepoint_lab.cli.cpd_cli",
        "--input",
        str(csv_path),
        "--output",
        str(out_dir),
        "edivisive",
        "--columns",
        "count",
        "--min-size",
        "2",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert profile_path.exists() and profile_path.stat().st_size > 0


@pytest.mark.integration
def test_cli_export_formats(tmp_path: Path):
    csv_path = tmp_path / "counts.csv"
    csv_path.write_text("count\n0\n1\n0\n1\n0\n1\n")
    out_dir = tmp_path / "out"
    res = _run_cli(csv_path, out_dir, tmp_path)
    assert res.returncode == 0, res.stderr
    with (out_dir / "edivisive_metadata.json").open() as f:
        meta = json.load(f)
    json_path = tmp_path / "export.json"
    with json_path.open("w") as f:
        json.dump(meta, f)
    cps = np.load(out_dir / "edivisive_results.npz")["change_points"]
    graphml_path = tmp_path / "export.graphml"
    gml = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
    graph = ET.SubElement(gml, "graph", edgedefault="undirected")
    for i in range(len(cps)):
        ET.SubElement(graph, "node", id=str(i))
    for i in range(len(cps)):
        ET.SubElement(graph, "edge", source=str(i), target=str(i))
    ET.ElementTree(gml).write(graphml_path)
    with json_path.open() as f:
        json.load(f)
    ET.parse(graphml_path)
    assert json_path.stat().st_size > 0
    assert graphml_path.stat().st_size > 0
