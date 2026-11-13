"""The Phase 9 latency plot reads every speed CSV, including the spilled point."""

import csv
from argparse import Namespace

from ncm_lab.bench import long_outputs

FIELDS = ["length", "model", "batch_size", "median_ms"]


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_plot_writes_png_and_skips_eager_and_ratio_files(tmp_path, monkeypatch):
    monkeypatch.setattr(long_outputs, "RESULTS", tmp_path)
    for L in (128, 1024):
        _write(tmp_path / f"phase9_speed_{L}.csv",
               [{"length": L, "model": m, "batch_size": bs, "median_ms": L * k}
                for m, k in (("ncm", 1), ("ar-gpt2", 5)) for bs in (1, 8)])
    # Neither file has the speed columns: the glob must not pick them up.
    (tmp_path / "phase9_speed_128_ratios.csv").write_text("x\n1\n")
    (tmp_path / "phase9_speed_128_eager.csv").write_text("x\n1\n")

    long_outputs.plot(Namespace())

    assert (tmp_path / "phase9_latency_vs_length.png").stat().st_size > 10_000
