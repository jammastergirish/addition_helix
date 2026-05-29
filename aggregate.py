"""Walk `out/*/*/<pool>/fig_layer_sweep.json` and aggregate every cell's
L=0-vs-peak helix-share and per-layer peaks into two top-level files:

  out/_index.json     -- machine-readable; consumed by the React site
  out/_l0_share.csv   -- the same data in the historical csv format

A "cell" is one (model, script, pool, n_max, periods) combination. The pool
directory name carries the n_max and periods suffixes (e.g.
`mean_n600_p2-5-10-60-100`); we parse them back out.

Run this AFTER `run.sh` (or any subset of `main.py --sweep` runs) to
refresh the aggregates. It is fast and idempotent.

Usage:
    uv run aggregate.py
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"

POOL_RE = re.compile(
    r"^(?P<pool>mean|last)"
    r"(?:_n(?P<n_max>\d+))?"
    r"(?:_p(?P<periods>[\d-]+))?$"
)


def parse_pool_dir(name: str) -> dict | None:
    m = POOL_RE.match(name)
    if not m:
        return None
    n_max = int(m.group("n_max")) if m.group("n_max") else 100
    if m.group("periods"):
        periods = [int(t) for t in m.group("periods").split("-")]
    else:
        periods = [2, 5, 10, 100]
    return {"pool": m.group("pool"), "n_max": n_max, "periods": periods}


def main() -> None:
    if not OUT.exists():
        raise SystemExit(f"no {OUT}/ -- run main.py / run.sh first")

    cells = []
    for sweep_path in sorted(OUT.glob("*/*/*/fig_layer_sweep.json")):
        # out/<model_dir>/<script>/<pool_dir>/fig_layer_sweep.json
        pool_dir = sweep_path.parent
        script = pool_dir.parent.name
        model_dir = pool_dir.parent.parent.name
        model = model_dir.replace("__", "/")

        parsed = parse_pool_dir(pool_dir.name)
        if parsed is None:
            print(f"skip (unparseable pool dir): {pool_dir}")
            continue

        with open(sweep_path, encoding="utf-8") as f:
            sweep = json.load(f)

        meta = sweep.get("meta", {})
        peaks = sweep.get("peaks", {})
        l0 = sweep.get("l0_share", {})

        cells.append(
            {
                "model": model,
                "script": script,
                "pool": parsed["pool"],
                "n_max": parsed["n_max"],
                "periods": parsed["periods"],
                "n_layers": meta.get("n_layers"),
                # L=0 vs peak diagnostic (the blogpost's headline finding)
                "helix_r2_l0": l0.get("helix_r2_l0"),
                "helix_r2_peak": l0.get("helix_r2_peak"),
                "peak_layer": l0.get("peak_layer"),
                "rho": l0.get("rho"),
                # Other per-metric peaks
                "pc1_r2_peak": peaks.get("pc1_r2", {}).get("value"),
                "pc1_r2_peak_layer": peaks.get("pc1_r2", {}).get("layer"),
                "helix_over_pca_peak": peaks.get("helix_over_pca", {}).get("value"),
                "helix_over_pca_peak_layer": peaks.get("helix_over_pca", {}).get(
                    "layer"
                ),
                # Relative paths for the site's lazy loaders
                "paths": {
                    "dir": str(pool_dir.relative_to(OUT)),
                    "layer_sweep": str(sweep_path.relative_to(OUT)),
                    "fourier_pc1": _maybe_rel(pool_dir / "fig2_fourier_pc1.json"),
                    "circles": _maybe_rel(pool_dir / "fig3_circles_and_line.json"),
                    "helix_3d": _maybe_rel_glob(pool_dir, "fig1_helix_T*.json"),
                    "pca_2d": _maybe_rel(pool_dir / "fig4_pca_2d.json"),
                    "meta": _maybe_rel(pool_dir / "_meta.json"),
                },
            }
        )

    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_cells": len(cells),
        "cells": cells,
    }
    idx_path = OUT / "_index.json"
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, separators=(",", ":"), ensure_ascii=False)
    print(f"wrote {idx_path}  ({len(cells)} cells)")

    csv_path = OUT / "_l0_share.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "script",
                "L0_helix_r2",
                "peak_helix_r2",
                "peak_layer",
                "L0_over_peak",
            ]
        )
        for c in cells:
            if c["n_max"] != 100 or c["periods"] != [2, 5, 10, 100]:
                # The historical csv only covered the paper-default cell.
                continue
            w.writerow(
                [
                    c["model"],
                    c["script"],
                    _fmt(c["helix_r2_l0"]),
                    _fmt(c["helix_r2_peak"]),
                    c["peak_layer"] if c["peak_layer"] is not None else "",
                    _fmt(c["rho"]),
                ]
            )
    print(f"wrote {csv_path}")


def _maybe_rel(p: Path) -> str | None:
    return str(p.relative_to(OUT)) if p.exists() else None


def _maybe_rel_glob(dir_: Path, pattern: str) -> str | None:
    hits = sorted(dir_.glob(pattern))
    return str(hits[0].relative_to(OUT)) if hits else None


def _fmt(v) -> str:
    return "" if v is None else f"{v:.4f}"


if __name__ == "__main__":
    main()
