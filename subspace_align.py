"""Subspace alignment (linear CKA) between L=0 and peak-layer
representations across (model, script) cells.

The motivation (from reviewer feedback): ρ alone says "depth did not
improve the helix score," but cannot distinguish:

    case             | ρ      | CKA(L=0, peak) | interpretation
    -----------------|--------|----------------|------------------------
    preserved        | high   | high           | structure passes through
    rebuilt          | high   | low            | depth destroys + rebuilds
    depth-amplified  | low    | high           | depth refines L=0 structure
    depth-built      | low    | low            | depth constructs the helix

For each (model, script) the script captures H[L=0] and H[peak_layer],
where peak_layer is read from the existing mean/fig_layer_sweep.json
(n_max=100, paper basis). It computes linear CKA both on the whole
representation and on the helix-projected subspace (B · W), and merges
results into out/_subspace_alignment.json (de-duplicating by
(model, script) so re-runs don't pile up).

Loading each model is the slow part (esp. 32B class), so the script
iterates over scripts inside one model-load.

Usage:
    uv run subspace_align.py                                 # all 8 models, latin only
    uv run subspace_align.py --scripts latin,devanagari,roman # multiple scripts
    uv run subspace_align.py --model X --scripts all          # one model, all 12 scripts
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from helix_lib import (  # noqa: E402
    ALL_MODELS,
    ALL_SCRIPTS,
    PERIODS,
    fit_helix,
    format_number,
    get_num_layers,
    helix_basis,
)

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"


# ── CKA ──────────────────────────────────────────────────────────────────


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment between two same-row matrices.

    X: (n, d_x), Y: (n, d_y). Mean-centers each, then:

        CKA = ||X^T Y||_F^2 / ( ||X^T X||_F * ||Y^T Y||_F ).

    Range [0, 1]; 1 means the row-similarity structures of X and Y are
    identical up to invertible linear transform. Insensitive to absolute
    scale or orthogonal rotation in feature space — exactly what we want
    for "is the representation the same."
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    XtY = Xc.T @ Yc
    XtX = Xc.T @ Xc
    YtY = Yc.T @ Yc
    num = float((XtY * XtY).sum())
    den = float(np.sqrt((XtX * XtX).sum() * (YtY * YtY).sum()))
    return num / den if den > 0 else 0.0


def pick_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_peak_layer(model: str, script: str) -> int | None:
    p = OUT / model.replace("/", "__") / script / "mean" / "fig_layer_sweep.json"
    if not p.exists():
        return None
    return int(json.loads(p.read_text())["peaks"]["helix_r2"]["layer"])


@torch.no_grad()
def collect_l0_and_peak(
    model, tokenizer, n_max: int, peak_layer: int, device, script: str
):
    """Forward each integer once; keep only hidden_states[0] and
    hidden_states[peak_layer], mean-pooled over the numeral's tokens."""
    # Match main.py's BOS handling exactly: fall back to EOS only when
    # bos_token_id is genuinely absent. Using `or` here would mis-treat a
    # valid bos_token_id of 0 as falsy and substitute EOS.
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    H_l0 = []
    H_peak = []
    for n in tqdm(range(n_max), desc=f"  {script} L=0,L={peak_layer}", leave=False):
        text = " " + format_number(n, script)
        numeral_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ]
        n_tok = numeral_ids.shape[1]
        if bos is not None:
            ids = torch.cat([torch.tensor([[bos]]), numeral_ids], dim=1)
            start, end = 1, 1 + n_tok
        else:
            ids, start, end = numeral_ids, 0, n_tok
        ids = ids.to(device)
        out = model(ids, output_hidden_states=True, use_cache=False)
        h0 = out.hidden_states[0][0, start:end, :].mean(dim=0)
        hpk = out.hidden_states[peak_layer][0, start:end, :].mean(dim=0)
        H_l0.append(h0.float().cpu().numpy())
        H_peak.append(hpk.float().cpu().numpy())
    return np.stack(H_l0), np.stack(H_peak)


def run_one_script(
    model, tokenizer, model_id: str, script: str, n_max: int, device, n_layers: int
) -> dict:
    peak_layer = load_peak_layer(model_id, script)
    if peak_layer is None:
        return {"model": model_id, "script": script, "error": "no layer_sweep.json"}
    if peak_layer == 0:
        # CKA(L=0, L=0) is trivially 1; record it without re-running.
        return {
            "model": model_id,
            "script": script,
            "n_max": int(n_max),
            "peak_layer": 0,
            "n_layers": n_layers,
            "r2_l0": None,
            "r2_peak": None,
            "rho": 1.0,
            "cka_full": 1.0,
            "cka_helix": 1.0,
            "trivial": True,
        }

    H_l0, H_peak = collect_l0_and_peak(
        model, tokenizer, n_max, peak_layer, device, script=script
    )

    numbers = np.arange(n_max)
    W_l0, _, r2_l0 = fit_helix(H_l0, numbers)
    W_peak, _, r2_peak = fit_helix(H_peak, numbers)
    rho = r2_l0 / r2_peak if r2_peak > 0 else None

    cka_full = linear_cka(H_l0, H_peak)
    B = helix_basis(numbers)
    cka_helix = linear_cka(B @ W_l0, B @ W_peak)

    print(
        f"  {script:11} L={peak_layer:>3}  "
        f"R²={r2_l0:.3f}/{r2_peak:.3f}  ρ={rho:.2f}  "
        f"CKA(full)={cka_full:.2f}  CKA(helix)={cka_helix:.2f}"
    )

    return {
        "model": model_id,
        "script": script,
        "n_max": int(n_max),
        "peak_layer": int(peak_layer),
        "n_layers": int(n_layers),
        "r2_l0": round(r2_l0, 6),
        "r2_peak": round(r2_peak, 6),
        "rho": None if rho is None else round(rho, 6),
        "cka_full": round(cka_full, 6),
        "cka_helix": round(cka_helix, 6),
    }


def _load_existing_keys() -> set[tuple[str, str]]:
    """Return (model, script) keys already present in _subspace_alignment.json
    so we can skip the expensive forward passes on re-runs."""
    out_path = OUT / "_subspace_alignment.json"
    if not out_path.exists():
        return set()
    try:
        rows = json.loads(out_path.read_text()).get("results", [])
    except json.JSONDecodeError:
        return set()
    # Only count successful rows -- skip-and-retry on prior errors.
    return {
        (r["model"], r["script"])
        for r in rows
        if "error" not in r and r.get("rho") is not None
    }


def run_one_model(
    model_id: str,
    scripts,
    n_max: int,
    dtype: torch.dtype,
    existing: set[tuple[str, str]],
) -> list[dict]:
    """Load the model once, iterate scripts.

    Skips any (model, script) already in `existing` -- which means a
    repeated `uv run subspace_align.py` only pays the model-load cost
    for new cells. Returns ONLY the newly-computed rows; merging is the
    caller's job (so the merge stays atomic).
    """
    pending = [s for s in scripts if (model_id, s) not in existing]
    if not pending:
        print(f"\n=== {model_id} === (all {len(scripts)} cells cached, skip)")
        return []

    print(f"\n=== {model_id} === ({len(pending)}/{len(scripts)} cells to compute)")
    tok = AutoTokenizer.from_pretrained(model_id)
    print(f"  loading ({dtype}) ...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    device = pick_device()
    model = model.to(device).eval()
    n_layers = get_num_layers(model)

    results = []
    for script in pending:
        try:
            results.append(
                run_one_script(model, tok, model_id, script, n_max, device, n_layers)
            )
        except Exception as e:
            print(f"  !! {script} FAILED: {e}")
            results.append({"model": model_id, "script": script, "error": str(e)})

    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def _merge_results(out_path: Path, new_results: list[dict]) -> list[dict]:
    """Merge new_results into the existing JSON, replacing rows that share
    (model, script). Lets `--scripts X` runs incrementally build up the
    full matrix without losing prior cells."""
    existing = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text()).get("results", [])
        except json.JSONDecodeError:
            pass
    by_key = {(r["model"], r["script"]): r for r in existing}
    for r in new_results:
        by_key[(r["model"], r["script"])] = r
    return list(by_key.values())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--model", default=None, help="single HF id; default = all 8.")
    ap.add_argument(
        "--scripts",
        default="latin",
        help="comma-separated script list, or 'all' for every script. Default: latin.",
    )
    ap.add_argument("--n_max", type=int, default=100)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    args = ap.parse_args()

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        args.dtype
    ]
    models = [args.model] if args.model else ALL_MODELS
    scripts = (
        ALL_SCRIPTS
        if args.scripts == "all"
        else [s.strip() for s in args.scripts.split(",")]
    )

    existing = _load_existing_keys()
    if existing:
        print(
            f"({len(existing)} (model, script) cells already in _subspace_alignment.json — will skip)"
        )

    all_new = []
    for m in models:
        try:
            all_new.extend(run_one_model(m, scripts, args.n_max, dtype, existing))
        except Exception as e:
            print(f"  !! {m} FAILED: {e}")
            for s in scripts:
                if (m, s) not in existing:
                    all_new.append({"model": m, "script": s, "error": str(e)})

    out_path = OUT / "_subspace_alignment.json"
    merged = _merge_results(out_path, all_new)
    out_path.write_text(
        json.dumps(
            {
                "kind": "subspace_alignment_cka",
                "generated_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "n_max": args.n_max,
                "results": merged,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(
        f"\nwrote {out_path}  ({len(merged)} cells total, {len(all_new)} updated this run)"
    )


if __name__ == "__main__":
    main()
