# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.40",
#     "python-dotenv>=1.0",
#     "numpy",
#     "scikit-learn",
# ]
# ///
"""Random-embedding control across models, scripts, and bases.

For each (model, script, n_max, periods) combination, this script:
  1. Loads the tokenizer (no model weights).
  2. Builds a RANDOM embedding matrix at the model's d_model.
  3. Tokenizes each integer's rendered string and mean-pools the random
     embeddings to get a synthetic L=0 representation H_random.
  4. Fits the trig basis to H_random and reports helix R².
  5. Compares to the LEARNED-embedding L=0 helix R² that the main sweep
     captured (looked up from out/<model>/<script>/<pool>/fig_layer_sweep.json).

Interpretation:
  - random_R² ≈ learned_R²: the L=0 helix is mechanical (renderer + pooling).
  - learned_R² ≫ random_R²: the embedding table absorbed structure
    beyond what the renderer mechanically produces.

For Babylonian on every model the gap is tiny (the additive within-column
renderer + mean-pool produces the helix mechanically). For per-digit
positional scripts the gap is large (random per-digit vectors don't encode
digit value, so the helix on those cells reflects learned digit semantics).

Writes out/_random_embed_control.json. Run as the last pass after the
main sweep + aggregate.

Usage:
    uv run embed_control.py                       # all 8 models, all 12 scripts
    uv run embed_control.py --model X --script Y  # one combo
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

import numpy as np  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from transformers import AutoConfig, AutoTokenizer  # noqa: E402

from helix_lib import (  # noqa: E402
    ALL_MODELS, ALL_SCRIPTS,
    format_number, get_d_model, helix_basis,
)

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"


# Configurations to run. Mirrors run.sh's three passes.
def default_combos():
    combos = []
    # Pass 1: every script at the paper default
    for s in ALL_SCRIPTS:
        combos.append((s, 100, [2, 5, 10, 100]))
    # Babylonian wider window, both bases
    combos.append(("babylonian",  600, [2, 5, 10, 100]))
    combos.append(("babylonian",  600, [2, 5, 10, 60, 100]))
    # Binary wider window, both bases (paper basis vs binary-native)
    combos.append(("binary",      1024, [2, 5, 10, 100]))
    combos.append(("binary",      1024, [2, 4, 8, 16, 32, 64]))
    # Hex wider window, both bases (paper basis vs hex-native)
    combos.append(("hexadecimal", 1024, [2, 5, 10, 100]))
    combos.append(("hexadecimal", 1024, [16, 32, 64, 256]))
    return combos


def get_vocab_size(cfg, tokenizer) -> int:
    # Size the random embedding table to cover every id the tokenizer can
    # emit. cfg.vocab_size is sometimes smaller than len(tokenizer) (added /
    # special tokens), which would make rand_embed[ids] raise IndexError if
    # a rendered numeral ever tokenized to a high id. Take the max so the
    # lookup is always in range.
    cfg_vocab = getattr(cfg, "vocab_size", 0) or 0
    return max(cfg_vocab, len(tokenizer))


def fit_r2(H, numbers, periods):
    B = helix_basis(numbers, periods=periods)
    reg = LinearRegression(fit_intercept=True).fit(B, H)
    return float(r2_score(H, reg.predict(B), multioutput="variance_weighted"))


def load_learned_L0_r2(model: str, script: str, n_max: int, periods: list[int]):
    """Look up the L=0 helix R² that the main sweep already computed.
    Path mirrors run.sh marker_for() rules."""
    pool = "mean" if n_max == 100 else f"mean_n{n_max}"
    if periods != [2, 5, 10, 100]:
        pool = f"{pool}_p{'-'.join(map(str, periods))}"
    p = OUT / model.replace("/", "__") / script / pool / "fig_layer_sweep.json"
    if not p.exists():
        return None
    return float(json.loads(p.read_text())["helix_r2"][0])


# ─── runner ─────────────────────────────────────────────────────────────

def run_model(model: str, combos, seed: int) -> list[dict]:
    print(f"\n=== {model} ===")
    tokenizer = AutoTokenizer.from_pretrained(model)
    cfg = AutoConfig.from_pretrained(model)
    d_model = get_d_model(cfg)
    vocab = get_vocab_size(cfg, tokenizer)
    print(f"  d_model={d_model}  vocab={vocab}")

    rng = np.random.default_rng(seed)
    rand_embed = rng.standard_normal((vocab, d_model), dtype=np.float32) / np.sqrt(d_model)

    rows = []
    for script, n_max, periods in combos:
        numbers = np.arange(n_max)
        H = np.zeros((n_max, d_model), dtype=np.float32)
        for n in numbers:
            text = " " + format_number(int(n), script)
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            H[n] = rand_embed[ids].mean(axis=0)

        r2_rand    = fit_r2(H, numbers, periods)
        r2_learned = load_learned_L0_r2(model, script, n_max, periods)
        delta      = None if r2_learned is None else r2_learned - r2_rand

        row = {
            "model":         model,
            "script":        script,
            "n_max":         int(n_max),
            "periods":       periods,
            "random_r2":     round(r2_rand, 6),
            "learned_L0_r2": None if r2_learned is None else round(r2_learned, 6),
            "delta":         None if delta is None else round(delta, 6),
            "seed":          int(seed),
            "d_model":       int(d_model),
        }
        rows.append(row)

        delta_s = f"+{delta:+.4f}" if delta is not None and delta >= 0 else (f"{delta:+.4f}" if delta is not None else "—")
        learned_s = f"{r2_learned:.4f}" if r2_learned is not None else "—"
        print(f"  {script:11} n={n_max:>3} T={periods}  random={r2_rand:.4f}  learned_L0={learned_s}  Δ={delta_s}")

    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model",  default=None, help="single HF id; default = all 8.")
    ap.add_argument("--script", default=None, help="single script; default = all 8.")
    ap.add_argument("--seed",   type=int, default=0)
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    combos = default_combos()
    if args.script:
        combos = [c for c in combos if c[0] == args.script]
    models = [args.model] if args.model else ALL_MODELS

    all_rows = []
    for m in models:
        try:
            all_rows.extend(run_model(m, combos, args.seed))
        except Exception as e:
            print(f"  !! FAILED {m}: {e}")

    out_path = OUT / "_random_embed_control.json"
    out_path.write_text(json.dumps({
        "kind":         "random_embed_control",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed":         args.seed,
        "results":      all_rows,
    }, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_path}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
