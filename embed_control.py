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
    uv run embed_control.py                       # all 8 models, all 8 scripts
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

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"


# ─── inlined helpers from main.py (kept separate so this script doesn't
#     drag in torch / matplotlib via main.py's imports) ──────────────────

def helix_basis(a, periods):
    """Trig basis B(a) = [a, cos(2πa/T), sin(2πa/T)] per period T."""
    a = np.asarray(a, dtype=np.float64)
    cols = [a]
    for T in periods:
        cols.append(np.cos(2 * np.pi * a / T))
        cols.append(np.sin(2 * np.pi * a / T))
    return np.stack(cols, axis=-1)


def to_roman(n: int) -> str:
    if n == 0:
        return "nulla"
    pairs = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
             (100, "C"),  (90, "XC"),  (50, "L"),  (40, "XL"),
             (10, "X"),   (9, "IX"),   (5, "V"),   (4, "IV"), (1, "I")]
    out = []
    for v, s in pairs:
        while n >= v:
            out.append(s); n -= v
    return "".join(out)


GREEK_UNITS = ["", "α", "β", "γ", "δ", "ε", "ϛ", "ζ", "η", "θ"]
GREEK_TENS  = ["", "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π", "ϟ"]


def to_greek(n: int) -> str:
    if n == 0:
        return "Ø"
    if n > 99:
        raise ValueError(f"to_greek: got {n}")
    t, u = divmod(n, 10)
    return GREEK_TENS[t] + GREEK_UNITS[u]


CHINESE_POSITIONAL = "〇一二三四五六七八九"


def to_chinese_positional(n: int) -> str:
    return "".join(CHINESE_POSITIONAL[int(d)] for d in str(n))


BAB_ONE  = "\U00012079"   # 𒁹
BAB_TEN  = "\U0001230B"   # 𒌋
BAB_ZERO = "\U0001244A"   # 𒑊 (late-period zero placeholder)


def _bab_column(v: int) -> str:
    if v == 0:
        return BAB_ZERO
    tens, ones = divmod(v, 10)
    return BAB_TEN * tens + BAB_ONE * ones


def to_babylonian(n: int) -> str:
    if n < 0:
        raise ValueError(f"to_babylonian: got {n}")
    if n == 0:
        return BAB_ZERO
    cols = []
    rest = n
    while rest > 0:
        rest, lsb = divmod(rest, 60)
        cols.append(lsb)
    cols.reverse()
    return " ".join(_bab_column(c) for c in cols)


def format_number(n: int, script: str) -> str:
    if script == "latin":
        return str(n)
    if script == "arabic":
        return "".join(chr(0x0660 + int(d)) for d in str(n))
    if script == "persian":
        return "".join(chr(0x06F0 + int(d)) for d in str(n))
    if script == "devanagari":
        return "".join(chr(0x0966 + int(d)) for d in str(n))
    if script == "chinese":
        return to_chinese_positional(n)
    if script == "greek":
        return to_greek(n)
    if script == "roman":
        return to_roman(n)
    if script == "babylonian":
        return to_babylonian(n)
    raise ValueError(f"unknown script: {script!r}")


# ─── config helpers ─────────────────────────────────────────────────────

ALL_MODELS = [
    "EleutherAI/pythia-6.9b",
    "EleutherAI/gpt-j-6b",
    "meta-llama/Llama-3.1-8B",
    "google/gemma-4-E4B",
    "google/gemma-4-31B",
    "allenai/Olmo-3-1125-32B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-32B",
]
ALL_SCRIPTS = ["latin", "arabic", "persian", "devanagari",
               "chinese", "greek", "roman", "babylonian"]


# Configurations to run. Mirrors run.sh's three passes.
def default_combos():
    combos = []
    # Pass 1: every script at the paper default
    for s in ALL_SCRIPTS:
        combos.append((s, 100, [2, 5, 10, 100]))
    # Pass 2/3: Babylonian wider window, both bases
    combos.append(("babylonian", 600, [2, 5, 10, 100]))
    combos.append(("babylonian", 600, [2, 5, 10, 60, 100]))
    return combos


def get_d_model(cfg) -> int:
    if hasattr(cfg, "hidden_size"):
        return cfg.hidden_size
    for sub in ("text_config", "language_model_config"):
        if hasattr(cfg, sub):
            sub_cfg = getattr(cfg, sub)
            if hasattr(sub_cfg, "hidden_size"):
                return sub_cfg.hidden_size
    raise AttributeError("no hidden_size on config")


def get_vocab_size(cfg, tokenizer) -> int:
    if hasattr(cfg, "vocab_size") and cfg.vocab_size:
        return cfg.vocab_size
    return len(tokenizer)


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
