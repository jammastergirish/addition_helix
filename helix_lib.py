"""Shared helpers used by main.py, embed_control.py, and subspace_align.py.

Kept torch-free (numpy + sklearn only) so embed_control.py, which doesn't
load model weights, imports nothing heavy through it. Anything that actually
needs torch (pick_device in main/subspace_align) is small enough to keep
inline in each script.

What lives here:
  - Numeral renderers + format_number (12 scripts).
  - PERIODS, ALL_MODELS, ALL_SCRIPTS constants.
  - helix_basis B(a), fit_helix (numpy + sklearn).
  - get_num_layers / get_d_model (read HF model.config; no torch import).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# The four periods the paper discovered via Fourier analysis. T=2 captures
# parity, T=5 captures (a mod 5), T=10 the decimal units digit, T=100 the
# coarse position. Together with a linear magnitude axis they form a
# 9-parameter (1 + 2*4) encoding of any integer a — a "generalized helix".
PERIODS = [2, 5, 10, 100]

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

ALL_SCRIPTS = [
    "latin",
    "arabic",
    "persian",
    "devanagari",
    "thai",
    "chinese",
    "binary",
    "hexadecimal",
    "greek",
    "hebrew",
    "roman",
    "babylonian",
]


# ============================================================================
#  Helix basis B(a) — the central object of the paper.
# ============================================================================
def helix_basis(a, periods=PERIODS):
    """Return row-vector B(a) (or a matrix of rows if `a` is an array).

    Layout per row:  [a, cos(2πa/T_1), sin(2πa/T_1), cos(2πa/T_2), ...].
    """
    a = np.asarray(a, dtype=np.float64)
    cols = [a]
    for T in periods:
        cols.append(np.cos(2 * np.pi * a / T))
        cols.append(np.sin(2 * np.pi * a / T))
    return np.stack(cols, axis=-1)


def fit_helix(H, numbers, periods=PERIODS):
    """Solve  H ~= B @ W  by least squares; return W, intercept, R².

    sklearn stores coef_ as (n_outputs, n_features) = (d, n_basis). We
    transpose so callers get W of shape (n_basis, d) — convenient since
    they multiply  B @ W  with B of shape (N, n_basis).
    """
    B = helix_basis(numbers, periods=periods)
    reg = LinearRegression(fit_intercept=True).fit(B, H)
    H_hat = reg.predict(B)
    r2 = float(r2_score(H, H_hat, multioutput="variance_weighted"))
    return reg.coef_.T, reg.intercept_, r2


# ============================================================================
#  NUMERAL RENDERING
# ============================================================================
def to_roman(n: int) -> str:
    """Convert n in [0, ...) to its Roman numeral. n=0 -> 'nulla'.

    Not positional: 9 -> IX and 10 -> X have wildly different glyphs.
    """
    if n == 0:
        return "nulla"
    pairs = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    out = []
    for v, s in pairs:
        while n >= v:
            out.append(s)
            n -= v
    return "".join(out)


# Greek alphabetic numerals (Milesian system). 1-9 use one alphabet block,
# 10-90 use another. ADDITIVE: 23 = κ + γ = "κγ".
# Classical letter for 6 is digamma ϛ (U+03DB stigma in modern Unicode);
# for 90 it's koppa ϟ (U+03DF). Greek had no zero — we use "Ø" as a
# non-colliding placeholder.
GREEK_UNITS = ["", "α", "β", "γ", "δ", "ε", "ϛ", "ζ", "η", "θ"]
GREEK_TENS = ["", "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π", "ϟ"]


def to_greek(n: int) -> str:
    """Convert n in [0, 99] to its Greek alphabetic numeral."""
    if n == 0:
        return "Ø"
    if n < 0 or n > 99:
        raise ValueError(f"to_greek only supports 0..99, got {n}")
    tens, units = divmod(n, 10)
    return GREEK_TENS[tens] + GREEK_UNITS[units]


CHINESE_POSITIONAL = "〇一二三四五六七八九"


def to_chinese_positional(n: int) -> str:
    """Positional base-10 in CJK glyphs: 23 -> '二三'."""
    return "".join(CHINESE_POSITIONAL[int(d)] for d in str(n))


# Babylonian cuneiform: POSITIONAL at base 60, ADDITIVE within each column.
# Two wedges only: 𒁹 = 1 and 𒌋 = 10. Late-period zero placeholder is 𒑊.
BAB_ONE = "\U00012079"  # 𒁹  CUNEIFORM SIGN DISH (= 1)
BAB_TEN = "\U0001230b"  # 𒌋  CUNEIFORM SIGN U    (= 10)
BAB_ZERO = "\U0001244a"  # 𒑊  CUNEIFORM NUMERIC SIGN TWO ASH TENU


def _bab_column(v: int) -> str:
    """Additive rendering of 0..59 within a single sexagesimal column."""
    if v == 0:
        return BAB_ZERO
    tens, ones = divmod(v, 10)
    return BAB_TEN * tens + BAB_ONE * ones


def to_babylonian(n: int) -> str:
    """Convert n >= 0 to its Babylonian cuneiform numeral.

    Decomposes into base-60 columns, most-significant first, separated
    by spaces. Each column is rendered additively via _bab_column.
    """
    if n < 0:
        raise ValueError(f"to_babylonian only supports n >= 0, got {n}")
    if n == 0:
        return BAB_ZERO
    columns = []
    rest = n
    while rest > 0:
        rest, lsb = divmod(rest, 60)
        columns.append(lsb)
    columns.reverse()
    return " ".join(_bab_column(c) for c in columns)


# Hebrew alphabetic numerals (gematria). Like Greek, each letter has a
# fixed value; ADDITIVE. Special cases at 15/16 avoid the Tetragrammaton.
HEBREW_UNITS = ["", "א", "ב", "ג", "ד", "ה", "ו", "ז", "ח", "ט"]
HEBREW_TENS = ["", "י", "כ", "ל", "מ", "נ", "ס", "ע", "פ", "צ"]


def to_hebrew(n: int) -> str:
    """Convert n in [0, 99] to its Hebrew alphabetic numeral."""
    if n == 0:
        return "אפס"
    if n < 0 or n > 99:
        raise ValueError(f"to_hebrew only supports 0..99, got {n}")
    if n == 15:
        return "טו"
    if n == 16:
        return "טז"
    tens, units = divmod(n, 10)
    return HEBREW_TENS[tens] + HEBREW_UNITS[units]


def to_binary(n: int) -> str:
    """Convert n >= 0 to its base-2 string (no '0b' prefix)."""
    if n < 0:
        raise ValueError(f"to_binary only supports n >= 0, got {n}")
    return bin(n)[2:]


def to_hexadecimal(n: int) -> str:
    """Convert n >= 0 to its base-16 string (lowercase, no '0x' prefix)."""
    if n < 0:
        raise ValueError(f"to_hexadecimal only supports n >= 0, got {n}")
    return format(n, "x")


# Per-script digit-block offsets for the four positional Unicode scripts
# that map decimal digits 0–9 onto a contiguous codepoint range.
_DIGIT_BASE = {
    "arabic": 0x0660,  # Arabic-Indic ٠..٩
    "persian": 0x06F0,  # Extended Arabic-Indic / Persian ۰..۹
    "devanagari": 0x0966,  # Devanagari ०..९
    "thai": 0x0E50,  # Thai ๐..๙
}


def format_number(n: int, script: str) -> str:
    """Render an integer in the requested numeral script."""
    if script == "latin":
        return str(n)
    if script in _DIGIT_BASE:
        base = _DIGIT_BASE[script]
        return "".join(chr(base + int(d)) for d in str(n))
    if script == "chinese":
        return to_chinese_positional(n)
    if script == "binary":
        return to_binary(n)
    if script == "hexadecimal":
        return to_hexadecimal(n)
    if script == "greek":
        return to_greek(n)
    if script == "hebrew":
        return to_hebrew(n)
    if script == "roman":
        return to_roman(n)
    if script == "babylonian":
        return to_babylonian(n)
    raise ValueError(f"unknown script: {script!r}")


# ============================================================================
#  HF MODEL CONFIG INTROSPECTION  (no torch import needed — these just walk
#  attributes on `model.config`)
# ============================================================================
def get_num_layers(model) -> int:
    """Find the number of transformer blocks in a HF model.

    Different model families expose this under different attribute names;
    multi-component configs (Gemma 4, some VLMs) nest it in a sub-config.
    """
    cfg = model.config
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
    for sub in ("text_config", "language_model_config", "decoder"):
        if hasattr(cfg, sub):
            sub_cfg = getattr(cfg, sub)
            for attr in ("num_hidden_layers", "n_layer", "num_layers"):
                if hasattr(sub_cfg, attr):
                    return getattr(sub_cfg, attr)
    raise AttributeError(
        "Could not find layer count on model.config. Pass --layer "
        "explicitly to bypass the auto-detect."
    )


def get_d_model(cfg) -> int:
    """Find the hidden_size on an HF config, including nested sub-configs."""
    if hasattr(cfg, "hidden_size"):
        return cfg.hidden_size
    for sub in ("text_config", "language_model_config"):
        if hasattr(cfg, sub):
            sub_cfg = getattr(cfg, sub)
            if hasattr(sub_cfg, "hidden_size"):
                return sub_cfg.hidden_size
    raise AttributeError("no hidden_size on config")
