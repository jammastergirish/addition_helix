# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.1",
#     "transformers>=4.40",
#     "accelerate>=0.30",
#     "hf-transfer>=0.1.6",
#     "python-dotenv>=1.0",
#     "numpy",
#     "scipy",
#     "scikit-learn",
#     "matplotlib",
#     "tqdm",
# ]
# ///
"""
Language Models Use Trigonometry to Do Addition
================================================
A from-scratch replication of Kantamneni & Tegmark (arXiv:2502.00873, 2025),
extended with optional cross-script rendering of the integers.

THE CENTRAL CLAIM
-----------------
For an integer a in [0, 99], the LLM's residual stream encodes a as

    h(a) ~= W^T @ B(a)
    B(a) = [a, cos(2 pi a/T_1), sin(2 pi a/T_1), ..., cos(2 pi a/T_K), sin(2 pi a/T_K)]
    T    = [2, 5, 10, 100]

In words: ONE number-line + FOUR modular circles. Each (cos, sin) pair
places a on a circle of period T; the linear coordinate lifts the point
vertically by a. Stacked, this is a *generalized helix*. The four periods
were not chosen by hand -- they emerge from a Fourier analysis of the
residual stream (see `plot_fourier_and_pc1`).

The model performs ADDITION by rotating each circle: angle(a + b) =
angle(a) + angle(b) on each clock. This is the paper's "Clock" algorithm.

HOW TO READ THIS FILE
---------------------
The code follows the paper's discovery process:

  STEP 1  Pull residual-stream activations h(a) for a = 0..99.
          See `collect_activations` and the `main` flow.

  STEP 2  FFT the activations -> *discover* the periods.
          The peaks at f = 1/2, 1/5, 1/10, 1/100 are not put in by us --
          they emerge from the data. See `plot_fourier_and_pc1`.

  STEP 3  Fit the helix model B(a) -> h(a) by linear regression.
          Report R^2 so we know how much variance the helix explains.
          See `fit_helix`.

  STEP 4  Project h(a) onto each (cos_T, sin_T) plane -- you'll SEE
          the modular circles in the residual stream. See
          `plot_circles_and_line`.

  STEP 5  Project h(a) onto (cos_10, sin_10, lin) -- the iconic 3D
          T=10 helix. See `plot_helix_3d`.

Each plot has a counterpart figure in the paper; filenames match.

NUMERAL SCRIPT  (`--script`)
----------------------------
The default is Latin digits (`0, 1, 2, ..., 99`) -- the paper's setup.
You can also render the integers in any of seven other numeral systems
to probe whether the helix is glyph-induced or value-induced:

  positional base-10 (helix predicted to appear):
    latin       0, 1, 2, ..., 99                  (single tokens on Pythia)
    arabic      ٠, ١, ٢, ..., ٩٩                  (Arabic-Indic digits)
    persian     ۰, ۱, ۲, ..., ۹۹                  (different Unicode block from arabic)
    devanagari  ०, १, २, ..., ९९                  (Hindi digits)
    chinese     〇, 一, 二, ..., 九九             (CJK positional, e.g. 23 = 二三)

  non-positional / additive (helix predicted to vanish):
    greek       α, β, γ, ..., ϟθ                  (Greek alphabetic numerals)
    roman       I, II, III, ..., XCIX             (Roman)

  positional base-60, additive within each column (mixed -- helix uncertain):
    babylonian  𒁹, 𒁹𒁹, ..., 𒁹 𒌋𒌋𒌋𒁹𒁹𒁹𒁹𒁹𒁹𒁹𒁹𒁹   (cuneiform; 23 = 𒌋𒌋𒁹𒁹𒁹)

POOLING  (`--pool`)
-------------------
Latin numbers are single tokens on Pythia, so reading "the last token"
at the operand position is unambiguous. For multi-token numerals
(everything else on Pythia; everything including Latin on Llama 3),
the last token is just the last byte/digit -- which CYCLES with period
10 across `a` by construction. That cycling produces fake helix peaks
in the FFT without the model doing anything.

  --pool last  (the paper's default for single-token Latin)
  --pool mean  (averages across the numeral's sub-tokens; the honest
                cross-script test)

Run:
    uv run main.py --model EleutherAI/pythia-6.9b
    uv run main.py --model meta-llama/Llama-3.1-8B --script arabic --pool mean
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

# Load .env BEFORE importing transformers/huggingface_hub. Those libraries
# read HF_TOKEN and HF_HUB_ENABLE_HF_TRANSFER from os.environ at import
# time, so we must populate them first.
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
# Use the Rust-based hf_transfer downloader: 5-10x faster on big shards.
# If it ever silently stalls at 0%, set HF_HUB_ENABLE_HF_TRANSFER=0 in
# .env to fall back to the pure-Python downloader.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150})


def _configure_fonts() -> None:
    """Pick the first font in a fallback stack that actually exists, and
    set matplotlib's per-character fallback so non-Latin glyphs render
    instead of showing as empty boxes.

    matplotlib >= 3.6 walks `font.family` in order for *every character*,
    using the first font that contains that glyph. So a wide list with
    DejaVu Sans first (for Latin/Greek) followed by script-specific
    fonts covers most of what we throw at it. Cuneiform is the one gap
    on stock macOS -- if Noto Sans Cuneiform isn't installed, those
    plots will still box. See README for install instructions.
    """
    import matplotlib.font_manager as fm
    candidates = [
        "DejaVu Sans",            # Latin, Greek, Cyrillic, much of BMP
        "Arial Unicode MS",       # very broad coverage on macOS
        "Geeza Pro",              # Arabic, Persian
        "Damascus",               # Arabic alt
        "Devanagari Sangam MN",   # Devanagari (Hindi)
        "Kohinoor Devanagari",    # Devanagari alt
        "Hiragino Sans GB",       # CJK simplified Chinese (visible to matplotlib;
                                  # PingFang ships as .ttc and is not enumerated)
        "Heiti TC",               # CJK traditional Chinese
        "Noto Sans Cuneiform",    # Babylonian; install via `brew install --cask
                                  # font-noto-sans-cuneiform`
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    stack = [name for name in candidates if name in available]
    if stack:
        plt.rcParams["font.family"] = stack
    plt.rcParams["axes.unicode_minus"] = False
    missing = [c for c in candidates if c not in available]
    if missing:
        # Quiet note rather than a warning -- only matters for plotting,
        # never for the analysis itself.
        print(f"(matplotlib note: missing fonts {missing}; some scripts may still box)")


_configure_fonts()

# The four periods the paper discovered via Fourier analysis. T=2 captures
# parity, T=5 captures (a mod 5), T=10 the decimal units digit, T=100 the
# coarse position. Together with a linear magnitude axis they form a
# 9-parameter (1 + 2*4) encoding of any integer a -- a "generalized helix".
PERIODS = [2, 5, 10, 100]
OUT_DIR = Path(__file__).parent


# ============================================================================
#  HELPERS
# ============================================================================
def get_num_layers(model) -> int:
    """Find the number of transformer blocks in a HF model.

    Different model families expose this under different attribute names,
    and recent multi-component models (Gemma 4, some vision-language
    models) nest it inside a sub-config. We try the common paths in
    order, then fall back to peeking at the hidden_states length.
    """
    cfg = model.config
    # Most architectures (Pythia, GPT-2, Llama, Mistral, Qwen, ...)
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
    # Multi-component configs (Gemma 4, some VLMs): the language-model
    # block lives in a sub-config.
    for sub in ("text_config", "language_model_config", "decoder"):
        if hasattr(cfg, sub):
            sub_cfg = getattr(cfg, sub)
            for attr in ("num_hidden_layers", "n_layer", "num_layers"):
                if hasattr(sub_cfg, attr):
                    return getattr(sub_cfg, attr)
    raise AttributeError(
        "Could not find layer count on model.config. Pass --layer explicitly "
        "to bypass the auto-detect.")


def pick_device() -> torch.device:
    """Prefer Apple MPS (Metal), then NVIDIA CUDA, then CPU.

    On Apple Silicon, the M-series chips share one big pool of RAM
    between CPU and GPU (this is "unified memory"). That means we can
    load a 12 GB model with ~12 GB of total memory, not 12 GB GPU + 12
    GB CPU like on a CUDA box. With 128 GB unified memory, GPT-J-6B
    and Pythia-6.9B run comfortably in bf16.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------------
#  The helix basis B(a)
#
#  This is THE central object of the paper. Given an integer a, B(a) is a
#  (1 + 2K)-dimensional feature vector that captures both magnitude and
#  multiple kinds of modular structure.
# ----------------------------------------------------------------------------
def helix_basis(a, periods=PERIODS):
    """Return the row-vector B(a) (or a matrix of rows if `a` is an array).

    Layout of a single row B(a):
        column 0     : a                      <- linear magnitude
        columns 1, 2 : cos(2*pi*a/T_1), sin(2*pi*a/T_1)   <- circle of period T_1
        columns 3, 4 : cos(2*pi*a/T_2), sin(2*pi*a/T_2)   <- circle of period T_2
        ...

    GEOMETRY
    --------
    A (cos(2*pi*a/T), sin(2*pi*a/T)) pair is a point on the UNIT CIRCLE
    at angle theta = 2*pi*a/T radians. As a goes 0, 1, 2, ..., that point
    rotates by 2*pi/T radians per step -- T steps to come back home.
    Integers sharing the same (a mod T) thus land at the SAME point on
    the T-circle, which is how the model can read "the units digit"
    without ever computing a % 10.

    Addition becomes adding angles -- the "Clock" algorithm.
    """
    a = np.asarray(a, dtype=np.float64)
    cols = [a]                                # the linear "rise" of the helix
    for T in periods:
        cols.append(np.cos(2 * np.pi * a / T))
        cols.append(np.sin(2 * np.pi * a / T))
    return np.stack(cols, axis=-1)


def basis_idx(feature, periods=PERIODS) -> int:
    """Look up the column index of a named feature in B(a).

    feature = "lin"          -> column 0
    feature = ("cos", T)     -> column for cos(2*pi*a/T)
    feature = ("sin", T)     -> column for sin(2*pi*a/T)
    """
    if feature == "lin":
        return 0
    kind, T = feature
    i = periods.index(T)
    return 1 + 2 * i + (0 if kind == "cos" else 1)


# ============================================================================
#  NUMERAL SCRIPT RENDERING  (only matters when `--script` != latin)
# ============================================================================
def to_roman(n: int) -> str:
    """Convert a non-negative integer to its Roman-numeral string.

    Roman numerals don't have a true zero -- we use "nulla" for n=0.
    For n=1..99 the output is standard: I, II, III, IV, ..., XCIX.

    Key property: NOT a positional system. Adjacent integers (e.g. 9 -> 10)
    have wildly different glyph sequences (IX -> X). So the model has no
    reason to wire up a period-10 structure when it sees Roman numerals --
    which is exactly the prediction the helix theory makes.
    """
    if n == 0:
        return "nulla"
    pairs = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
             (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
             (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = []
    for v, s in pairs:
        while n >= v:
            out.append(s)
            n -= v
    return "".join(out)


# Greek alphabetic numerals (Milesian system). 1-9 use one alphabet block,
# 10-90 use another. ADDITIVE: 23 = κ + γ = "κγ".
# Notes:
#   * Classical letter for 6 is digamma ϛ (U+03DB stigma in modern Unicode).
#   * Classical letter for 90 is koppa ϟ (U+03DF).
#   * Greek had no zero -- we use "Ø" as a non-colliding placeholder.
GREEK_UNITS = ["", "α", "β", "γ", "δ", "ε", "ϛ", "ζ", "η", "θ"]
GREEK_TENS  = ["", "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π", "ϟ"]


def to_greek(n: int) -> str:
    """Convert n in [0, 99] to its Greek alphabetic numeral.

    Like Roman, this is NOT positional. 9 -> 10 changes glyph entirely
    (θ -> ι), and 23 is κγ -- a 'tens' letter then a 'units' letter,
    not two copies of a digit set. Helix should vanish.
    """
    if n == 0:
        return "Ø"
    if n < 0 or n > 99:
        raise ValueError(f"to_greek only supports 0..99, got {n}")
    tens, units = divmod(n, 10)
    return GREEK_TENS[tens] + GREEK_UNITS[units]


# CJK digits used in POSITIONAL Chinese (e.g. 23 -> 二三, not 二十三).
CHINESE_POSITIONAL = "〇一二三四五六七八九"


def to_chinese_positional(n: int) -> str:
    """23 -> '二三', 100 -> '一〇〇'.  Positional base-10 in CJK glyphs.

    The OTHER way Chinese writes numbers uses place names (十=ten,
    百=hundred, 千=thousand), e.g. 23 = 二十三 ("two-ten-three"), which is
    additive in spirit (closer to Roman). We deliberately use the
    positional form here so this script stays base-10 positional.
    """
    return "".join(CHINESE_POSITIONAL[int(d)] for d in str(n))


# Babylonian cuneiform: POSITIONAL at base 60, ADDITIVE within each column.
# Two wedges only:  𒁹 = 1  and  𒌋 = 10. Numbers 0-59 fit in one column;
# 60-99 use two columns separated by a space. The Babylonians had no zero
# until very late -- we use the late-period placeholder 𒑊 for n=0 and for
# an empty second column (e.g. 60 = "𒁹 𒑊").
#
# Examples:  23 = 𒌋𒌋𒁹𒁹𒁹           (2 tens + 3 ones, single column)
#            59 = 𒌋𒌋𒌋𒌋𒌋𒁹𒁹𒁹𒁹𒁹𒁹𒁹𒁹𒁹   (5 tens + 9 ones)
#            60 = 𒁹 𒑊                (one sixty, zero ones)
#            61 = 𒁹 𒁹
#            99 = 𒁹 𒌋𒌋𒌋𒁹𒁹𒁹𒁹𒁹𒁹𒁹𒁹𒁹
#
# Helix prediction is genuinely uncertain here: within a column the system
# is additive (Roman-like, helix-killing), but across the 60-boundary it
# becomes positional (helix-friendly). For our 0-99 range the only
# 60-wraparound happens once, at n=60, so we'd more likely see *within-
# column* period-10 structure than the textbook period-60 structure of
# Babylonian. Whether the model has even seen enough cuneiform during
# pre-training to encode any of this is a further unknown.
BAB_ONE  = "\U00012079"   # 𒁹  CUNEIFORM SIGN DISH        (= 1)
BAB_TEN  = "\U0001230B"   # 𒌋  CUNEIFORM SIGN U           (= 10)
BAB_ZERO = "\U0001244A"   # 𒑊  CUNEIFORM NUMERIC SIGN TWO ASH TENU (late zero)


def to_babylonian(n: int) -> str:
    """Convert n in [0, 215999] to its Babylonian cuneiform numeral.

    Each sexagesimal column is rendered additively as
        (count of 𒌋 = 10) + (count of 𒁹 = 1)
    via `one_column`. Multiple columns are separated by a space, with
    the most-significant column on the left -- the same convention used
    in the standard "0; 30; 0" transliteration of cuneiform.

    Examples:
        23   = 𒌋𒌋𒁹𒁹𒁹            (single column, two 10s + three 1s)
        59   = 𒌋𒌋𒌋𒌋𒌋𒁹𒁹𒁹𒁹𒁹𒁹𒁹𒁹𒁹
        60   = 𒁹 𒑊                  (one in 60s column, zero in 1s)
        120  = 𒁹𒁹 𒑊                (two 60s, zero 1s)
        599  = 𒁹𒁹𒁹𒁹𒁹𒁹𒁹𒁹𒁹 𒌋𒌋𒌋𒌋𒌋𒁹𒁹𒁹𒁹𒁹𒁹𒁹𒁹𒁹
        600  = 𒌋 𒑊                  (one ten in 60s column = 600, zero 1s)
        3600 = 𒁹 𒑊 𒑊                (one in 3600s column, zero zero)
    """
    if n < 0:
        raise ValueError(f"to_babylonian only supports n >= 0, got {n}")
    if n == 0:
        return BAB_ZERO

    def one_column(v: int) -> str:
        # Additive rendering of 0..59 within a single sexagesimal column.
        if v == 0:
            return BAB_ZERO
        tens, ones = divmod(v, 10)
        return BAB_TEN * tens + BAB_ONE * ones

    # Decompose into base-60 columns, most-significant first.
    columns = []
    rest = n
    while rest > 0:
        rest, lsb = divmod(rest, 60)
        columns.append(lsb)
    columns.reverse()
    return " ".join(one_column(c) for c in columns)


def format_number(n: int, script: str) -> str:
    """Render an integer in the requested numeral script.

    Maps decimal digits to Unicode equivalents in the chosen script.
    For Roman/Greek, delegates to dedicated helpers.
    """
    if script == "latin":
        return str(n)
    if script == "arabic":
        # Arabic-Indic digits live at U+0660 (٠) through U+0669 (٩).
        return "".join(chr(0x0660 + int(d)) for d in str(n))
    if script == "persian":
        # Extended Arabic-Indic / Persian digits live at U+06F0 (۰) through
        # U+06F9 (۹). Different Unicode block from `arabic`; some glyphs
        # are visually identical, others aren't.
        return "".join(chr(0x06F0 + int(d)) for d in str(n))
    if script == "devanagari":
        # Devanagari digits at U+0966 (०) through U+096F (९).
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


# ============================================================================
#  ACTIVATION COLLECTION
# ============================================================================
@torch.no_grad()
def collect_activations(model, tokenizer, numbers, layer, device,
                         script="latin", pool="mean"):
    """For each n, feed '<bos> {numeral(n)}' and AGGREGATE the residual
    stream over the numeral's tokens at the chosen layer.

    WHY POOLING MATTERS FOR CROSS-SCRIPT COMPARISON
    -----------------------------------------------
    For Latin on Pythia, " {n}" is usually a single BPE token, so `pool`
    doesn't matter -- there's only one position to read. For Arabic,
    Chinese, Roman, etc. (or Latin on Llama 3, which is digit-by-digit),
    a numeral is several sub-tokens and the choice matters:

      pool="last"  -- read the very last sub-token's activation.
                      For "٢٣" that's the second byte of "٣" alone.
                      So 23, 33, 43, ... all read the SAME thing
                      (since they all end in "٣"). This bakes a
                      period-10 cycle into the data by construction
                      and produces "helix-like" FFT peaks for reasons
                      unrelated to anything the model learned.

      pool="mean"  -- average activations across ALL sub-tokens of the
                      numeral. Removes the last-byte-only artifact;
                      every sub-token contributes equally.

    Default is "mean".

    Returns: H of shape (len(numbers), d_model).
    """
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id

    activations = []
    for n in tqdm(numbers, desc="forward passes"):
        text = f" {format_number(int(n), script)}"
        # Tokenize the numeral WITHOUT BOS so we know how many tokens it
        # occupies. Then prepend BOS for the forward pass.
        numeral_ids = tokenizer(text, add_special_tokens=False,
                                 return_tensors="pt")["input_ids"]
        n_numeral_tokens = numeral_ids.shape[1]
        if bos is not None:
            ids = torch.cat([torch.tensor([[bos]]), numeral_ids], dim=1)
            # Numeral occupies positions [1, n_numeral_tokens] inclusive.
            start, end = 1, 1 + n_numeral_tokens
        else:
            ids = numeral_ids
            start, end = 0, n_numeral_tokens
        ids = ids.to(device)
        # hidden_states is a tuple of (num_layers + 1) tensors:
        #   hidden_states[0]   = output of token+positional embedding
        #   hidden_states[L]   = residual stream AFTER block L-1
        #   hidden_states[-1]  = final residual stream
        out = model(ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[layer][0]                # (T, d)
        if pool == "last":
            h = hs[-1, :]
        elif pool == "mean":
            h = hs[start:end, :].mean(dim=0)
        else:
            raise ValueError(f"unknown pool: {pool!r}")
        activations.append(h.float().cpu().numpy())
    return np.stack(activations, axis=0)


# ============================================================================
#  ALL-LAYERS COLLECTION  (for the layer sweep)
# ============================================================================
@torch.no_grad()
def collect_activations_all_layers(model, tokenizer, numbers, device,
                                     script="latin", pool="mean"):
    """Same as `collect_activations` but returns activations at EVERY layer
    in one forward pass per integer.

    Returns H of shape (n_numbers, n_layers + 1, d_model).
    H[a, L, :] is h_L(a), the residual stream for integer `a` at
    hidden_states[L]. Used by the layer sweep to find which depth
    (if any) the helix forms at, without committing to a fixed layer.
    """
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id

    all_acts = []
    for n in tqdm(numbers, desc="forward passes"):
        text = f" {format_number(int(n), script)}"
        numeral_ids = tokenizer(text, add_special_tokens=False,
                                 return_tensors="pt")["input_ids"]
        n_numeral_tokens = numeral_ids.shape[1]
        if bos is not None:
            ids = torch.cat([torch.tensor([[bos]]), numeral_ids], dim=1)
            start, end = 1, 1 + n_numeral_tokens
        else:
            ids = numeral_ids
            start, end = 0, n_numeral_tokens
        ids = ids.to(device)
        out = model(ids, output_hidden_states=True, use_cache=False)
        # For each layer L, pool over the numeral's positions.
        if pool == "last":
            per_layer = torch.stack(
                [h[0, -1, :].float().cpu() for h in out.hidden_states], dim=0)
        else:  # mean
            per_layer = torch.stack(
                [h[0, start:end, :].mean(dim=0).float().cpu()
                 for h in out.hidden_states], dim=0)
        all_acts.append(per_layer)
    return torch.stack(all_acts, dim=0).numpy()


def run_sweep_analysis(H_all, numbers, periods=PERIODS):
    """Per-layer metrics. Returns three arrays of length n_layers+1:
       pc1_r2[L]    -- linear-fit R² of PC1(H_L) vs numbers (the "spine")
       helix_r2[L]  -- variance-weighted R² of the trig basis fit
       pca_kd_r2[L] -- variance-weighted R² of the best K-D approximation,
                       where K = 1 + 2*len(periods)
                       (Eckart-Young upper bound on any K-D fit)
    """
    n_hidden = H_all.shape[1]
    pc1_r2 = np.zeros(n_hidden)
    helix_r2 = np.zeros(n_hidden)
    n_basis = 1 + 2 * len(periods)
    pca_kd_r2 = np.zeros(n_hidden)
    B = helix_basis(numbers, periods=periods)
    for L in range(n_hidden):
        H_L = H_all[:, L, :]
        # PC1 R²: how linear is the dominant direction in `a`?
        pca = PCA(n_components=min(10, H_L.shape[0], H_L.shape[1])).fit(H_L)
        pc1 = pca.transform(H_L)[:, 0]
        slope, _ = np.polyfit(numbers, pc1, 1)
        if slope < 0:
            pc1 = -pc1
        slope, intercept = np.polyfit(numbers, pc1, 1)
        pc1_r2[L] = r2_score(pc1, slope * numbers + intercept)
        # Helix R²
        reg = LinearRegression().fit(B, H_L)
        helix_r2[L] = r2_score(H_L, reg.predict(B),
                                multioutput="variance_weighted")
        # K-D PCA upper bound (same dimensionality as the trig basis).
        pca_kd = PCA(n_components=min(n_basis, *H_L.shape)).fit(H_L)
        pca_kd_r2[L] = r2_score(
            H_L, pca_kd.inverse_transform(pca_kd.transform(H_L)),
            multioutput="variance_weighted")
    return pc1_r2, helix_r2, pca_kd_r2


def plot_layer_sweep(pc1_r2, helix_r2, pca_kd_r2, savepath, title, n_basis=9):
    """Three side-by-side panels of per-layer R² metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    xs = np.arange(len(pc1_r2))
    axes[0].plot(xs, pc1_r2, color="C0", lw=1.8)
    axes[0].set_title("PC1 R²  (linear magnitude 'spine')")
    axes[1].plot(xs, helix_r2, color="C1", lw=1.8)
    axes[1].set_title("helix R²  (trig basis fit)")
    ratio = helix_r2 / np.maximum(pca_kd_r2, 1e-6)
    axes[2].plot(xs, ratio, color="C2", lw=1.8)
    axes[2].set_title(f"helix / {n_basis}-d PCA  (subspace dominance)")
    for ax in axes:
        ax.set_xlabel("layer L  (hidden_states index)")
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(savepath)
    print(f"  saved {savepath}")


# ============================================================================
#  HELIX FIT
# ============================================================================
def fit_helix(H, numbers, periods=PERIODS):
    """Solve  H ~= B @ W  by least squares; return W, intercept, R^2.

    R^2 close to 1.0 means the (1 + 2K) helix features explain almost
    all variance in the residual stream's variation with `a`. The paper
    reports R^2 within a few percent of the same-dimensional PCA upper
    bound -- meaning the helix isn't just A good (1+2K)-D fit, it's THE
    (1+2K)-D structure the model uses.
    """
    B = helix_basis(numbers, periods=periods)
    reg = LinearRegression(fit_intercept=True).fit(B, H)
    H_hat = reg.predict(B)
    r2 = r2_score(H, H_hat, multioutput="variance_weighted")
    # sklearn stores coef_ as (n_outputs, n_features) = (d, 9). We want
    # W of shape (9, d) so that  H ~= B @ W  with B of shape (N, 9).
    W = reg.coef_.T
    return W, reg.intercept_, r2


# ============================================================================
#  PLOTS  (one per paper figure)
# ============================================================================
def plot_fourier_and_pc1(H, numbers, savepath, periods=PERIODS):
    """Top panel: FFT magnitude across a. Bottom panel: PC1 vs a.

    HOW THE TOP PANEL DISCOVERS THE PERIODS
    ---------------------------------------
    For each hidden dim, the 100 values h(0)[j], h(1)[j], ..., h(99)[j]
    form a sequence indexed by `a`. We FFT each such column. A big
    magnitude at frequency 1/T means "this column oscillates with
    period T". Averaging magnitudes across all hidden dims surfaces the
    periods that many dims agree on.

    Peaks at f = 1/2, 1/5, 1/10, 1/100  =>  T = 2, 5, 10, 100  (the helix).

    THE BOTTOM PANEL TESTS THE LINEAR COMPONENT
    -------------------------------------------
    PC1 is the residual-stream direction of maximum variance. Plotting
    PC1 vs `a` should give a nearly straight line if h(a) has a strong
    linear-in-a component -- the "rise" of the helix.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    # --- TOP: FFT magnitude vs frequency, averaged across hidden dims ---
    Hc = H - H.mean(0, keepdims=True)           # remove DC bin
    fft_mag = np.abs(np.fft.rfft(Hc, axis=0))   # (N//2+1, d)
    avg_mag = fft_mag.mean(axis=1)
    freqs = np.fft.rfftfreq(H.shape[0])

    ax = axes[0]
    ax.plot(freqs, avg_mag, color="C0", lw=1.5, label="Fourier Decomposition")
    ymax = avg_mag[1:].max() * 1.05             # ignore the DC bin for scale
    ax.set_ylim(0, ymax)

    # --- Reference lines for the paper's predicted periods ---
    # These show the FOUR periods the K&T paper found for Pythia. If the
    # model under study uses these, peaks will sit on the gray lines. If
    # not, the auto-detector below will surface what the model actually
    # uses.
    for T in periods:
        ax.axvline(1.0 / T, ls="--", color="gray", alpha=0.25)
        ax.text(1.0 / T, ymax * 0.02, f" T={T} (ref)",
                fontsize=7, color="gray", ha="left", va="bottom", rotation=90)

    # --- Auto-detect peaks (the unbiased read) ---
    # Find the top-5 most prominent peaks in the spectrum, ignoring the
    # DC bin. This will surface ANY periodic structure the model uses --
    # the paper's T = {2, 5, 10, 100} if present, or different periods
    # if the model encodes integers some other way. Period = round(1/f).
    mag_no_dc = avg_mag.copy()
    mag_no_dc[0] = 0.0
    # `prominence` filters out small ripples; tuned to about half a std
    # above the spectrum's noise floor.
    peak_idx, _ = find_peaks(mag_no_dc, prominence=mag_no_dc.std() * 0.4)
    # Sort by magnitude descending; take top 5.
    peak_idx = sorted(peak_idx, key=lambda i: -mag_no_dc[i])[:5]
    for i in peak_idx:
        f = freqs[i]
        if f <= 0:
            continue
        T_peak = 1.0 / f
        ax.annotate(
            f"T≈{T_peak:.1f}",
            xy=(f, avg_mag[i]),
            xytext=(f, ymax * 0.85),
            fontsize=9, ha="center",
            bbox=dict(boxstyle="round", fc="lightyellow", ec="black", lw=0.7),
            arrowprops=dict(arrowstyle="-", lw=0.6, color="black"),
        )

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 0.51)

    # --- BOTTOM: PC1 vs a with linear best-fit ---
    pca = PCA(n_components=min(10, H.shape[0], H.shape[1])).fit(H)
    pc1 = pca.transform(H)[:, 0]
    slope, _ = np.polyfit(numbers, pc1, 1)
    # PCA sign is arbitrary; flip so PC1 grows with a.
    if slope < 0:
        pc1 = -pc1
    slope, intercept = np.polyfit(numbers, pc1, 1)
    yhat = slope * numbers + intercept
    r2 = r2_score(pc1, yhat)

    ax = axes[1]
    sc = ax.scatter(numbers, pc1, c=numbers, cmap="viridis", s=40, label="PC1")
    ax.plot(numbers, yhat, "r--", lw=1.8, label=f"Best fit (R$^2$ = {r2:.3f})")
    ax.set_xlabel("Number")
    ax.set_ylabel("PC1 Value")
    ax.legend(loc="upper left")
    fig.colorbar(sc, ax=ax)

    fig.tight_layout()
    fig.savefig(savepath)
    print(f"  saved {savepath}   (linear-fit R^2 on PC1 = {r2:.3f})")


def plot_circles_and_line(H, numbers, W, intercept, savepath,
                           script="latin", periods=PERIODS):
    """For each period T, project h(a) onto (cos_T direction, sin_T direction)
    in residual space. You'll see actual circles drawn out by the
    activations. Bottom strip: linear projection -- the number line.

    WHY WE QR-DECOMPOSE
    -------------------
    W[idx_cos_T] and W[idx_sin_T] aren't orthogonal in general. If we
    just project onto them naively the result is an ELLIPSE. QR finds
    an orthonormal basis Q spanning the same 2D plane -- now the circle
    looks like a circle.
    """
    Hc = H - intercept

    K = len(periods)
    fig = plt.figure(figsize=(max(3.5 * K, 8), 5))
    gs = fig.add_gridspec(2, K, height_ratios=[3, 1], hspace=0.45)

    for j, T in enumerate(periods):
        u_cos = W[basis_idx(("cos", T), periods=periods)]
        u_sin = W[basis_idx(("sin", T), periods=periods)]
        # Orthonormalise the (u_cos, u_sin) pair -> honest circle, not ellipse.
        Q, _ = np.linalg.qr(np.stack([u_cos, u_sin], axis=1))
        coords = Hc @ Q

        ax = fig.add_subplot(gs[0, j])
        # Colour by (a mod T) for small T -- same residue class same colour.
        ax.scatter(coords[:, 0], coords[:, 1],
                   c=numbers % T if T <= 10 else numbers,
                   cmap="viridis", s=40, alpha=0.85)
        seen = set()
        for n in numbers:
            r = n % T
            if T <= 10 and r not in seen:
                ax.annotate(format_number(int(n), script),
                            (coords[n, 0], coords[n, 1]),
                            fontsize=8, ha="center")
                seen.add(r)
            elif T > 10 and n % 5 == 0:
                ax.annotate(format_number(int(n), script),
                            (coords[n, 0], coords[n, 1]),
                            fontsize=7, ha="center", alpha=0.7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        ax.text(0.02, 0.98, f"cos(a|T={T})\nsin(a|T={T})",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(fc="white", ec="black", alpha=0.7, lw=0.6))

    # Bottom strip: linear projection (the number line).
    u_lin = W[basis_idx("lin", periods=periods)]
    u_lin = u_lin / np.linalg.norm(u_lin)
    lin = Hc @ u_lin
    ax = fig.add_subplot(gs[1, :])
    ax.scatter(lin, np.zeros_like(lin), c=numbers, cmap="viridis",
               s=55, alpha=0.85)
    for n in numbers[::5]:
        ax.annotate(format_number(int(n), script), (lin[n], 0.0),
                    xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=7)
    ax.set_yticks([])
    ax.set_xlabel("a   (projection on u_lin)")

    fig.savefig(savepath, bbox_inches="tight")
    print(f"  saved {savepath}")


def plot_helix_3d(H, numbers, W, intercept, T, savepath,
                   script="latin", periods=PERIODS):
    """The iconic T=10 helix: project h(a) onto (u_cos_T, u_sin_T, u_lin)
    and plot in 3D.

    What you should see: numbers with the same units digit stack
    vertically (same (cos, sin) coords, increasing along u_lin), and
    each "decade" of 10 numbers wraps around the helix once.
    """
    Hc = H - intercept
    u_cos = W[basis_idx(("cos", T), periods=periods)]
    u_sin = W[basis_idx(("sin", T), periods=periods)]
    u_lin = W[basis_idx("lin", periods=periods)]

    # QR-orthonormalise the 3D frame -> honest helix.
    Q, _ = np.linalg.qr(np.stack([u_cos, u_sin, u_lin], axis=1))
    coords = Hc @ Q

    fig = plt.figure(figsize=(7, 9))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.viridis(numbers / max(1, numbers.max()))
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c=colors, s=55, depthshade=True)
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
            color="black", lw=0.5, alpha=0.4)
    for n in numbers:
        ax.text(coords[n, 0], coords[n, 1], coords[n, 2],
                f" {format_number(int(n), script)}", fontsize=7, color="black")
    ax.set_title(f"T = {T} helix of h(a) in residual space\n"
                 "(orthonormalised u_cos, u_sin, u_lin)")
    ax.set_xlabel("u_cos");  ax.set_ylabel("u_sin");  ax.set_zlabel("u_lin")
    fig.savefig(savepath, bbox_inches="tight")
    print(f"  saved {savepath}")


def plot_pca_2d(H, numbers, savepath, script="latin"):
    """2D PCA scatter of h(a). Basis-free diagnostic: reveals structure
    that fitting a fixed trig basis (helix R²) hides.

    For Latin: you see the helix from above -- a disc with same-units
    clusters around its rim and a magnitude gradient through it.

    For Roman: a piecewise-linear staircase trajectory with jumps at
    threshold values (4->5, 9->10, 49->50, 89->90), reflecting the
    additive system's first-letter changes.

    For Greek alphabetic: tight clusters by tens-letter (ι*, κ*, λ*, ...)
    with no consistent ordering between clusters -- the model has not
    formed a magnitude representation.

    The black trajectory line connects consecutive `a`, so the eye can
    follow what the sequence is doing without needing periodicity.
    """
    Hc = H - H.mean(axis=0, keepdims=True)
    pca = PCA(n_components=2).fit(Hc)
    coords = pca.transform(Hc)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 9))
    # Faint trajectory connecting consecutive integers -- reveals
    # staircase / cluster-hop / spiral shape directly.
    ax.plot(coords[:, 0], coords[:, 1],
            color="black", lw=0.5, alpha=0.25, zorder=0)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=numbers,
                    cmap="viridis", s=60, alpha=0.85, zorder=2)
    for n in numbers:
        ax.annotate(format_number(int(n), script),
                    (coords[n, 0], coords[n, 1]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")
    ax.set_xlabel(f"PC1  ({var[0]:.1%} of variance)")
    ax.set_ylabel(f"PC2  ({var[1]:.1%} of variance)")
    ax.set_title(f"2D PCA of residual stream — script={script}\n"
                 "(basis-free: reveals non-periodic structure)")
    fig.colorbar(sc, ax=ax, label="a")
    fig.savefig(savepath, bbox_inches="tight")
    print(f"  saved {savepath}   "
          f"(PC1: {var[0]:.1%}, PC2: {var[1]:.1%}, cumulative: {var.sum():.1%})")


# ============================================================================
#  Main flow
# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default="EleutherAI/pythia-6.9b",
                    help="HF model id. Paper uses pythia-6.9b and gpt-j-6b. "
                         "On 128 GB unified memory either runs comfortably.")
    ap.add_argument("--script", default="latin",
                    choices=["latin", "arabic", "persian", "devanagari",
                             "chinese", "greek", "roman", "babylonian"],
                    help="numeral script in which to feed the model.\n"
                         "  positional base-10 (helix predicted):\n"
                         "    latin arabic persian devanagari chinese\n"
                         "  non-positional / additive:\n"
                         "    greek roman\n"
                         "  positional base-60, additive within each column:\n"
                         "    babylonian")
    ap.add_argument("--pool", default="mean", choices=["mean", "last"],
                    help="how to aggregate the residual stream across the\n"
                         "numeral's sub-tokens. Default: mean.")
    ap.add_argument("--layer", type=int, default=None,
                    help="hidden_states index to read. Default = num_layers // 2.")
    ap.add_argument("--n_max", type=int, default=100,
                    help="study integers in [0, n_max).")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep R² metrics across all layers (in addition to\n"
                         "the standard 3-figure output). Helps locate where\n"
                         "a model's helix peaks -- useful for models the\n"
                         "paper didn't study, like Gemma. Output adds\n"
                         "fig_layer_sweep.png.")
    ap.add_argument("--periods", type=str, default="2,5,10,100",
                    help="comma-separated list of periods for the trig basis.\n"
                         "Default 2,5,10,100 matches the paper's Latin findings.\n"
                         "For Babylonian try 2,5,10,60,100 (paper + base-60)\n"
                         "or 10,30,60 (Babylonian-native + harmonic). The basis\n"
                         "has 1 + 2K columns: linear axis plus cos/sin per period.\n"
                         "Output directory gets a `_p<periods>` suffix when\n"
                         "this differs from the default.")
    args = ap.parse_args()

    # Parse and validate periods.
    try:
        periods = sorted({int(t.strip()) for t in args.periods.split(",")
                          if t.strip()})
    except ValueError:
        raise SystemExit(f"--periods must be comma-separated ints, got {args.periods!r}")
    if not periods or any(T <= 0 for T in periods):
        raise SystemExit(f"--periods must be positive ints, got {periods}")
    args.periods_list = periods
    args.periods_is_default = (periods == [2, 5, 10, 100])

    device = pick_device()
    print(f"device: {device}")
    if os.environ.get("HF_TOKEN"):
        print("HF_TOKEN loaded from .env (gated/private models accessible, "
              "downloads via hf_transfer)")

    # bf16: 16-bit "brain float". Same exponent range as fp32 but only 8
    # mantissa bits. Great for inference on big LMs -- half the memory of
    # fp32 with no measurable accuracy loss. CPU has no efficient bf16
    # kernels, so fall back to fp32 there.
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16,
                 "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    if device.type == "cpu":
        dtype = torch.float32
    print(f"loading {args.model} ({dtype}) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    # transformers >= 4.45 renamed torch_dtype -> dtype; try new, fall back.
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype)
    model = model.to(device).eval()

    n_layers = get_num_layers(model)
    layer = args.layer if args.layer is not None else n_layers // 2
    print(f"reading residual stream at hidden_states[{layer}] "
          f"(model has {n_layers} transformer layers)")
    print(f"numeral script: {args.script}")
    print(f"sub-token pool: {args.pool}  "
          f"({'reads last sub-token only' if args.pool == 'last' else 'averages over all sub-tokens of the numeral'})")

    numbers = np.arange(args.n_max)

    # Tokenisation note. For non-Latin scripts (especially Roman) many
    # numerals will split into multiple BPE tokens. With --pool mean this
    # is fine: we average activations across all sub-tokens. Just print a
    # heads-up so it isn't surprising.
    multitoken = []
    for n in numbers:
        ids = tok(f" {format_number(int(n), args.script)}",
                  add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            multitoken.append((int(n), len(ids)))
    if multitoken:
        n_multi = len(multitoken)
        max_tokens = max(t for _, t in multitoken)
        print(f"note: {n_multi}/{len(numbers)} numerals split into multiple "
              f"BPE tokens (up to {max_tokens} sub-tokens).")

    # Per-model, per-script, per-pool output directory. We tag the pool
    # subfolder with `_n<n_max>` when n_max != 100 and with `_p<periods>`
    # when periods != the paper's [2,5,10,100], so extended-range and
    # extended-basis runs don't clobber the standard figures.
    pool_dir = args.pool
    if args.n_max != 100:
        pool_dir = f"{pool_dir}_n{args.n_max}"
    if not args.periods_is_default:
        pool_dir = f"{pool_dir}_p{'-'.join(map(str, periods))}"
    out_dir = (OUT_DIR / "out" / args.model.replace("/", "__") /
               args.script / pool_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing figures to {out_dir}")
    print(f"helix basis periods: {periods}  (1 + 2*{len(periods)} = {1+2*len(periods)} features)")

    if args.sweep:
        # ----- Optional: layer sweep -----
        # Capture every layer's residual stream in one pass per integer,
        # then compute PC1 / helix / 9-D PCA R² at each layer. This shows
        # WHERE in the network each metric peaks, so we can avoid the
        # "we read the wrong layer" trap when the model isn't Pythia.
        print(f"\n--- layer sweep ---")
        H_all = collect_activations_all_layers(
            model, tok, numbers, device, script=args.script, pool=args.pool)
        print(f"all-layer activations: H_all in R^{H_all.shape}")
        pc1_r2, helix_r2, pca_kd_r2 = run_sweep_analysis(
            H_all, numbers, periods=periods)
        print(f"  PC1 R²    peak {pc1_r2.max():.3f} @ layer {pc1_r2.argmax()}")
        print(f"  helix R²  peak {helix_r2.max():.3f} @ layer {helix_r2.argmax()}")
        ratio = helix_r2 / np.maximum(pca_kd_r2, 1e-6)
        print(f"  helix/PCA peak {ratio.max():.3f} @ layer {ratio.argmax()}")
        plot_layer_sweep(
            pc1_r2, helix_r2, pca_kd_r2,
            out_dir / "fig_layer_sweep.png",
            title=(f"Layer sweep — {args.model} / {args.script} / "
                   f"pool={args.pool} / periods={periods}"),
            n_basis=1 + 2 * len(periods))
        # Re-target the standard pipeline to the helix-R² peak layer.
        peak_layer = int(helix_r2.argmax())
        if args.layer is None:
            print(f"  using peak layer {peak_layer} for the three standard figures")
            layer = peak_layer
        else:
            print(f"  (keeping user-specified layer {args.layer} for standard figs)")
        # We already have the activations at this layer in H_all; slice them.
        H = H_all[:, layer, :]
    else:
        # STEP 1: collect activations at the chosen single layer
        H = collect_activations(model, tok, numbers, layer, device,
                                script=args.script, pool=args.pool)

    print(f"activations: H in R^{H.shape}")

    # STEP 2: Fourier discovery + PC1 sanity check (paper Fig 2)
    plot_fourier_and_pc1(H, numbers, out_dir / "fig2_fourier_pc1.png",
                         periods=periods)

    # STEP 3: helix regression
    W, intercept, r2 = fit_helix(H, numbers, periods=periods)
    n_basis = 1 + 2 * len(periods)
    print(f"\nhelix fit R^2 (variance-weighted) = {r2:.4f}")
    pca_kd = PCA(n_components=min(n_basis, *H.shape)).fit(H)
    H_pca_kd = pca_kd.inverse_transform(pca_kd.transform(H))
    r2_pca_kd = r2_score(H, H_pca_kd, multioutput="variance_weighted")
    print(f"  {n_basis}-d PCA reconstruction R^2  = {r2_pca_kd:.4f}")
    print("(For Latin: helix R^2 ~ PCA R^2 within a few %. "
          "For non-positional scripts or basis/period mismatches: gap widens.)")

    # STEP 4: per-T circles + linear strip (paper Fig 3)
    plot_circles_and_line(H, numbers, W, intercept,
                          out_dir / "fig3_circles_and_line.png",
                          script=args.script, periods=periods)

    # STEP 5: T=10 helix in 3D (paper Fig 1, right). Use T=10 if available;
    # otherwise fall back to the first period in the list.
    T_helix = 10 if 10 in periods else periods[0]
    plot_helix_3d(H, numbers, W, intercept, T=T_helix,
                  savepath=out_dir / f"fig1_helix_T{T_helix}.png",
                  script=args.script, periods=periods)

    # STEP 6: 2D PCA scatter -- basis-free diagnostic for non-periodic
    # structure (Roman staircase, Greek clusters, etc.).
    plot_pca_2d(H, numbers, out_dir / "fig4_pca_2d.png",
                script=args.script)

    print("\nall figures written to", out_dir)


if __name__ == "__main__":
    main()
