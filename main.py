# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.1",
#     "transformers>=4.40",
#     "accelerate>=0.30",
#     "hf-transfer>=0.1.6",
#     "python-dotenv>=1.0",
#     "numpy",
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
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150})

# The four periods the paper discovered via Fourier analysis. T=2 captures
# parity, T=5 captures (a mod 5), T=10 the decimal units digit, T=100 the
# coarse position. Together with a linear magnitude axis they form a
# 9-parameter (1 + 2*4) encoding of any integer a -- a "generalized helix".
PERIODS = [2, 5, 10, 100]
OUT_DIR = Path(__file__).parent


# ============================================================================
#  HELPERS
# ============================================================================
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
#  HELIX FIT
# ============================================================================
def fit_helix(H, numbers):
    """Solve  H ~= B @ W  by least squares; return W, intercept, R^2.

    R^2 close to 1.0 means the 9 helix features explain almost all
    variance in the residual stream's variation with `a`. The paper
    reports R^2 within a few percent of the same-dimensional PCA upper
    bound -- meaning the helix isn't just A good 9-D fit, it's THE
    9-D structure the model uses.
    """
    B = helix_basis(numbers)
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
def plot_fourier_and_pc1(H, numbers, savepath):
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
    for T in PERIODS:
        f = 1.0 / T
        bin_idx = int(round(f * H.shape[0]))
        ax.axvline(f, ls="--", color="gray", alpha=0.25)
        ax.annotate(
            f"T={T}",
            xy=(f, avg_mag[bin_idx]),
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


def plot_circles_and_line(H, numbers, W, intercept, savepath, script="latin"):
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

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1], hspace=0.45)

    for j, T in enumerate(PERIODS):
        u_cos = W[basis_idx(("cos", T))]
        u_sin = W[basis_idx(("sin", T))]
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
    u_lin = W[basis_idx("lin")]
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


def plot_helix_3d(H, numbers, W, intercept, T, savepath, script="latin"):
    """The iconic T=10 helix: project h(a) onto (u_cos_T, u_sin_T, u_lin)
    and plot in 3D.

    What you should see: numbers with the same units digit stack
    vertically (same (cos, sin) coords, increasing along u_lin), and
    each "decade" of 10 numbers wraps around the helix once.
    """
    Hc = H - intercept
    u_cos = W[basis_idx(("cos", T))]
    u_sin = W[basis_idx(("sin", T))]
    u_lin = W[basis_idx("lin")]

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
                             "chinese", "greek", "roman"],
                    help="numeral script in which to feed the model.\n"
                         "  positional base-10 (helix predicted):\n"
                         "    latin arabic persian devanagari chinese\n"
                         "  non-positional / additive:\n"
                         "    greek roman")
    ap.add_argument("--pool", default="mean", choices=["mean", "last"],
                    help="how to aggregate the residual stream across the\n"
                         "numeral's sub-tokens. Default: mean.")
    ap.add_argument("--layer", type=int, default=None,
                    help="hidden_states index to read. Default = num_layers // 2.")
    ap.add_argument("--n_max", type=int, default=100,
                    help="study integers in [0, n_max).")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    args = ap.parse_args()

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

    n_layers = model.config.num_hidden_layers
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

    # STEP 1: collect activations
    H = collect_activations(model, tok, numbers, layer, device,
                            script=args.script, pool=args.pool)
    print(f"activations: H in R^{H.shape}")

    # Per-model, per-script, per-pool output directory.
    out_dir = (OUT_DIR / "out" / args.model.replace("/", "__") /
               args.script / args.pool)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing figures to {out_dir}")

    # STEP 2: Fourier discovery + PC1 sanity check (paper Fig 2)
    plot_fourier_and_pc1(H, numbers, out_dir / "fig2_fourier_pc1.png")

    # STEP 3: helix regression
    W, intercept, r2 = fit_helix(H, numbers)
    print(f"\nhelix fit R^2 (variance-weighted) = {r2:.4f}")
    pca9 = PCA(n_components=9).fit(H)
    H_pca9 = pca9.inverse_transform(pca9.transform(H))
    r2_pca9 = r2_score(H, H_pca9, multioutput="variance_weighted")
    print(f"  9-d PCA reconstruction R^2  = {r2_pca9:.4f}")
    print("(For Latin: helix R^2 ~ PCA R^2 within a few %. "
          "For non-positional scripts: gap is usually larger.)")

    # STEP 4: per-T circles + linear strip (paper Fig 3)
    plot_circles_and_line(H, numbers, W, intercept,
                          out_dir / "fig3_circles_and_line.png",
                          script=args.script)

    # STEP 5: T=10 helix in 3D (paper Fig 1, right)
    plot_helix_3d(H, numbers, W, intercept, T=10,
                  savepath=out_dir / "fig1_helix_T10.png",
                  script=args.script)

    print("\nall figures written to", out_dir)


if __name__ == "__main__":
    main()
