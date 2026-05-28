# Walkthrough

A guided tour of every moving part in this repo and the reasoning behind each
decision. Reads top-to-bottom; cross-references files at `path.py:line` so you
can jump in.

This is *what we did and why* — not a results summary. For numbers, read the
blogpost or `paper.pdf`.

---

## 0. The research question

Kantamneni & Tegmark (2025, arXiv:2502.00873; "K&T" from here on) showed that
three mid-sized LLMs encode each integer $a \in [0, 99]$ as a "generalised
helix" in their residual stream: a linear "number-line" direction plus
modular circles at periods $T \in \{2, 5, 10, 100\}$. They also showed
(causally, with patching) that the helix is used to perform addition.

The question this repo asks is **where the geometry comes from**. A helix
that fits well at the model's peak layer can mean any of:

1. **Depth-built**: the transformer constructs the geometry layer by layer.
   This is K&T's reading.
2. **Inherited from the input pipeline**: rendering, tokenization, embedding
   lookup, and mean-pooling between them produce a vector that the
   trig basis fits as a helix, before any transformer block runs.
3. **A measurement artifact**: the trig basis you chose can't see the
   period the model actually uses, or your input range is too short to
   identify a real period, and the score looks low through no fault of
   the model.

To distinguish these, we report four measurements per
$(\text{model}, \text{script})$ cell:

- **$\rho$** = helix $R^2$ at $L=0$ divided by helix $R^2$ at the peak
  layer. Catches case 2.
- **Quality ratio** = helix $R^2$ / best-possible $K$-dimensional PCA
  $R^2$. Tells us whether the helix is actually the dominant low-d
  structure or just one fit among many.
- **Random-embedding control** = replace the model's learned embedding
  matrix with a Gaussian random matrix at matched scale, redo the L=0 fit,
  compare. Tells us how much of the L=0 geometry is mechanical vs learned.
- **CKA** between the helix subspaces at $L=0$ and the peak layer.
  Distinguishes "depth left the geometry alone" from "depth produced a
  similar-quality helix with a different integer-to-integer pattern".

We run those four checks across **8 models × 12 numeral scripts (96 cells)**,
plus extended-basis variants for binary, hex, and Babylonian (which fail
case 3 at paper defaults).

That's the whole framing. Everything else in this repo is plumbing for it.

---

## 1. Repository layout

```
addition_helix/
├── helix_lib.py        shared utilities (numerals, helix basis, OLS fit, config)
├── main.py             per-cell sweep: collect activations, fit helix, dump JSON
├── aggregate.py        walk out/ and build out/_index.json
├── embed_control.py    random-embedding control across all cells
├── subspace_align.py   CKA across all cells
├── compare.py          stitch per-model layer-sweep PNGs into one image
├── run.sh              driver: every (model, script, basis) combo, idempotent
├── build_paper.sh      two xelatex passes for paper.pdf
├── paper.tex           4-6 page ICML-ballpark write-up
├── README.md           the 50-line orientation
├── walkthrough.md      this file
├── www/                React + Vite blogpost site
├── out/                generated data + figures (gitignored)
└── .env                HF_TOKEN for gated models (not committed)
```

The Python side has **one purpose per script**, all chained by `run.sh`. The
React side reads `out/_index.json` and lazy-loads per-figure JSON to render
charts. The paper reads the same `out/` for static PNGs.

### Why PEP 723 inline metadata + `uv run`?

Each `.py` file at the top of the repo declares its own dependencies in a
`# /// script` header:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.1",
#     "transformers>=4.40",
#     ...
# ]
# ///
```

`uv run main.py` reads this header, builds an isolated venv with just those
deps, and executes the script. Benefits over a single `pyproject.toml`:

- **No shared venv state to corrupt.** Each script's deps are declared
  next to its code, so cross-script breakage isn't possible.
- **`embed_control.py` doesn't need torch.** It only loads tokenizers, builds
  random embedding matrices, mean-pools, and fits OLS — `numpy + sklearn +
  transformers` is enough. Its venv is ~200 MB instead of ~7 GB.
- **The repo is self-bootstrapping**: `uv run main.py` is the first command
  a new user can type and have it work.

The downside is that PEP 723 deps must be a *superset* of what every helper
module they import needs. `helix_lib.py` uses only `numpy + sklearn`, so any
caller's PEP 723 header is sufficient.

### Why both a paper and a blogpost?

They cover the same content; the blogpost is interactive (3D helix you can
rotate, layer sweeps with hover, the ρ heatmap and CKA scatter as live d3
charts) and the paper is the static archival form. Both read out of the same
`out/*.json` files.

---

## 2. `helix_lib.py` — the shared math

This is the smallest file but it's the conceptual core. It defines:

- The **constants** (`PERIODS`, `ALL_MODELS`, `ALL_SCRIPTS`).
- The **numeral renderers** (12 scripts; `format_number` dispatches).
- The **helix basis** $B(a)$ and the OLS **helix fit**.
- Two **HF config introspection helpers** (`get_num_layers`,
  `get_d_model`) that walk the model's config to find depth and width
  even when those live inside a sub-config.

It is *torch-free* on purpose so `embed_control.py` can import it without
pulling in torch.

### 2.1 Constants

```python
PERIODS = [2, 5, 10, 100]
```

These four periods came out of K&T's Fourier analysis of Pythia/Latin. T=2
captures parity, T=5 captures (a mod 5), T=10 captures the units digit,
T=100 captures coarse position. Together with a linear magnitude axis they
form a 9-feature basis. We use these as the **paper-default basis** on
every cell; for scripts with non-decimal natural periods we run additional
extended-basis variants (see §6).

`ALL_MODELS` lists the 8 HuggingFace IDs in display order. The order matters
because the React site renders the ρ heatmap and other tables in this
sequence; see `www/src/lib/data.ts:MODEL_ORDER`.

`ALL_SCRIPTS` lists the 12 scripts grouped by family. The React site mirrors
this in `SCRIPT_ORDER`.

### 2.2 The helix basis $B(a)$

```python
def helix_basis(a, periods=PERIODS):
    a = np.asarray(a, dtype=np.float64)
    cols = [a]
    for T in periods:
        cols.append(np.cos(2 * np.pi * a / T))
        cols.append(np.sin(2 * np.pi * a / T))
    return np.stack(cols, axis=-1)
```

For an array of integers $a$ of length $n$, this returns an
$(n, 1 + 2|\text{periods}|)$ matrix. Each row is

$$B(a) = [a, \cos(2\pi a/T_1), \sin(2\pi a/T_1), \cos(2\pi a/T_2), \sin(2\pi a/T_2), \ldots]$$

The `(cos, sin)` pair for a given period $T$ traces out the unit circle as
$a$ varies. Integers sharing the same $(a \bmod T)$ land at the same point
on the $T$-circle. Stacked with the linear column, the result is a helix:
the cos/sin pair rotates around as $a$ increases, and the linear column
lifts each point.

The trig features are deliberately *separated* `cos` and `sin` rather than
the complex exponential. Two reasons:

1. The fit is real-valued OLS. We don't need complex arithmetic.
2. Each `(cos, sin)` pair gives the model a 2D plane to rotate in — which
   is exactly the "Clock" algorithm K&T identify: addition becomes rotation
   of each circle.

### 2.3 The helix fit

```python
def fit_helix(H, numbers, periods=PERIODS):
    B = helix_basis(numbers, periods=periods)
    reg = LinearRegression(fit_intercept=True).fit(B, H)
    H_hat = reg.predict(B)
    r2 = float(r2_score(H, H_hat, multioutput="variance_weighted"))
    return reg.coef_.T, reg.intercept_, r2
```

For activations $H \in \mathbb{R}^{n \times d}$ (one row per integer, $d$ is
the model's hidden dimension, e.g. 4096), this solves $H \approx B W$ by
ordinary least squares, with intercept. sklearn returns `coef_` as
$(d, K)$ where $K = 1 + 2|\text{periods}|$; we transpose so callers get
$W$ of shape $(K, d)$ — convenient when they do `B @ W` later.

The R² is **variance-weighted across hidden dimensions**. Why not the
simpler unweighted mean? Some hidden dimensions are nearly constant in $a$;
their per-dim R² would be near 0 even if the helix model is correct for the
ones that vary. Variance-weighting puts the weight on the dimensions where
there *is* something to predict.

### 2.4 The numeral renderers

We have 12 scripts. Three families of implementation:

**Positional base-10 with a contiguous Unicode digit block** (Arabic-Indic,
Persian, Devanagari, Thai): a single dict lookup per script's digit-block
offset, then string-substitute each decimal digit. The implementation
lives in `format_number` via the `_DIGIT_BASE` table:

```python
_DIGIT_BASE = {
    "arabic":     0x0660,
    "persian":    0x06F0,
    "devanagari": 0x0966,
    "thai":       0x0E50,
}
```

Each one is "render the decimal digits with the chosen script's
codepoint base." Cheap, exact, no special cases.

**Positional base-10 in CJK glyphs** (`to_chinese_positional`):

```python
CHINESE_POSITIONAL = "〇一二三四五六七八九"
```

We use the *positional* form (23 → 二三) rather than the additive form
(23 → 二十三). The additive form is closer to Roman in spirit (it has
a 十 = "ten" marker word) and would muddy the comparison. The positional
form gives us a clean base-10-in-different-glyphs control.

**Non-positional renderers** (`to_roman`, `to_greek`, `to_hebrew`):
hand-coded lookup tables. Greek and Hebrew alphabetic numerals are
*additive* (23 = κ + γ; tens-letter + units-letter from disjoint blocks),
which is exactly what we want to compare against positional. Roman is also
additive but with the classic subtractive shortcuts (IV, IX, XL...) which
we encode in the standard largest-first greedy loop:

```python
def to_roman(n: int) -> str:
    if n == 0: return "nulla"
    pairs = [(1000, "M"), (900, "CM"), (500, "D"), ...]
    out = []
    for v, s in pairs:
        while n >= v:
            out.append(s); n -= v
    return "".join(out)
```

Hebrew has two special cases at 15 and 16 (טו, טז instead of יה, יו) to
avoid spelling part of the Tetragrammaton — a real convention from the
text, which we honour.

**Babylonian cuneiform** (`to_babylonian`): positional base-60, additive
within each sexagesimal column. The wedges are:

```python
BAB_ONE  = "𒁹"  # = 1
BAB_TEN  = "𒌋"  # = 10
BAB_ZERO = "𒑊"  # late-period zero placeholder
```

For 23: two ten-wedges + three one-wedges = 𒌋𒌋𒁹𒁹𒁹. For 99: one
sexagesimal column for 60 + one column for 39 = `𒁹 𒌋𒌋𒌋𒁹𒁹𒁹𒁹𒁹𒁹𒁹𒁹𒁹`.
The columns are separated by a single space — standard cuneiform
transliteration convention. `_bab_column(v)` renders a single column
additively; the main loop decomposes $n$ into base-60 columns
most-significant-first.

This script is the key one. Babylonian's *additive within each column*
structure means that for $a$ between 0 and 59, the mean over wedge tokens
is a convex combination of $e_{\text{ten}}$ and $e_{\text{one}}$ whose
weights vary with $a$. This produces a smooth manifold in residual space
for *any* embedding choice — the structure is in the rendering+pooling,
not in the embeddings. See §6.3 for the random-embedding control that
makes this concrete.

**Binary / hex** are trivially `bin(n)[2:]` and `format(n, "x")`. We
included them as additional controls: binary has natural periods
$T \in \{2, 4, 8, 16, 32, 64\}$ and hex has $T \in \{16, 32, 64, 256\}$,
neither of which the paper-default basis $\mathcal{T} = \{2, 5, 10, 100\}$
catches except for the trivial $T=2$ on binary.

### 2.5 `format_number(n, script)` dispatch

A single function that maps `(int, str_name)` → the rendered string. Used
*everywhere* in the project: by `main.py` to build the input text, by
`embed_control.py` to do the same for the random-embed control, by
`subspace_align.py` for the CKA pass, by every annotation that labels a
point on a plot, and by the React site's `SCRIPT_EXAMPLE` table.

Pulling this into one shared dispatch was the entire point of `helix_lib.py`
existing: before the DRY pass it lived (with cosmetic drift) in three
separate files.

### 2.6 HF config helpers

`get_num_layers(model)` and `get_d_model(cfg)` walk `model.config` for
`num_hidden_layers` / `n_layer` / `num_layers` and `hidden_size`. Different
model families expose these under different attribute names, and Gemma 4's
config nests its language-model block in `config.text_config`. Both helpers
try the common paths in order and fall back to the nested-config path. If
neither is found, they raise — the user gets a clear error rather than a
mysterious downstream KeyError.

---

## 3. `main.py` — the per-cell sweep

The big workhorse (1050 lines). For one `(model, script, pool, n_max,
periods)` cell, it:

1. Loads the model and tokenizer.
2. For each integer $a \in \{0, ..., n_{\max}-1\}$, runs one forward pass,
   captures `hidden_states` at every layer, and mean-pools (or
   last-token-pools) over the numeral's sub-tokens.
3. Fits the helix basis at every layer.
4. Computes the K-d PCA upper bound at every layer.
5. Identifies the peak layer.
6. Generates four standard figures (FFT+PC1, circles, 3D helix, PCA 2D) at
   the peak layer.
7. Writes everything to `out/<model>/<script>/<pool>/` as PNG + JSON.

### 3.1 The preamble

```python
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
import json
import matplotlib.pyplot as plt
...
```

The `load_dotenv` call happens *before* `transformers` is imported. This is
load-bearing: `transformers` reads `HF_TOKEN` and `HF_HUB_ENABLE_HF_TRANSFER`
from `os.environ` at *import* time, not at first use. If you `.env`-set
them after importing transformers, it's too late.

`HF_HUB_ENABLE_HF_TRANSFER=1` switches the HuggingFace downloader to the
Rust-based `hf_transfer` library, which is 5-10× faster on big shards
(important for Gemma-31B / Qwen-32B / OLMo-32B).

### 3.2 tqdm to a real TTY

```python
try:
    _TQDM_OUT = open("/dev/tty", "w")
except OSError:
    _TQDM_OUT = sys.stderr
```

`run.sh` wraps each `main.py` call in `script(1)` (a Linux/macOS pseudo-TTY
recorder) so tqdm bars render live with `\r` updates. Writing tqdm to
`/dev/tty` directly is the most robust path; if there's no controlling
terminal (CI, some Docker setups), we fall back to stderr.

### 3.3 JSON export helpers

```python
def _round(v): ...
def _jsonable(o): ...
def dump_json(path, obj): ...
```

Every plot also writes a sibling `.json` so the React site can re-render the
same data interactively. `_round` recursively rounds floats to 6 decimal
places to keep file sizes small; `_jsonable` converts numpy arrays / scalars
to plain Python; `dump_json` writes single-line JSON (no whitespace) for
compact wire format.

The `RUN_META` dict at module level carries provenance (model, script,
pool, n_max, periods, layer) into every dumped JSON. It's mutated as
`main.py` progresses (e.g. `RUN_META["layer"] = peak_layer` before the
figure-generation calls). **Important**: `_jsonable` walks `RUN_META` and
snapshots it at serialise time, so changes between dumps are correctly
recorded — but mutating `RUN_META` is a footgun if you reorder calls.

### 3.4 Font configuration

`_configure_fonts()` walks a fallback stack and sets matplotlib's
`font.family` to whichever fonts are actually installed. The order matters
because matplotlib 3.6+ walks the stack *per character*, so DejaVu Sans
covers Latin/Greek/Cyrillic, Geeza Pro covers Arabic/Persian, Devanagari
Sangam MN covers Hindi digits, Hiragino covers CJK, Noto Sans Cuneiform
covers Babylonian. If a font is missing, that script will render as empty
boxes in the PNGs — the prompt at startup tells the user which fonts to
install.

### 3.5 The activation collection

```python
@torch.no_grad()
def collect_activations_all_layers(model, tokenizer, numbers, device,
                                     script="latin", pool="mean"):
    bos = tokenizer.bos_token_id
    all_acts = []
    for n in tqdm(numbers, ...):
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
        if pool == "last":
            per_layer = torch.stack(
                [h[0, -1, :].float().cpu() for h in out.hidden_states])
        else:  # mean
            per_layer = torch.stack(
                [h[0, start:end, :].mean(dim=0).float().cpu()
                 for h in out.hidden_states])
        all_acts.append(per_layer)
    return torch.stack(all_acts, dim=0).numpy()
```

Several design decisions baked in here:

- **`text = f" {format_number(...)}"`** — the input is *space-prefixed*.
  This matches K&T's convention. GPT-style tokenizers treat `"23"` and
  `" 23"` differently; the leading space is what numbers look like in
  natural prompts ("answer: 23", "compute 17 + 6"). Whether the model
  encodes a number depends on the *natural* token sequence, not the
  bare-string sequence.
- **`add_special_tokens=False`** — we manually prepend BOS so we know
  exactly how many tokens the numeral occupies. If we let the tokenizer
  add specials, some tokenizers add `<s>` and some add `<bos>` and the
  offsets become annoying to track. Doing it ourselves keeps `start, end`
  simple.
- **BOS handling**: when present, we prepend it. The numeral starts at
  position 1 and ends at position `1 + n_numeral_tokens` (exclusive). When
  there's no BOS, the numeral starts at position 0.
- **`output_hidden_states=True, use_cache=False`** — we ask HuggingFace for
  every layer's hidden state in a single forward pass. `use_cache=False`
  saves memory; we don't need the KV cache because we run one forward per
  number, not autoregressive generation.
- **The hidden_states tuple has length `n_layers + 1`**:
  - `hidden_states[0]` = output of token embedding + positional embedding,
    *before any transformer block runs*. **This is our $L=0$.**
  - `hidden_states[L]` for $L \in \{1, ..., n_\text{layers}\}$ = output of
    block $L-1$, equivalently input to block $L$.
- **Pooling**: for `pool == "last"` we read the activation at the very last
  sub-token; for `pool == "mean"` we average across all sub-tokens of the
  numeral. We use mean throughout — see §3.6.

**Why one forward pass per integer rather than batching?** The numerals
have variable sub-token counts (Latin "23" is 1 token on Pythia but 2 on
Llama; Arabic "٢٣" is 2-4 depending on the tokenizer). Batching would
require padding and offset-tracking, complicating the pooling step. A
single-integer loop is slower wall-clock but cleaner and bug-free. With
~100-1024 integers per cell and modest forward-pass cost, the entire
sweep takes 6-8 hours on a single M5 Max.

### 3.6 Why mean pooling

For Pythia/GPT-J/Llama on Latin with $n_\max = 100$, every integer is a
single token (K&T's "single-token regime"). Mean and last produce the
same number. For everything else, the choice matters.

The failure case for `pool="last"`: for `" 23"` tokenized into per-digit
sub-tokens, the last token is `"3"`. Numbers $\{3, 13, 23, 33, ...\}$
all end in `"3"`, so they all read the same activation. This bakes a
period-10 cycle into the data *by construction*, producing FFT peaks at
$T=10$ that have nothing to do with anything the model learned.

Mean pooling avoids this artifact: every sub-token contributes equally.
But it has its own consequence — for additive renderings like Babylonian
and binary, mean pooling turns symbol counts into linear arithmetic on
token embeddings. That arithmetic *is* the helix score on those scripts,
even with random embeddings (see §6.3). Mean pooling is the honest choice,
but it's not neutral — it's part of the provenance story.

K&T sidestep this by staying in the single-token regime. We can't, because
Qwen and Gemma split Latin numbers per-digit even at $n_\max = 100$, every
non-Latin script is multi-token everywhere, and the extended-basis runs
($n=600$ for Babylonian, $n=1024$ for binary/hex) go well past any model's
single-token range.

### 3.7 The layer sweep

```python
def run_sweep_analysis(H_all, numbers, periods=PERIODS):
    pc1_r2 = np.zeros(n_hidden)
    helix_r2 = np.zeros(n_hidden)
    pca_kd_r2 = np.zeros(n_hidden)
    B = helix_basis(numbers, periods=periods)
    for L in range(n_hidden):
        H_L = H_all[:, L, :]
        # PC1 R² (the "spine")
        pca = PCA(n_components=...).fit(H_L)
        pc1 = pca.transform(H_L)[:, 0]
        slope, _ = np.polyfit(numbers, pc1, 1)
        if slope < 0:
            pc1 = -pc1
        slope, intercept = np.polyfit(numbers, pc1, 1)
        pc1_r2[L] = r2_score(pc1, slope * numbers + intercept)
        # helix R²
        reg = LinearRegression().fit(B, H_L)
        helix_r2[L] = r2_score(H_L, reg.predict(B),
                                multioutput="variance_weighted")
        # K-d PCA R² (Eckart-Young upper bound)
        pca_kd = PCA(n_components=...).fit(H_L)
        pca_kd_r2[L] = r2_score(
            H_L, pca_kd.inverse_transform(pca_kd.transform(H_L)),
            multioutput="variance_weighted")
    return pc1_r2, helix_r2, pca_kd_r2
```

For each layer $L$, three things:

- **PC1 R²**: how linear-in-$a$ the dominant principal component is. This
  catches the "spine" of the helix — the magnitude direction. PC1 has an
  ambiguous sign (PCA is sign-flip-invariant); we flip if the slope is
  negative so we always report the absolute linear fit quality.
- **Helix R²**: the trig basis fit at this layer.
- **K-d PCA R²**: the Eckart-Young upper bound on any $K$-dimensional
  basis. This is what `helix R² / PCA R²` divides by to get our **quality
  ratio**. The ratio is what tells us whether the helix is *the* $K$-d
  structure or just one good $K$-d fit among many.

We sweep every layer because where the helix lives varies enormously across
models. Pythia peaks at the literal final layer; Qwen2.5-7B/Latin peaks at
$L=0$ (the embedding output itself); Gemma-4-31B has a sharp spike halfway
up. K&T originally read at the middle layer; we couldn't, because the
middle layer is the right layer for Pythia and wrong for everything else.

After the sweep, $\rho = \text{helix R}^2(L=0) \,/\, \max_L \text{helix R}^2(L)$.
This is the single number that anchors the entire provenance question.

### 3.8 The four standard figures

After the layer sweep identifies the peak layer, we generate four PNGs
(and matching JSON):

- **`fig1_helix_T10.png`** — the iconic 3D helix at $T=10$. Take the cos,
  sin, and linear directions from the fitted $W$, QR-orthonormalise them
  into a clean orthonormal frame, project the activations onto that
  frame, and 3D-scatter. Why QR? The raw cos/sin/linear directions
  *aren't* orthogonal in residual space — they're just the columns of
  $W$ that landed on those basis features. Plotting them without
  orthonormalising would render a true circle as a tilted ellipse (a
  "bent ruler" problem). QR gives us the cleanest 2D plane containing
  the cos/sin pair and the cleanest line containing the magnitude
  direction, which is what we want visually.
- **`fig2_fourier_pc1.png`** — top panel: FFT of the activations across
  $a$, averaged over hidden dimensions. Peaks at $1/2$, $1/5$, $1/10$,
  $1/100$ are the evidence the periods are real (rather than something
  we put in by assumption). The auto-peak-detector labels the top 5
  peaks regardless of where they land — so if the model uses different
  periods, we see them. Bottom panel: PC1 vs $a$, with the linear fit
  R² annotated.
- **`fig3_circles_and_line.png`** — four 2D panels, one per period,
  showing the modular circles. Bottom strip: the linear "number line"
  projection. This is K&T's Figure 3.
- **`fig4_pca_2d.png`** — basis-free 2D PCA scatter. Reveals structure
  that fitting a fixed trig basis would hide. For Roman digits this
  shows the piecewise-linear staircase trajectory with jumps at
  threshold values (4→5, 9→10, 49→50). For Greek alphabetic it shows
  tight clusters by tens-letter with no consistent ordering — the
  model has not formed a magnitude representation.

Every PNG also emits a sibling JSON with the same data so the React site
can re-render interactively.

### 3.9 The argparse interface

```bash
uv run main.py \
    --model meta-llama/Llama-3.1-8B \
    --script latin \
    --pool mean \
    --n_max 100 \
    --periods 2,5,10,100 \
    --sweep
```

- `--model` is the HuggingFace ID.
- `--script` picks one of the 12 numeral systems.
- `--pool` is `mean` (default) or `last` — keep `mean` unless you want
  to reproduce the multi-token artifact for pedagogical comparison.
- `--n_max` defaults to 100; we override to 600 (Babylonian) or 1024
  (binary, hex) when we want enough wraparounds for native periods.
- `--periods` is the trig basis as a comma-separated list. Default
  `2,5,10,100`. Override for native-basis runs.
- `--sweep` enables the layer-sweep + auto-targeting of the four standard
  figures at the helix-R² peak. Without `--sweep`, we read at
  `num_layers // 2` (the middle layer), which is the K&T default. We use
  `--sweep` for everything in `run.sh`.

### 3.10 Worked example: reading a layer sweep

For one cell, the layer sweep gives us three curves indexed by layer
$L$ (from 0 to $n_\text{layers}$). Each tells you something different.

**PC1 R² ("the spine")**. PCA finds the direction of maximum variance
across the 100 integer-vectors at layer $L$. Project all 100 vectors
onto that single direction → 100 scalar values. Fit a line
$pc_1(a) = m \cdot a + b$ and report R².

- High PC1 R² (~1) = the most-variance direction *is* a number line.
  The model's most prominent way of distinguishing integers is by
  magnitude.
- Low PC1 R² = the dominant variance direction is something else (some
  grammatical feature? a tokenizer artifact?).

For Pythia/Latin at $L=0$, PC1 R² is moderate: embeddings for 0-99
have *some* linear structure but not a clean ramp. By the peak layer,
PC1 R² is ~0.95: depth has organised the magnitude axis into a clean
number line.

**Helix R²**. The 9-feature trig basis $B$ fitted to the 4096-d
residual stream at this layer. Variance-weighted R² across hidden
dimensions.

The helix R² at the peak layer is K&T's headline number. We compute it
at every layer because we don't know in advance where the peak lives.

**K-d PCA R² (the quality ceiling)**. Helix R² alone is just a score;
it doesn't tell you whether the helix is *the* dominant 9-d structure
or one of many 9-d structures that could fit. So we ask: what is the
best possible 9-d fit to $H$? That's the Eckart-Young upper bound:
take the top-9 principal components of $H$ and reconstruct. The R² of
that reconstruction is a *ceiling*: no 9-feature basis can fit better.

The **quality ratio** = helix R² / K-d PCA R² then tells you how
dominant the helix subspace is. Ratio of 0.85 means the trig basis
captures 85% of what any 9-d basis could capture — so it isn't just *a*
good 9-d fit, it basically *is* the 9-d structure.

**Where the helix lives in the stack — why we sweep.**

K&T originally read at the middle layer (`n_layers // 2`). For Pythia
(32 layers), that's L=16. For Llama (32 layers), L=16. Both happen to
have peaks around there. So "read the middle" was a fine heuristic
for K&T's three models.

It is the *wrong* heuristic for any of the five models we added:

- Pythia peaks at the literal final layer (L=32 of 32).
- GPT-J does the same (L=28 of 28).
- Llama peaks mid-stack (L=15 of 32).
- Qwen2.5-7B/Latin peaks at L=0 — the helix is complete *before any
  block has run*. Reading the middle would miss the peak entirely.
- Gemma-4-31B sits at moderate helix R² across most of the stack with
  a sharp spike to peak at L=29 of 60.
- Gemma-4-E4B has a double-peaked structure with the global max at
  L=17 of 42.
- OLMo-3-32B rises gradually to a peak at L=23 of 64.

Without the sweep we couldn't even compute ρ — there'd be no
`max_L R²` to ratio L=0 against. The sweep also lets us auto-target
the four standard figures (FFT+PC1, circles, 3D helix, PCA 2D) at the
peak layer, so we always see the cleanest version of the geometry for
each model rather than the version at whatever layer K&T happened to
read.

**How to read a sweep plot.** A sweep PNG has three side-by-side
panels with layer L on the x-axis, R² on the y-axis. From the shape
you can tell:

- *Pythia/Latin*: helix R² rises monotonically from low at L=0 to high
  at the final layer. PC1 R² and quality ratio rise similarly. This is
  the textbook "depth builds the helix" pattern.
- *Qwen2.5-7B/Latin*: helix R² starts high at L=0, peaks immediately,
  decays gently. The PC1 R² is already saturated by L=0. The helix
  exists in the embedding+pooling output and depth doesn't add to it.
- *Gemma-4-31B/Latin*: helix R² sits around 0.5 for most layers with a
  pronounced spike to 0.7 at L=29 then falls back. PC1 R² is similar.
  Depth is *doing something* at L=29 specifically.

The sweep also reveals when something interesting is happening that
ρ alone would hide. A cell with ρ = 0.6 could have a smooth rise from
L=0 to peak (depth refines) *or* a flat-then-spike pattern (depth
suddenly builds something at one layer). The shape of the curve
matters even though ρ collapses it to a single number.

### 3.11 Output layout

Every run writes:

```
out/<model_dir>/<script>/<pool_dir>/
    ├── _meta.json
    ├── fig1_helix_T10.png       (.json)
    ├── fig2_fourier_pc1.png     (.json)
    ├── fig3_circles_and_line.png (.json)
    ├── fig4_pca_2d.png          (.json)
    ├── fig_layer_sweep.png      (.json)
```

Where:

- `<model_dir>` is the HF ID with `/` → `__` (e.g. `EleutherAI__pythia-6.9b`).
- `<pool_dir>` encodes the non-default params: `mean` for paper defaults,
  `mean_n600` for $n=600$ paper-basis, `mean_n600_p2-5-10-60-100` for
  $n=600$ with extended basis. The directory name is what `run.sh`'s
  `marker_for()` function uses to decide whether to skip an already-done
  cell. Same naming convention is parsed back by `aggregate.py` to
  reconstruct $n_{\max}$ and periods.

---

## 4. `embed_control.py` — the random-embedding control

The single sharpest measurement in the project. Replaces the model's
embedding table with a Gaussian random matrix at matched scale, recomputes
the L=0 representation, and compares its helix R² to the learned one.

If the gap $\Delta = R^2_\text{learned} - R^2_\text{random}$ is large, the
embedding table carries learned digit semantics. If $\Delta \approx 0$, the
L=0 helix is mechanical: the renderer + tokenizer + mean-pooling generated
it, and the embedding values are irrelevant.

### 4.1 Why no torch?

```python
# /// script
# dependencies = [
#     "transformers>=4.40",
#     "python-dotenv>=1.0",
#     "numpy",
#     "scikit-learn",
# ]
# ///
```

We never load model weights. We only need:

- The tokenizer (to split each numeral into sub-token IDs).
- The model's `d_model` and vocab size (read from `AutoConfig` — no weights).

The rest is `numpy` to draw the random embedding matrix, look up rows by
token ID, mean-pool, fit the helix. Skipping torch makes the venv tiny and
the wall-clock time per cell trivial (a few seconds).

### 4.2 The control's mechanics

For one `(model, script, n_max, periods)` cell:

1. Load the tokenizer (no weights).
2. Look up $d = \text{hidden\_size}$ from the model config.
3. Draw $\tilde{E} \in \mathbb{R}^{V \times d}$, where each entry is
   independently sampled from $\mathcal{N}(0, 1/\sqrt{d})$. **Why $1/\sqrt{d}$?**
   It matches the per-coordinate scale of a Xavier-initialised embedding
   matrix, which is what most LLMs start training from. The mean-norm of
   each row is $O(1)$.
4. For each integer $a$:
   - Tokenize `" " + format_number(a, script)` to get sub-token IDs.
   - Mean over the rows of $\tilde{E}$ indexed by those IDs. Call this
     $\tilde{h}(a)$. This is the *random-embedding L=0 representation*.
5. Build the matrix $\tilde{H}$ with rows $\tilde{h}(a)$.
6. Fit the helix on $\tilde{H}$ → $R^2_\text{random}$.
7. Look up $R^2_\text{learned}$ from `out/<model>/<script>/<pool>/fig_layer_sweep.json`
   (specifically the `helix_r2[0]` entry — the L=0 score that `main.py`
   already computed).
8. Report $\Delta = R^2_\text{learned} - R^2_\text{random}$.

The key insight is what *changes* between the learned and random cases:

| | learned | random |
|---|---|---|
| renderer | same | same |
| tokenizer | same | same |
| embedding values | learned during training | $\mathcal{N}(0, 1/\sqrt{d})$ |
| pooling | mean, same | mean, same |
| forward pass | not run (L=0) | not run (L=0) |

The *only* difference is the embedding lookup. So $\Delta > 0$ means
"learned embeddings carry structure beyond what rendering+pooling
mechanically produce"; $\Delta \approx 0$ means "the structure is in the
mechanical pipeline and the embeddings are irrelevant".

### 4.3 The combo list

`default_combos()` enumerates the protocol variations:

```python
# every script at the paper default (n=100, [2,5,10,100])
for s in ALL_SCRIPTS:
    combos.append((s, 100, [2, 5, 10, 100]))
# Babylonian wider window, both bases
combos.append(("babylonian",  600, [2, 5, 10, 100]))
combos.append(("babylonian",  600, [2, 5, 10, 60, 100]))
# Binary wider window, both bases
combos.append(("binary",      1024, [2, 5, 10, 100]))
combos.append(("binary",      1024, [2, 4, 8, 16, 32, 64]))
# Hex wider window, both bases
combos.append(("hexadecimal", 1024, [2, 5, 10, 100]))
combos.append(("hexadecimal", 1024, [16, 32, 64, 256]))
```

This mirrors `run.sh`'s passes. For Babylonian/binary/hex, we run *both*
the paper-default basis (to demonstrate the basis is mismeasuring the
structure) and the native basis (to recover the real signal). The control
quantifies how much of the recovered signal comes from learned embeddings
vs mechanically from rendering + pooling.

### 4.4 Output

A single `out/_random_embed_control.json` with a `results` array, one
entry per `(model, script, n_max, periods)` cell. Each entry has
`random_r2`, `learned_L0_r2`, `delta`, `seed`, `d_model`. The React site
reads this for the Finding 3 (Babylonian) and Finding-bases tables.

### 4.5 Worked example: n = 23, Babylonian on Pythia

The four-step chain is `rendering → tokenization → embedding lookup
→ mean-pooling`. Walking it through concretely:

**Step 1: rendering** — `format_number(23, "babylonian")` returns
`"𒌋𒌋𒁹𒁹𒁹"`: two ten-wedges followed by three one-wedges. The
structure is *additive*: the number of 𒌋 equals the tens digit, the
number of 𒁹 equals the ones digit. This is the renderer's job — it
turns the integer into a string whose shape encodes the arithmetic of
the integer.

**Step 2: tokenization** — we prepend a space and feed
`" 𒌋𒌋𒁹𒁹𒁹"` to Pythia's byte-level BPE tokenizer. Cuneiform was
not in Pythia's training corpus, so each codepoint falls back to its
UTF-8 byte representation (4 bytes per cuneiform codepoint, often
each byte becoming its own sub-token). Crucially, **every 𒌋 expands
to the same fixed byte-token sequence**, and likewise for 𒁹. If we
write `T_𒌋` for the sequence of byte-tokens making up one 𒌋, the
tokenization of 23 is

```
[space, T_𒌋, T_𒌋, T_𒁹, T_𒁹, T_𒁹]
```

i.e. a constant prefix plus 2 copies of `T_𒌋` plus 3 copies of `T_𒁹`.
The *count* structure of step 1 is preserved.

**Step 3: embedding lookup** — each token ID indexes into the
embedding matrix $E \in \mathbb{R}^{V \times d}$ (Pythia: $V \approx
50000$, $d = 4096$). We get one vector per sub-token. If we write
$e_{𒌋}$ for the contribution of one $T_{𒌋}$ (the sum or
concatenation of the byte-token rows), then the sequence of vectors is

```
[e_space, e_𒌋, e_𒌋, e_𒁹, e_𒁹, e_𒁹]
```

**Step 4: mean-pooling** — average the rows:

$$h(23) \approx \tfrac{1}{|\text{tokens}|}(e_\text{space} + 2\,e_{𒌋} + 3\,e_{𒁹})$$

This is the L=0 representation of the integer 23. It is a **linear
combination of three fixed vectors** with coefficients that come from
the renderer's count structure: 1 for the space, 2 (= the tens digit
of 23) for $e_{𒌋}$, 3 (= the ones digit) for $e_{𒁹}$.

Generalising to all $n < 60$:

$$h(n) \approx \alpha(n)\,e_{𒌋} + \beta(n)\,e_{𒁹} + \text{constant}$$

where $\alpha(n) = \lfloor n/10 \rfloor$ and $\beta(n) = n \bmod 10$.
Both are smooth/periodic functions of $n$.

**What the random-embedding control changes.** Steps 1, 2, 4 are
unchanged. Only step 3 changes: $e_{𒌋}$ and $e_{𒁹}$ are replaced by
fresh draws from $\mathcal{N}(0, 1/\sqrt{d})$. So

$$\tilde{h}(n) \approx \alpha(n)\,\tilde{e}_{𒌋} + \beta(n)\,\tilde{e}_{𒁹} + \text{constant}$$

The vectors $\tilde{e}_{𒌋}$ and $\tilde{e}_{𒁹}$ now point in random
directions in $\mathbb{R}^{4096}$, but they're still **fixed** vectors,
and the coefficients $\alpha(n)$ and $\beta(n)$ are the same smooth
functions of $n$. The trig basis fits the smooth coefficient structure
regardless of which directions $\tilde{e}_{𒌋}$ and $\tilde{e}_{𒁹}$
happen to point: $\beta(n) = n \bmod 10$ is fit by the $T = 10$
cos/sin columns, $\alpha(n) = \lfloor n/10 \rfloor$ is approximately
linear in $n$ for $n < 60$ so the linear column fits it.

The *direction* of the fitted helix in residual space rotates with
each random seed; the *quality* of the fit (R²) doesn't change.

**Why Latin is different.** Pythia tokenizes integers in [0, 557] as
single tokens, so for n = 23 on Latin:

```
" 23"  →  [T_23]  →  [e_23]
```

Mean-pooling over one token is the identity. So $h(23) = e_{23}$ for
Latin/Pythia, and the helix score is asking whether the 100 *learned
embedding vectors* $\{e_0, e_1, \ldots, e_{99}\}$ themselves lie on a
helix. With learned Pythia weights they do — Pythia learned that "23"
is the embedding-space neighbour of "22" and "24". Replace $E$ with a
random matrix and now $\{\tilde{e}_0, \ldots, \tilde{e}_{99}\}$ are
100 independent random vectors with no structure between them; the
helix R² collapses to ~0.08 (chance for fitting any low-d basis to
100 random points in $\mathbb{R}^{4096}$).

So the **same control** (swap learned E for random E) tells very
different stories on different cells:

- **Latin on Pythia**: large positive Δ ≈ 0.10. The L=0 helix is
  carried by the learned embeddings themselves; no rendering or
  pooling chain to fall back on (one token, one vector).
- **Babylonian on Pythia**: Δ ≈ 0. The L=0 helix is generated by
  the rendering+tokenization+pooling chain operating on counts, and
  the embedding values don't matter.

This is what we mean by "geometry from rendering alone." It isn't
mystical — it's the deterministic consequence of (i) an additive
renderer that turns $n$ into symbol counts, (ii) a tokenizer that
preserves those counts in the token sequence, and (iii) mean-pooling
that turns counts into convex combinations of fixed vectors. The trig
basis fits the resulting convex-combination weights as smooth
functions of $n$ — which is exactly what we asked it to do.

---

## 5. `subspace_align.py` — CKA

ρ alone is silent about *shape*. Two cells with the same high ρ can have
very different mechanistic stories:

- **Pass-through**: depth leaves the L=0 geometry untouched. ρ is high
  because the score doesn't change; CKA between L=0 and peak is also high.
- **Rebuild**: depth replaces the L=0 geometry with a different
  integer-to-integer pattern that happens to score the same. ρ is high
  (score unchanged), CKA is low.

To distinguish these we compute **linear CKA** between the helix-projected
representations at L=0 and the peak layer.

### 5.1 Linear CKA formula

```python
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    XtY = Xc.T @ Yc
    XtX = Xc.T @ Xc
    YtY = Yc.T @ Yc
    num = float((XtY * XtY).sum())
    den = float(np.sqrt((XtX * XtX).sum() * (YtY * YtY).sum()))
    return num / den if den > 0 else 0.0
```

Mean-centre both matrices, then

$$\mathrm{CKA}(X, Y) = \frac{\|X^\top Y\|_F^2}{\|X^\top X\|_F \cdot \|Y^\top Y\|_F}$$

Range $[0, 1]$. CKA is invariant to rotation and rescaling of the feature
spaces — it measures whether the *pattern of distances between rows* is
the same, not whether the rows themselves are aligned. (Kornblith,
Norouzi, Lee, Hinton 2019, "Similarity of neural network representations
revisited", arXiv:1905.00414).

### 5.2 What we measure CKA on

We have two natural choices:

- **Full residual stream**: $X = H^{(L=0)}$, $Y = H^{(\text{peak})}$.
- **Helix-projected**: $X = B W^{(L=0)}$, $Y = B W^{(\text{peak})}$.

We compute and report both (`cka_full` and `cka_helix`). The helix-projected
version is the one we read for the pass-through-vs-rebuild distinction.
Why? Because we care about whether *the helix* looks the same, not whether
the rest of the model state (which is doing whatever it does — language,
reasoning, etc.) looks the same. Projecting onto the helix subspace
isolates the geometry the trig basis fits.

### 5.3 The collection mechanics

`subspace_align.py` is structurally similar to `main.py` but does only what
CKA needs:

- For each model, load weights once and iterate all scripts inside that
  load (the bulk cost is loading 32B-class models).
- For each script:
  - Read the peak layer from `out/<model>/<script>/mean/fig_layer_sweep.json`
    (which `main.py` already wrote).
  - Collect activations at exactly two layers: `hidden_states[0]` and
    `hidden_states[peak_layer]`. This is cheaper than the all-layers sweep
    in `main.py` because we don't keep the intermediate layers.
  - Fit the helix basis on each → get $W^{(L=0)}$ and $W^{(\text{peak})}$.
  - Compute `cka_full(H_L0, H_peak)` and `cka_helix(B@W_L0, B@W_peak)`.
- Merge results into `out/_subspace_alignment.json`, de-duplicating by
  `(model, script)`. Re-running after the main sweep only costs new cells.

### 5.4 Peak layer = 0 edge case

When a cell's peak layer is 0 (e.g. Qwen2.5-7B/Latin), $H^{(L=0)}$ and
$H^{(\text{peak})}$ are the *same matrix*, so CKA is trivially 1. The
output JSON marks these cells as `trivial: true` and the React scatter
shows them faded. They're not informative for the alignment question;
they're literally the same point on the (ρ, CKA) plane.

### 5.5 Worked example: same ρ, different CKA

Two cells, both with high ρ — meaning the helix score at the peak
layer is mostly already at L=0. By ρ alone they look the same. CKA
reveals two very different mechanisms.

**Setup for both cells**:

1. Run `main.py` to get H at every layer and identify the peak layer.
2. Collect $H^{(L=0)}$ and $H^{(\text{peak})}$ — two matrices of shape
   $(100, 4096)$, one row per integer.
3. Fit the helix basis on each → $W^{(L=0)}$ and $W^{(\text{peak})}$.
4. Form the helix-projected representations $X = B W^{(L=0)}$ and
   $Y = B W^{(\text{peak})}$ (shape $(100, 4096)$ each — the best
   9-dimensional helix approximations of each layer, expanded back
   into the full hidden dimension).
5. Compute $\mathrm{CKA}(X, Y)$.

**Cell A: Qwen2.5-32B / Latin.** ρ ≈ 0.89. The helix at L=0 is already
near the peak score (Qwen splits Latin numbers per-digit, so
rendering + tokenization + mean-pooling already supplies a fittable
manifold). CKA ≈ 0.91.

Reading: both the score *and* the integer-to-integer relationships are
preserved through depth. **Pass-through**. The transformer doesn't
touch the helix.

**Cell B: Gemma-4-31B / Latin.** ρ ≈ 0.79. Similar to A — the score
barely changes. CKA ≈ 0.33.

Reading: same score story, but the integer-to-integer relationships
have *substantially shifted*. The peak-layer representation arranges
the 100 integers into a helix, but it's not the same arrangement as
the L=0 helix. **Rebuild**. Depth is doing significant representational
work; it just doesn't show up as a score change.

**Concrete intuition.** Imagine a 3D-helix plot at L=0 with integers
labelled. Now plot the 3D helix at the peak layer of the same model
on the same script.

- For Qwen-32B, the two plots are almost identical: integer 23 is in
  the same position on the helix in both, integer 47 is in the same
  position, etc. Depth left the arrangement alone.
- For Gemma-31B, the two plots have the same overall helical shape
  but the labels have moved: integer 23 might be at the back of the
  helix at L=0 and at the front at the peak. The helix is still a
  helix, but a *different* one — the row-similarity pattern (which
  integers are nearby which) has changed.

CKA is the formal version of "look at the two plots and tell me if the
labels are in the same positions". It's invariant to rotation and
rescaling of the feature columns (we don't care if the helix has spun
or stretched in residual space), so what it measures is precisely the
row-similarity pattern across integers.

**Why we report helix-projected CKA**. We compute two flavours:

- `cka_full` = $\mathrm{CKA}(H^{(L=0)}, H^{(\text{peak})})$ — across
  the full 4096-d residual stream.
- `cka_helix` = $\mathrm{CKA}(B W^{(L=0)}, B W^{(\text{peak})})$ —
  across the 9-d helix subspace re-embedded into 4096-d.

The full version mixes in everything: language structure, syntactic
features, attention metadata, everything the residual stream carries.
That's noisy for our question — we want to know whether *the helix*
moved, not whether the whole model state did. The helix-projected
version isolates the trig-fittable part of the activations and
compares only that.

**What this enables: the four-quadrant scatter.** Plot ρ on the x-axis
and CKA on the y-axis. Each (model, script) cell becomes one point.
Four named regions emerge:

- **Top-right** (high ρ, high CKA) — **pass-through**. The input
  pipeline supplies the helix and depth doesn't touch it.
  Qwen2.5-32B/Latin, OLMo/Latin, Pythia × non-Latin positional.
- **Top-left** (low ρ, high CKA) — **depth refines**. The L=0 helix
  is partial; depth amplifies its score while preserving its shape.
  Pythia/Latin, Llama/Latin, GPT-J/Latin — K&T's three clean cells.
- **Bottom-right** (high ρ, low CKA) — **rebuild**. Score barely
  changes; geometry does. Whole Gemma family, Babylonian on
  Pythia/Llama/OLMo, Qwen on Devanagari.
- **Bottom-left** (low ρ, low CKA) — **depth builds new**. Score
  improves *and* shape changes. Rare. Gemma's strongest non-Latin
  cells edge here.

ρ alone collapses top-right and bottom-right into one category
("high ρ → inherited"). CKA recovers the distinction: top-right is
genuine inheritance (depth idle), bottom-right is depth doing
significant work in a way that doesn't move the score. The whole
"Gemma/Latin is mostly tokenizer" reading from ρ alone turns out to
be wrong once CKA is added — Gemma is rebuilding, not inheriting.

---

## 6. `aggregate.py` — building `_index.json`

A trivial walker that collects every per-cell `fig_layer_sweep.json` under
`out/` into a single master index file.

```python
for path in OUT.rglob("fig_layer_sweep.json"):
    # extract (model, script, pool, n_max, periods) from path components
    model = path.parts[-4]      # e.g. "EleutherAI__pythia-6.9b"
    script = path.parts[-3]     # e.g. "latin"
    pool_dir = path.parts[-2]   # e.g. "mean" or "mean_n600_p2-5-10-60-100"
    parsed = parse_pool_dir(pool_dir)
    if parsed is None: continue
    ...
```

`parse_pool_dir(name)` is the inverse of the `_n<>` / `_p<>` suffix logic
in `main.py`. Given `"mean_n600_p2-5-10-60-100"`, it returns
`{"pool": "mean", "n_max": 600, "periods": [2,5,10,60,100]}`.

The output `out/_index.json` is a JSON document with one entry per cell:

```json
{
  "generated_at": "2026-05-27T...",
  "n_cells": 144,
  "cells": [
    {
      "model": "EleutherAI/pythia-6.9b",
      "script": "latin",
      "pool": "mean",
      "n_max": 100,
      "periods": [2, 5, 10, 100],
      "n_layers": 32,
      "helix_r2_l0": 0.183,
      "helix_r2_peak": 0.533,
      "peak_layer": 32,
      "rho": 0.344,
      ...
      "paths": {
        "dir": "...",
        "layer_sweep": "...",
        "fourier_pc1": "...",
        ...
      }
    },
    ...
  ]
}
```

The `paths` block holds relative paths so the React site can `fetch()`
them via the `/data/` Vite alias. The aggregator also writes
`out/_l0_share.csv` for spreadsheet-style consumption.

This is the single file the blogpost loads first — every chart finds its
cell here, then lazy-loads the per-figure JSON pointed to by `paths`.

---

## 7. `compare.py` — stacked PNG image

A static utility that stitches per-model layer-sweep PNGs into a single
tall comparison image for the paper. Reads each model's
`fig_layer_sweep.png`, adds a header strip labelled with the model ID and
its total transformer-layer count (read from the matching `.json` so the
number stays in sync with what the sweep actually saw), stacks them
vertically. Writes `out/_compare/fig_latin_sweep_<N>models.png`.

This is the only piece of the pipeline that produces an artifact specifically
for the paper rather than the website. The website draws layer sweeps
interactively from the per-cell JSONs.

---

## 8. `run.sh` — the driver

A bash script that orchestrates everything. Idempotent, filterable, and
verbose enough to follow without reading the log files.

### 8.1 The plan

```
Pass 1: 8 models × 12 scripts at n=100, basis=[2,5,10,100]  (paper-default)
Pass 2: 8 models × babylonian at n=600, basis=[2,5,10,100]  (window-only fix)
Pass 3: 8 models × babylonian at n=600, basis=[2,5,10,60,100] (window + T=60)
Pass 4: 8 models × binary at n=1024, basis=[2,5,10,100]
Pass 5: 8 models × binary at n=1024, basis=[2,4,8,16,32,64]
Pass 6: 8 models × hex at n=1024, basis=[2,5,10,100]
Pass 7: 8 models × hex at n=1024, basis=[16,32,64,256]
Then: aggregate, embed_control, subspace_align, compare
```

Passes 2-7 exist because the paper-default protocol is the *wrong*
protocol for Babylonian, binary, and hex. We run both the paper-default
basis (to demonstrate the failure) and the native basis (to recover the
real signal). The contrast between the two gives the basis-bandwidth
diagnostic concrete numbers.

### 8.2 Why these specific extended bases?

**Babylonian at n=600**: T=60 needs the input window to wrap many times
to be identifiable. n=100 wraps T=60 only once. n=600 wraps it ten times,
which is enough for any FFT/regression to see it. Adding T=60 to the basis
then gives the trig basis a column to fit the structure with.

**Binary at n=1024**: binary's natural periods are powers of two up to 64.
n=1024 gives 16 wraps of T=64, plenty for identification. The basis
`[2,4,8,16,32,64]` is "all powers of 2 below n_max/16".

**Hex at n=1024**: hex's natural periods are 16, 32, 64, 256. n=1024 gives
4 wraps of T=256. The basis `[16,32,64,256]` is the native set;
128 we drop because it doesn't appear as a natural feature (it's not
"one nibble" of anything common).

### 8.3 Marker files for idempotency

```bash
marker_for() {
  local model="$1"; local script="$2"; local n_max="$3"; local periods="$4"
  local model_dir="${model//\//__}"
  local pool="mean"
  if (( n_max != 100 )); then pool="mean_n${n_max}"; fi
  if [[ -n "${periods}" && "${periods}" != "2,5,10,100" ]]; then
    pool="${pool}_p${periods//,/-}"
  fi
  echo "out/${model_dir}/${script}/${pool}/fig_layer_sweep.json"
}
```

This must mirror `main.py`'s output-directory logic exactly. If they
diverge, `run.sh` will think a cell is done when it isn't (or vice
versa). Marker is the JSON not the PNG so that any cell produced by an
older version of `main.py` (pre-JSON-export) gets re-run.

### 8.4 `script(1)` for live tqdm

```bash
script -q "${log_file}" uv run main.py --model "${model}" --script ...
```

`script(1)` records a session inside a pseudo-TTY. This is so `main.py`'s
stderr looks like a real terminal to tqdm, which then refreshes the bar
in place (`\r`) rather than buffering it through `tee` and dumping at the
end. Also tells HuggingFace's "Loading checkpoint shards" bar to display
live during the multi-minute 32B-model loads.

### 8.5 Pre-flight count and per-cell status lines

Before running anything, `run.sh` counts the total combos and how many
are already done. Each cell prints either `SKIP` or `RUN`, and on
completion prints the headline numbers from the log (`PC1 R²`, `helix R²`,
peak layer, basis periods). On failure, it prints `!! FAILED` and
continues — one bad cell doesn't abort the rest.

### 8.6 The post-sweep tools

After the per-cell sweep, `run.sh` runs aggregate / embed_control /
subspace_align / compare in sequence. Each writes to its own log under
`out/_logs/` so failures can be debugged independently.

### 8.7 Worked example: why basis + window matter

The paper-default protocol is $n_{\max} = 100$ with periods
$\mathcal{T} = \{2, 5, 10, 100\}$. This is the right protocol for
Latin on Pythia/Llama: the model encodes integers using period-10
(the units digit) plus small auxiliary periods like T=2 (parity) and
T=5 (mod-5). The window of 100 integers wraps T=10 ten times and T=5
twenty times — plenty for either FFT or regression to identify them.

But the protocol's two assumptions can fail.

**Failure 1: basis bandwidth.** The trig basis can only fit periods
it explicitly contains. Babylonian renders numbers using base 60. If
the model encodes that structure at all, the natural period is T=60.
But T=60 is not in $\{2, 5, 10, 100\}$, so the basis has no cos/sin
column to fit it. Even if the model perfectly encodes base-60
structure, the fit reports low R² because the basis doesn't have the
right columns.

The analogy: trying to detect a 1 kHz tone with a microphone that
only captures frequencies between 5 kHz and 20 kHz. The tone is there;
the instrument can't see it. The microphone reports "no signal" but
that's not a fact about the room.

**Failure 2: window wrap.** Even if we put T=60 in the basis, we
still have a problem at $n_{\max} = 100$. From 0 to 99, T=60 wraps
*once* (at n=60). One wrap is not enough to identify a periodic
structure.

The Fourier intuition: the FFT of a length-100 signal at frequency
$f = 1/60$ doesn't form a clean single bin because 100 isn't an
integer multiple of 60. The energy of a true T=60 component spreads
across multiple frequency bins ("spectral leakage"), making it
indistinguishable from noise. With 10 cycles in 600 samples (i.e.
$n_{\max} = 600$), the leakage is small and the peak at f = 1/60 is
sharp and unambiguous.

The regression intuition: with one wrap of T=60 over 100 samples,
the columns $\cos(2\pi a/60)$ and $\sin(2\pi a/60)$ are *highly
correlated* with all sorts of other columns and with each other.
The OLS solver can fit "something" but the result is unstable and not
meaningfully attributable to period-60 structure. With ten wraps, the
columns are well-separated from each other and from auxiliary
features; the solver gets a clean, well-identified fit.

Rule of thumb: you need the input range to wrap any candidate period
at least 8-10 times for clean identification.

**The fix has to address both.** For Babylonian we run two extended
passes:

| pass | n_max | basis | what's tested |
|---|---|---|---|
| 1 | 100 | $\{2,5,10,100\}$ | paper default — score is low |
| 2 | 600 | $\{2,5,10,100\}$ | wider window only — score still low |
| 3 | 600 | $\{2,5,10,60,100\}$ | window + basis fixed — score jumps |

Pass 2 is informative on its own: it tells us that fixing only the
window isn't enough — the basis was also wrong. Pass 3 then shows
that the structure was actually there in the model; the original "no
helix on Babylonian" reading at the paper defaults was a measurement
artifact, not a fact about the model.

The contrast between pass 2 (window-only fix, helix R² stays low) and
pass 3 (window + basis fix, helix R² jumps 8-16pp uniformly across all
8 models) is the basis-bandwidth diagnostic with numbers attached.
The basis-free FFT in pass 3 also shows clean peaks at multiples of
1/60, confirming the model really does encode period-60 structure
that the paper basis could never have caught.

**Same logic for binary and hex.**

Binary's natural periods are powers of two: $T \in \{2, 4, 8, 16,
32, 64\}$. The paper basis $\{2, 5, 10, 100\}$ catches only T=2. We
run at $n_{\max} = 1024$ (16 wraps of T=64, ample headroom) with
basis $\{2, 4, 8, 16, 32, 64\}$. Helix R² recovers +0.10 to +0.37
across models.

Hex's natural periods are $\{16, 32, 64, 256\}$. Paper basis catches
*none* of these — even T=16 is missing. We run at $n_{\max} = 1024$
(4 wraps of T=256) with basis $\{16, 32, 64, 256\}$. Helix R² recovers
+0.19 to +0.29 across models.

The recipe is identical in each case: extend the window past several
wraps of the script's natural period, add the natural period to the
basis. The result is mechanical; what *differs* across the three
case studies is the random-embedding control's verdict (mechanical
for Babylonian and binary; learned for hex). The basis/window fix is
a *precondition* for asking the provenance question; the random-embed
control then answers it.

**Why this matters beyond non-decimal bases.** The basis-bandwidth /
window-wrap failure modes aren't specific to Babylonian or binary or
hex. Any paper looking at a representation through "fit a trig basis
and report R²" can fall into them. If you propose that a model
encodes some quantity X using periods $\{T_i\}$, you have to ensure:

1. Your basis explicitly contains $(\cos, \sin)$ columns for each
   $T_i$. If you guess wrong, the fit reports false-negative.
2. Your input range wraps each $T_i$ many times. If the window is too
   short, even a correct basis fails to identify the period.

The check for both is just: vary the basis, vary the window, and see
if the score changes. If it does, the original protocol was hiding
real structure; if it doesn't, the original reading was right.
Cheap, mechanical, and worth doing on any cell that reports a
surprisingly low (or surprisingly high) score.

---

## 9. The React site (`www/`)

A from-scratch interactive write-up. React 18 + TypeScript + Vite + Tailwind
+ d3 (for SVG charts) + three.js (for the 3D helix).

### 9.1 Project structure

```
www/
├── index.html
├── package.json
├── tailwind.config.js
├── vite.config.ts
├── scripts/stage-data.mjs    rsync-style copy out/ → dist/data/ for production
└── src/
    ├── main.tsx              React entry point
    ├── App.tsx               composes the sections in order
    ├── index.css             Tailwind + page typography
    ├── lib/
    │   ├── data.ts           memoised JSON loader; MODEL_ORDER, SCRIPT_ORDER, etc.
    │   ├── types.ts          TypeScript mirrors of every JSON schema
    │   ├── svg.ts            VIRIDIS + ρ heatmap colour scale
    │   └── numerals.ts       per-script label generators
    └── components/
        ├── ChartFrame.tsx    lazy-load wrapper + "data not available" fallback
        ├── PartHeader.tsx    "Part 1: Setup" / "Part 2: ..." headings
        ├── charts/
        │   ├── FFTSpectrum.tsx       (top panel of fig2)
        │   ├── PC1Scatter.tsx        (bottom panel of fig2)
        │   ├── CirclePanel.tsx       (one panel of fig3)
        │   ├── HelixViewer3D.tsx     (three.js 3D helix)
        │   ├── LayerSweep.tsx        (the three-panel R² sweep)
        │   ├── RhoHeatmap.tsx        (8x12 ρ matrix)
        │   ├── ClassificationGrid.tsx (8x12 grid of cell classes)
        │   ├── PCA2DScatter.tsx      (2D PCA basis-free)
        │   └── RhoCkaScatter.tsx     (ρ vs CKA, the 4-quadrant scatter)
        └── sections/
            ├── Hero.tsx              title + abstract
            ├── Findings.tsx          TL;DR claim list
            ├── NumeralExplainer.tsx  12 scripts side-by-side
            ├── WhatTheHelixIs.tsx    helix intuition + 3D viewer
            ├── TheDiagnostic.tsx     the four checks + methods box
            ├── Finding1Layers.tsx    layer sweep on Latin
            ├── Finding2Scripts.tsx   cross-script panel
            ├── Finding3Babylon.tsx   Babylonian case study
            ├── FindingBases.tsx      binary/hex case study
            ├── Finding4L0.tsx        ρ heatmap + classification grid
            ├── Finding5CKA.tsx       representation alignment (ρ, CKA) scatter
            ├── Conclusion.tsx        four-way classification + lesson
            └── Reproducing.tsx       how to run the code
```

### 9.2 The data layer

`src/lib/data.ts` is the single source of truth for:

- `MODEL_ORDER` (8 entries, HF IDs in display order)
- `MODEL_LABEL` (HF ID → display name like "Pythia-6.9B")
- `MODEL_COLOR` (HF ID → hex color, used in every chart that needs a
  per-model legend; extracted in the DRY pass)
- `SCRIPT_ORDER` (12 entries)
- `SCRIPT_LABEL` (script ID → display name)
- `SCRIPT_EXAMPLE` (script ID → "23" rendered in that script)

Plus three loader functions:

```typescript
export function dataUrl(rel: string): string {
  return `/data/${rel.replace(/^\/+/, "")}`;
}

export async function loadJson<T>(rel: string): Promise<T> {
  // memoised fetch, returns cached promise on repeat calls
}

export function loadIndex(): Promise<IndexDoc> {
  return loadJson<IndexDoc>("_index.json");
}

export function findCell(cells, match): IndexCell | undefined {
  // linear scan for the cell matching {model, script, n_max, periods}
}
```

`/data/` is a Vite dev-server alias to `../out/` (set in `vite.config.ts`).
In production, `scripts/stage-data.mjs` copies the JSONs (excluding PNGs,
which the site doesn't need) into `dist/data/`.

The memoisation matters because each section component independently calls
`findCell()` and `loadJson()` for its target cell — without caching we'd
fetch each JSON multiple times.

### 9.3 How a section renders

Take `Finding1Layers.tsx` (the demo of "where the helix lives varies
across models"):

```typescript
export function Finding1Layers({ index }: Props) {
  const [docs, setDocs] = useState<...>([]);
  useEffect(() => {
    if (!index) return;
    const cells = MODEL_ORDER.map((m) =>
      findCell(index.cells, { model: m, script: "latin", n_max: 100, periods: [2,5,10,100] }),
    );
    Promise.all(
      cells.map(async (c) => c?.paths.layer_sweep
        ? { label: MODEL_LABEL[c.model], color: MODEL_COLOR[c.model],
            doc: await loadJson<LayerSweepDoc>(c.paths.layer_sweep) }
        : null,
      ),
    ).then((arr) => setDocs(arr.filter(Boolean)));
  }, [index]);
  return <LayerSweep docs={docs} metric="helix_r2" />;
}
```

It pulls the eight Latin layer-sweep JSONs in parallel, wraps each with the
model's display label + color, and passes them to the `LayerSweep` chart
component which overlays them.

Every chart works this way: the chart component is pure (data in, SVG/Canvas
out), the section component does the data wrangling and section-level prose.

### 9.4 Why d3 + React (not just one or the other)?

- d3 has the math primitives we need (scales, color interpolators, line
  generators, axis tick helpers).
- React owns the DOM. Charts are SVG elements declared in JSX — re-renders
  are cheap and we get React's whole composability story.

We never let d3 touch the DOM. d3 computes layout, React renders. This is
sometimes called "render-with-React, layout-with-d3" — a well-known
pattern for keeping the two libraries from fighting.

### 9.5 The 3D helix

`HelixViewer3D.tsx` is the only chart using three.js. The 3D helix from
`fig1_helix_T10.json` (which contains pre-computed 3D coordinates from the
QR-orthonormalised projection) is rendered as a Points cloud + Line
trajectory. Rotation is handled by OrbitControls. We use a `useEffect` to
spin up the renderer on mount and tear it down on unmount.

### 9.6 Sections compose into Parts

`App.tsx` composes the components into five named Parts:

```
Part 1: Setup           NumeralExplainer + WhatTheHelixIs
Part 2: What can go     TheDiagnostic
        wrong with R²
Part 3: Reading the     Finding1Layers + Finding2Scripts + Finding4L0 + Finding5CKA
        matrix
Part 4: Case studies    Finding3Babylon + FindingBases
Part 5: Implications    Conclusion
```

Plus Hero + Findings TL;DR above Part 1, and Reproducing below Part 5. The
PartHeader component prints the "Part N" label and styled title.

---

## 10. `paper.tex` — the static archive

A self-contained two-column write-up that approximates ICML 2024 styling
without needing the official `icml2024.sty`.

### 10.1 Preamble decisions

- **Plain article class** with `twocolumn` switched on manually after the
  title block. We tried the standard `\twocolumn[\begin{@twocolumnfalse}...]`
  pattern but it has scoping bugs with `\begin{abstract}`. The current
  layout uses `\maketitle` + a manual `\begin{center}\large\bfseries Abstract
  \end{center}\begin{quote}...\end{quote}` block, then `\twocolumn` to start
  the body.
- **xelatex** (not pdflatex) so `fontspec` and `xeCJK` can render the 12
  numeral scripts inline (Arabic-Indic, Devanagari, Thai, CJK, cuneiform,
  Greek, Hebrew, Roman, binary, hex, Latin).
- **Manual `\setCJKmainfont{Hiragino Sans GB}`** because the xeCJK default
  Fandol isn't installed on macOS.
- **`natbib` with `[round,authoryear]`** for inline `\citet{...}` and
  `\citep{...}` citations.
- **No external `.sty`**. The user told me to stay "in the ballpark" of
  ICML formatting without requiring the official style file.

### 10.2 Structure

```
Section                          Pages
─────────────────────────────────────
Title + abstract                 p1
1. Introduction                  p2
2. What We Measure Per Cell      p2-p3
   (Table 1: scripts at top p3)
3. Experimental Setup            p4
4. Results                       p4-p6
   (Figure 2: ρ heatmap, p5)
5. Discussion                    p6
References                       p7
A. Methods details               p8
B. Per-model peak layers
C. Binary and hex case studies
D. Random-embed across all cells
E. (ρ, CKA) scatter + 4 quadrants
F. Classification grid (full)
G. Reproducibility               p10
```

The two prominent visual elements are:

- **Figure 1**: a single static 3D helix PNG from
  `out/EleutherAI__pythia-6.9b/latin/mean/fig1_helix_T10.png`, embedded in
  §1 as the "this is what we're studying" anchor.
- **Figure 2**: a hand-coded LaTeX heatmap of all 96 ρ values, with
  `\cellcolor{...}` per cell drawn from a 5-band teal-to-orange palette.
  No external image; the heatmap is rendered by LaTeX itself, so the
  numbers are searchable and the colors print consistently.

Table 1 (the 12 numeral scripts with renderings of 23 *and* 24) spans
both columns at the top of p3 thanks to the `\begin{table*}[t]` float and
its placement near the top of §2.

### 10.3 Bibliography

Inline `\begin{thebibliography}` (no `.bib` file), wrapped in `{\small ...}`
so the 6 references fit at the bottom of p6 alongside Discussion's tail.
Each entry uses the natbib `\bibitem[Author(Year)]{key}` format for
author-year citations.

---

## 11. Cross-cutting design decisions

A few things that show up everywhere and deserve their own subsections.

### 11.1 The space prefix on every input

`text = f" {format_number(int(n), script)}"` in `main.py`,
`embed_control.py`, and `subspace_align.py`. This matches K&T's convention,
which exists for two reasons:

1. **GPT-style tokenizers split `"23"` and `" 23"` into different tokens.**
   The leading space marks the start of a token, and BPE tokenizers
   treat `" 23"` as one token (or one prefix) where `"23"` would be
   handled differently. The leading space gives us the natural-prompt
   tokenization.
2. **In real prompts, numbers come after a space or operator.** A number
   that the model has been trained to recognize comes in context like
   `"answer: 23"` or `"compute 17 + 6"`. We want the tokenization that
   matches how the model would actually see the number.

If you remove the space, the tokenization for some models changes, and
the L=0 mean-pooled vector changes accordingly. That's not a bug, but
it would mean we're measuring a different thing than K&T did, and the
comparison to their result would be off.

### 11.2 What L=0 means

We chose `hidden_states[0]` = the residual stream *before any transformer
block has run* = the output of token embedding + positional embedding +
mean-pooling. This is one step *before* K&T's "following layer 0", which
is the output of block 0 in their setup.

We made this choice deliberately for the provenance question: we want to
know what the *input pipeline alone* (rendering, tokenizing, looking up
embeddings, pooling) supplies. If we measured after one block of
computation, we'd be conflating "what the input pipeline supplies" with
"what one transformer block does on top of it".

This convention is called out in §2 of the paper and in the methods box
of the blogpost, so a reader comparing to K&T's plots knows they're not
identical baselines.

### 11.3 Mean pooling

Mean over the numeral's sub-tokens, at every layer. Three reasons:

1. **Last-token pooling produces a period-10 cycle artifact** on multi-token
   numerals because numbers ending in the same digit share the same last
   sub-token. We discussed this in §3.6.
2. **Mean pooling is more symmetric across scripts.** "23" is one token
   on Pythia/Latin, two on Pythia/Arabic, multiple on Babylonian. Mean
   gives every position equal weight, which is what we want when
   comparing across tokenizers.
3. **Mean pooling is itself part of the provenance story.** For additive
   renderings (Babylonian, binary, Roman), mean turns symbol counts into
   linear arithmetic on token embeddings — which is exactly the kind of
   structure the trig basis fits. See §6 in the paper, the Babylonian
   case study in the blogpost, and §3.6 above.

### 11.4 What gets recorded per cell

Every `main.py` run on a cell writes:

- A `_meta.json` with the run parameters.
- For every layer L: PC1 R², helix R², K-d PCA R², helix/PCA ratio.
- The peak layer (where helix R² is maximised).
- ρ = helix R²(L=0) / helix R²(peak).
- The four standard figures (PNG + JSON) at the peak layer.

`aggregate.py` rolls these up into `out/_index.json`. `embed_control.py`
adds `out/_random_embed_control.json`. `subspace_align.py` adds
`out/_subspace_alignment.json`. The blogpost and paper read this set as
a single coherent dataset.

### 11.5 Idempotency everywhere

Both `run.sh` (skip combos whose `fig_layer_sweep.json` exists) and
`subspace_align.py` (merge results, skip rows already in the output)
are idempotent. This matters because the full sweep takes 6-8 hours;
adding a new model or script should only cost the new cells, not the
re-run of every existing one.

### 11.6 Why these 8 models and not others

| model | role |
|---|---|
| Pythia-6.9B | K&T's first model. Single-token Latin. Monotonic helix to final layer. |
| GPT-J-6B | K&T's second. Same single-token regime, similar peak-at-final-layer pattern. |
| Llama-3.1-8B | K&T's third. Single-token Latin too, mid-stack peak. |
| Gemma-4-E4B | Smaller multilingual. Per-digit tokenizer. Rebuild signature. |
| Gemma-4-31B | Larger multilingual. Sharp narrow peak. |
| OLMo-3-32B | Recent open base model with documented training mix. |
| Qwen2.5-7B | Per-digit tokenizer + Latin peak at literal L=0. Pre-transformer extreme. |
| Qwen2.5-32B | Same family at scale, slightly different. |

The set extends K&T's three with five models chosen to span the dimensions
that K&T's set doesn't: multilingual (Gemma), per-digit tokenization (Qwen),
and a fully-documented open model (OLMo). The 32B-class triple
(Gemma-31B, OLMo-32B, Qwen-32B) lets us see whether scale changes the
provenance picture independent of architecture (it doesn't, much).

### 11.7 Why these 12 scripts and not others

| family | scripts | purpose |
|---|---|---|
| positional base-10, Unicode-offset | Latin, Arabic-Indic, Persian, Devanagari, Thai | core comparison: same math, different glyphs |
| positional base-10, CJK | CJK digit string | tokenization edge case (CJK is multi-byte UTF-8) |
| non-decimal positional | binary, hexadecimal | basis-bandwidth test (paper basis misses native periods) |
| additive | Greek alphabetic, Hebrew alphabetic, Roman | negative control: no positional period-10 structure |
| base-60 positional with additive cols | Babylonian cuneiform | the cleanest mechanical case |

12 scripts × 8 models = 96 cells at paper defaults, plus 6 extended-basis
runs × 8 models = 48 additional cells. 144 cells total in the
sweep.

---

## 12. The full data flow

Following one cell from start to finish:

```
input: model="Pythia-6.9B", script="latin", n_max=100, periods=[2,5,10,100]

run.sh
  → marker check: out/EleutherAI__pythia-6.9b/latin/mean/fig_layer_sweep.json
  → not present, run

uv run main.py --model EleutherAI/pythia-6.9b --script latin --sweep
  → load model + tokenizer (Pythia, 32 layers, d=4096)
  → for each a in 0..99:
       text = " " + str(a)
       ids = [BOS, *tokenize(text)]
       out = model(ids, output_hidden_states=True)
       per_layer[L] = mean(out.hidden_states[L][0, start:end, :])
  → H_all: array of shape (100, 33, 4096)
  → run_sweep_analysis: PC1 R², helix R², PCA R² at every L
  → peak L = argmax(helix_r2)
  → ρ = helix_r2[0] / helix_r2[peak]
  → set RUN_META["layer"] = peak
  → plot_fourier_and_pc1, plot_circles_and_line, plot_helix_3d, plot_pca_2d
  → dump out/EleutherAI__pythia-6.9b/latin/mean/{fig1..fig4,fig_layer_sweep}.{png,json}

run.sh (continues with next cell)
... (96 + 48 cells)

uv run aggregate.py
  → walk out/, write out/_index.json + out/_l0_share.csv

uv run embed_control.py
  → for each cell in default_combos():
       tokenize each integer's rendering
       random_E ~ N(0, 1/sqrt(d))
       random_h(a) = mean of random_E rows for the numeral's sub-tokens
       random_r2 = helix fit on random_H
       learned_r2 = read from fig_layer_sweep.json's helix_r2[0]
       delta = learned_r2 - random_r2
  → write out/_random_embed_control.json

uv run subspace_align.py --scripts all
  → for each model:
      load model once
      for each script:
        peak_layer from fig_layer_sweep.json
        collect hidden_states[0] and hidden_states[peak]
        fit helix on each → W_L0, W_peak
        compute linear_cka(H_L0, H_peak) and linear_cka(B@W_L0, B@W_peak)
  → merge into out/_subspace_alignment.json

uv run compare.py
  → stitch per-model Latin layer-sweep PNGs into one tall image
  → write out/_compare/fig_latin_sweep_8models.png

(blogpost dev)
cd www && npm run dev
  → vite dev server on :5173, /data/ aliased to ../out/
  → React loads /data/_index.json
  → each section finds its cell and fetches per-figure JSONs lazily

(paper build)
./build_paper.sh
  → xelatex paper.tex (twice)
  → paper.pdf, 10 pages
```

---

## 13. Reading order if you've just cloned this

1. **README.md** — 50-line orientation.
2. **This file** (walkthrough.md) — what you're reading.
3. **`helix_lib.py`** — read in full, ~250 lines. The math + numerals.
4. **`main.py`** — skim the `collect_activations_all_layers` and
   `run_sweep_analysis` functions; the plot functions are routine.
5. **`embed_control.py`** — the random-embedding control's mechanics.
6. **`subspace_align.py`** — CKA, peak-layer collection.
7. **`run.sh`** — the orchestration.
8. **`www/src/App.tsx`** — see how sections compose.
9. **`www/src/lib/data.ts`** — the data layer.
10. **`paper.tex`** — the formal write-up.
11. **The blogpost** at `http://localhost:5173` after `npm run dev` — the
    interactive version with charts.

---

## 14. What this walkthrough does not cover

- **Numbers and results.** Read the blogpost or `paper.pdf`.
- **Future work / open questions.** §5 of the paper.
- **The K&T paper itself.** Read it (arXiv:2502.00873).
- **Mechanistic interpretability prerequisites.** The Nanda et al. progress
  measures paper (arXiv:2301.05217) is the right entry point for the
  small-model side.
- **Why CKA over other similarity metrics.** Kornblith et al.
  (arXiv:1905.00414) defended linear CKA against alternatives. We adopt
  their conclusion.

Everything *we did* and *why* is here. If a decision in the code surprises
you and it's not explained, that's a bug in this walkthrough — open an issue.
