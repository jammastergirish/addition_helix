# addition_helix

A from-scratch replication of **Kantamneni & Tegmark, _"Language Models Use
Trigonometry to Do Addition"_** ([arXiv:2502.00873](https://arxiv.org/abs/2502.00873),
2025), with an extension that lets you swap the numeral script the model
sees (Latin, Arabic-Indic, Persian, Devanagari, Chinese-positional, Greek
alphabetic, or Roman).

## The claim, in one paragraph

For an integer `a ∈ [0, 99]`, the residual stream of a transformer LM
encodes `a` as a **generalised helix** — one linear "number-line" axis
plus four orthogonal modular circles of period `T ∈ {2, 5, 10, 100}`.
T=2 captures parity, T=5 captures `a mod 5`, T=10 captures the units
digit, T=100 captures coarse position. Addition is performed by
*rotating* these circles — the paper's "Clock" algorithm.

## What `main.py` does

### Step 1 — Collect activations

We run the model on each integer `a ∈ [0, 99]` (one forward pass per
integer) and grab the **residual stream** at a middle layer. The
residual stream is the per-token `d`-dimensional vector that every
transformer block reads from and adds to — the running scratch-pad of
the model. For Pythia-6.9B, `d = 4096`.

Stacking those vectors gives a matrix

```
H ∈ ℝ^(100 × 4096)
```

where row `a` is the model's representation of integer `a` at the
chosen depth. Every analysis below operates on this matrix.

### Step 2 — FFT each hidden dimension (the discovery step)

This is the most pedagogically interesting move. Take a single column
of `H`, say column 17:

```
   a:    0       1       2       3       ...   99
   H[:,17]:  +0.12  -0.45  +0.07  -0.61  ...  +0.31
```

That's 100 numbers indexed by integer `a`. Forget for a second that
`a` is an integer; treat it as the time axis. **This is a 100-step
signal**, and the question we ask of any signal is the FFT question:

> "Can I write this wiggle as a sum of sine waves? Which frequencies
> have the biggest amplitudes?"

The FFT decomposes the column into a sum of sinusoids
`A_k · sin(2π k a / 100 + φ_k)` and returns the amplitudes `A_k`. A
big `A_k` at frequency `f = k/100` means *"this column oscillates with
period `100/k`"*.

So:
- A peak at `f = 1/10` ⇒ this column has a strong wiggle with **period
  10** ⇒ the dimension cares about `a mod 10` (i.e. the units digit).
- A peak at `f = 1/2` ⇒ period 2 ⇒ the dimension flips with parity.

Now do this for *every one of the 4096 columns* and average the
magnitudes. Random noise washes out; periods that many dimensions
agree on survive averaging. The top panel of `fig2_fourier_pc1.png` is
this averaged spectrum. **We did not put the periods in.** The peaks
at `T = 2, 5, 10, 100` emerge from the data — the paper's central
empirical finding.

### Step 3 — Verify the linear "spine"

PCA finds the directions of maximum variance in `H`. PC1 — the single
direction explaining the most variance — should be nearly **linear in
`a`** if `h(a)` has a strong magnitude axis (the "rise" of the
helix). The bottom panel of `fig2_fourier_pc1.png` plots PC1 vs `a`
and reports the R² of the linear fit; values above ~0.9 mean the
number line is cleanly present.

### Step 4 — Fit the helix

We construct the helix basis matrix

```
B(a) = [a, cos(2πa/T_1), sin(2πa/T_1), ..., cos(2πa/T_K), sin(2πa/T_K)]
```

with `T = [2, 5, 10, 100]` — 1 linear + 8 trig columns = 9 features.
We solve `H ≈ B @ W` by ordinary least squares. The columns of `W`
tell us **the direction in residual space** corresponding to each
helix feature: where `cos(2πa/10)` lives, where the linear axis
points, etc. Those directions are what the next two plots project
onto.

The script prints **two R² numbers**:
- `helix R²`: how much variance the 9-feature trig basis explains.
- `9-d PCA R²`: how much *any* 9-dim basis could possibly explain
  (the Eckart-Young upper bound).

If their ratio is close to 1.0, the helix isn't just *a* good 9-D
fit — it *is* the 9-D structure of the residual stream. The paper's
quantitative claim is that this ratio is ~0.85 for Latin on Pythia
(reproduced here).

### Step 5 — Visualise the modular circles

For each period `T`, take the two columns of `W` corresponding to
`cos(2πa/T)` and `sin(2πa/T)`. These are two vectors in residual
space — the directions where the cosine and sine features live. To
draw a 2D plot we want to use these two vectors as the axes of the
plot, but they aren't perpendicular to each other and aren't the
same length, so using them naively would render a true circle as a
tilted, stretched ellipse (a "bent ruler" problem).

We fix this by replacing the two vectors with a clean pair that
spans the **same 2D plane** but is **perpendicular and unit-length**.
This is the standard Gram-Schmidt orthonormalisation procedure (in
the code, `np.linalg.qr` does it in one call). Projecting onto the
clean pair gives an honest 2D plot.

The result: the 100 integers trace out a **circle**, with numbers
sharing the same `(a mod T)` landing on the same point. T=10 gives
the canonical "clock face" with the units digit. T=2 separates even
from odd. T=5 produces five clusters.

`fig3_circles_and_line.png` shows all four circles plus a bottom
strip — the projection onto the linear direction — which lays the
integers out as a number line.

### Step 6 — Visualise the 3D helix

Same idea but with three directions: `cos(2πa/10)`, `sin(2πa/10)`,
and the linear axis `a`. Orthonormalise them (same Gram-Schmidt
procedure as above, just on three vectors instead of two), project
`H` onto the resulting 3D frame, and plot.

You should see a helix that winds once per 10 integers around the
two circle axes and rises along the linear axis. This is the iconic
Figure 1 of the paper.

### The three saved figures

| file | paper figure | content |
|---|---|---|
| `fig2_fourier_pc1.png` | Fig 2 | FFT spectrum (top) + PC1-vs-`a` linear ramp (bottom) |
| `fig3_circles_and_line.png` | Fig 3 | four `(cos_T, sin_T)` panels + linear "number line" strip |
| `fig1_helix_T10.png` | Fig 1 (right) | the iconic 3D `T=10` helix |

## Requirements

- Python ≥ 3.10
- `uv` (handles the PEP 723 inline script metadata at the top of
  `main.py` automatically — no separate venv needed)
- A Hugging Face token in `.env` as `HF_TOKEN=…` (needed for gated
  models like Llama 3)
- Hardware: defaults to Apple MPS, falls back to CUDA, then CPU. With
  128 GB unified memory either Pythia-6.9B or Llama-3.1-8B fits in bf16.

## Run

```bash
# default: Pythia-6.9B, Latin digits, mean-pooled
uv run main.py

# pick a model
uv run main.py --model meta-llama/Llama-3.1-8B
uv run main.py --model google/gemma-4-E4B
uv run main.py --model google/gemma-4-31B

# pick a numeral script
uv run main.py --script arabic
uv run main.py --script chinese
uv run main.py --script roman

# pick a pooling mode (see "pooling" below)
uv run main.py --pool last       # paper's default for single-token Latin
uv run main.py --pool mean       # the default; honest for multi-token numerals
```

Or run the full sweep — `run.sh` covers four models × seven scripts:

```bash
./run.sh
```

The models in the sweep:

| model | params |
|---|---|
| `EleutherAI/pythia-6.9b` | 6.9 B |
| `meta-llama/Llama-3.1-8B` | 8 B |
| `google/gemma-4-E4B` | ~4 B (efficient) |
| `google/gemma-4-31B` | 31 B |

All are **base** (non-instruction-tuned) checkpoints, matching the paper's setup.

## Numeral scripts

| script | example (23) | family |
|---|---|---|
| `latin` (default) | `23` | positional base-10 |
| `arabic` | `٢٣` | positional base-10, Arabic-Indic |
| `persian` | `۲۳` | positional base-10, Persian |
| `devanagari` | `२३` | positional base-10, Hindi |
| `chinese` | `二三` | positional base-10, CJK glyphs |
| `greek` | `κγ` | additive (Milesian) |
| `roman` | `XXIII` | additive |

## Pooling (`--pool`)

For Pythia, Latin numbers 0–99 are single BPE tokens. Reading the
residual stream "at the operand's last position" is unambiguous. For
every other script (or for Latin on Llama 3, which tokenizes
digit-by-digit), a numeral spans several sub-tokens and the choice of
pool matters:

- **`--pool last`** reads the activation at the very last sub-token.
  For `٢٣` that's literally the second byte of `٣` — and `23, 33, 43, …`
  all end in `٣`, so they all read the same thing. This bakes a
  period-10 cycle into the data **by construction** and produces fake
  "helix-like" FFT peaks for reasons unrelated to anything the model
  learned.

- **`--pool mean`** (default) averages activations across all
  sub-tokens of the numeral. Every position contributes equally; the
  last-byte artifact disappears. This is the honest test for non-Latin
  scripts.

For single-token regimes (Latin on Pythia) `mean` and `last` produce
identical activations.

## Output layout

```
out/<model_with_slashes_replaced>/<script>/<pool>/
    ├── fig1_helix_T10.png
    ├── fig2_fourier_pc1.png
    └── fig3_circles_and_line.png
```

For example: `out/EleutherAI__pythia-6.9b/latin/mean/fig1_helix_T10.png`.

This nesting means re-running with different `--script` / `--pool` /
model values leaves prior outputs intact and lets you compare side by
side.

## Console output you should read

For each run the script prints:

```
helix fit R^2 (variance-weighted) = 0.46
  9-d PCA reconstruction R^2      = 0.55
```

The 9-D PCA R² is the **upper bound** on what any 9-parameter fit can
capture. If `helix R² / PCA R²` is close to 1 (above ~0.8) the helix
basis is essentially the 9-D structure of the residual stream — the
paper's main quantitative claim. For non-positional scripts this ratio
typically drops well below 0.7.

## Citation

```
@article{kantamneni2025helix,
  title  = {Language Models Use Trigonometry to Do Addition},
  author = {Kantamneni, Subhash and Tegmark, Max},
  journal= {arXiv preprint arXiv:2502.00873},
  year   = {2025}
}
```
