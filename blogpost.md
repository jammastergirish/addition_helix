# Where the number helix lives, and how to find it

An extension of Kantamneni & Tegmark,
[*Language Models Use Trigonometry to Do Addition*](https://arxiv.org/abs/2502.00873)
(2025), across four models and eight numeral systems. The paper
showed the helix in GPT-J-6B, Pythia-6.9B, and Llama-3.1-8B on Latin
digits 0–99 with a four-period trig basis — itself the LLM-scale
follow-up to [Nanda et al. (2023)](https://arxiv.org/abs/2301.05217),
who first reverse-engineered the same trig basis in a small
transformer trained from scratch on modular addition (the original
"grokking" task of [Power et al. (2022)](https://arxiv.org/abs/2201.02177),
where a small network trained on modular arithmetic was famously left
running long past memorisation and suddenly generalised). The
questions here are: does the helix extend to a new architecture
family (yes — Gemma 4); does it survive non-Latin scripts (only on
multilingual models); and does it show up in *Babylonian cuneiform* —
a positional base-60 system — once you use the right measurement
window and basis (yes, on all four models).

## The findings, up front

1. **The helix appears in a new architecture family (Gemma), and
   re-measurement on Pythia and Llama places it at radically
   different depths.** The paper already covered Pythia-6.9B,
   GPT-J-6B, and Llama-3.1-8B on Latin/0–99; we substitute the Gemma
   family (Gemma-4-E4B and Gemma-4-31B) for GPT-J in the panel and
   re-measure Pythia and Llama under a wider cross-script/cross-window
   protocol. All four converge on the same geometric encoding on
   Latin — but the *layer* where it peaks varies wildly (Gemma-4-31B's
   helix is a sharp spike at L29 of 61, with helix R² essentially
   flat everywhere else). The paper's "read the middle layer" recipe
   is a Pythia/GPT-J convention, not a universal one; for any new
   model, a layer sweep is mandatory.

2. **The helix generalises to other positional numeral systems, but
   only on models that were trained on them.** Devanagari `२३`,
   Arabic-Indic `٢٣`, positional Chinese `二三` — same idea, different
   glyphs. Multilingual Gemma encodes the *quantity* identically across
   scripts. English-mostly Pythia treats non-Latin positional scripts
   essentially the way it treats non-positional scripts.

3. **Yes, base 60 is present in Babylonian cuneiform on all four
   models.** This was the most surprising result. The paper's standard
   measurement protocol initially looked like it said "no" — but that
   was a measurement artifact in two distinct ways:

   - The 0..99 input range only lets base-60 wrap *once*. At 0..599
     (ten wraps), an FFT peak at T=60 appears clearly on every model.
   - The paper's trig basis `[T=2, 5, 10, 100]` is *blind* to T=60.
     Adding T=60 to the basis recovers **7–13 percentage points** of
     helix R² uniformly across all four models.

   Both fixes point the same direction. Babylonian numerals are
   represented as numbers, with their actual base, even by models that
   have presumably seen very little cuneiform during training.

4. **The helix lives at radically different depths in different
   models.** Pythia and Llama: mid-stack. Gemma-4-E4B: very early (L5
   of ~30). Gemma-4-31B: a sharp spike at L29 of 61, with helix R²
   essentially flat everywhere else. "Read the middle layer" is a
   Pythia-ism; a layer sweep is mandatory for everything else.

## What the helix actually is

For an integer `a ∈ [0, 99]`, the original paper found that the
residual stream of a transformer encodes `a` as a generalised helix —
a 3-D spiral that combines a linear "number-line" axis with circular
loops at four periods, `T ∈ {2, 5, 10, 100}`. Period 10 gives the
units digit (numbers ending in 7 stack vertically); period 100 gives
coarse magnitude; period 2 gives parity; period 5 captures `a mod 5`.

Here it is, in Pythia-6.9B at the layer where the helix is cleanest:

![3-D helix of integers 0..99 inside Pythia-6.9B](out/EleutherAI__pythia-6.9b/latin/mean/fig1_helix_T10.png)

Each dot is one integer; the spiral winds once per ten numbers (the
units-digit loop) while rising along the linear axis from 0 to 99.
Addition becomes rotating around the loops — the paper's "Clock"
algorithm.

## How we measure it

Two diagnostics. Both are basis-free; together they characterise
whatever periodic-plus-linear structure is in the residual stream.

![FFT and PC1 panel — Pythia-6.9B on Latin digits](out/EleutherAI__pythia-6.9b/latin/mean/fig2_fourier_pc1.png)

**Top panel: FFT across each hidden dimension, averaged.** Each of
4096 hidden dimensions is a function `h(a)` of the integer being
read. We FFT each one (treating `a` as time), then average the
magnitudes. Random noise washes out; periods many dimensions agree on
survive. A clean peak at frequency `1/T` means "many dimensions
oscillate with period `T`." For Pythia/Latin, peaks land cleanly at
T = 2, 5, 10, 100 — the paper's predicted periods, reproduced.

**Bottom panel: PC1 of the residual stream plotted against `a`.**
The first principal component is the direction of maximum variance;
plotting it against `a` tells you whether the model lays numbers out
on a number line. For Pythia/Latin, R² ≈ 0.95 — almost perfectly
linear in magnitude.

Together these two metrics describe the helix: circles via the FFT,
rise via PC1. A third diagnostic — fitting the trig basis as a
regression target — is **basis-dependent**. We'll come back to that;
it's where the methodology gets interesting.

## Finding 1 — A new architecture family forms the helix, and re-measurement spreads it across very different depths

The paper showed the helix in Pythia-6.9B, GPT-J-6B, and
Llama-3.1-8B — three decoder-only transformers in the same broad
lineage, all read at the middle layer on Latin/0–99. The natural
next question is whether a structurally different family (Gemma 4)
does the same thing, and where in the stack — and, while we're at
it, whether the Pythia/Llama helix survives a wider measurement
protocol (cross-script, wider windows, extended basis). They do,
and the depths are surprising:

| model | helix R² | helix/PCA ratio | peak layer | total layers |
|---|---|---|---|---|
| Pythia-6.9B | 0.53 | 0.85 | 15 | 32 |
| Llama-3.1-8B | 0.63 | 0.85 | 14 | 32 |
| Gemma-4-E4B | 0.50 | 0.77 | 5 | ~30 |
| Gemma-4-31B | **0.70** | 0.75 | 29 | 61 |

(`helix/PCA` is helix R² divided by the best possible R² from any
9-dimensional approximation of the residual stream. Above ~0.8 means
the trig basis is essentially the 9-D structure of the residual
stream, not just *a* 9-D approximation.)

But layer depth varies dramatically — and the shape of the depth
profile varies even more:

![Layer sweep on Latin, four models side-by-side](out/_compare/fig_latin_sweep_4models.png)

Look at the orange "helix R²" panel in each row.

- **Pythia and Llama** rise into the back half of the stack and stay
  high. Either of `num_layers // 2` or `num_layers - 1` works.
- **Gemma-4-E4B** has its strongest helix at L0–L5; it *decays* in
  later layers. The paper's "middle of the stack" recipe would
  literally look at decayed structure here.
- **Gemma-4-31B** is essentially flat at ~0.5 across the whole stack,
  with a **sharp spike** at L29 reaching 0.70 — then falls back.
  Reading at the default `num_layers // 2 = 30` lands one layer past
  the peak. When I first ran this model without a sweep, I concluded
  "Gemma doesn't form a helix." That was wrong by a single layer.

So the helix extends to Llama and Gemma — modest news on its own —
but the depth at which it lives doesn't generalise from the paper's
two models to these new ones. **The methodological contribution
here isn't "the helix is general"; it's "you have to sweep the
stack to find it, because middle-layer-reading is family-specific."**

## Finding 2 — The helix generalises across positional scripts, conditional on training data

The cross-script result is where I expected the cleanest story and got
the most interesting one. I rendered every integer in eight different
numeral systems and re-ran the analysis.

### A quick tour of the eight scripts

For readers who haven't met all of these before:

- **Arabic-Indic** (`٠١٢٣٤٥٦٧٨٩`) and **Persian** (`۰۱۲۳۴۵۶۷۸۹`) are
  positional base-10, identical in structure to Latin — different
  glyphs only. Persian is a visual variant of Arabic-Indic. Numbers
  are written most-significant-first despite Arabic script flowing
  right-to-left.
- **Devanagari** (`०१२३४५६७८९`) is also positional base-10, used
  across Indian languages — Hindi, Sanskrit, Marathi, and others.
- **Positional Chinese** uses the digit characters
  `零一二三四五六七八九` in the same place-value layout as Latin
  (`二三` = 23). Classical Chinese also admits a non-positional form
  (`二十三`, literally "two-tens-three"); we don't use that here.
- **Greek alphabetic (Milesian)** is additive: each letter carries a
  fixed value (α=1…θ=9, ι=10, κ=20, …π=80, ϟ=90, ρ=100…). 23 is
  `κγ` = 20+3. There is no shared "ones digit" between 13, 23, 33.
- **Roman** is additive with subtractive shortcuts at 4, 9, 40, 90,
  etc.: I=1, V=5, X=10, L=50, C=100. 23 = `XXIII`. Glyphs shift at
  4↔5, 9↔10, 49↔50, 89↔90 — those transitions show up later as
  staircase jumps in the PC1 panel.
- **Babylonian cuneiform** is positional at base 60, additive within
  each sexagesimal column. Two glyphs combine inside a column:
  `𒁹` (one) and `𒌋` (ten). 23 is `𒌋𒌋𒁹𒁹𒁹` — two tens plus
  three ones, all in one column. 60 opens a new column to the left.

With those in hand, the eight systems split into three families:

- **Positional base-10**: Latin `23`, Arabic-Indic `٢٣`, Persian `۲۳`,
  Devanagari `२३`, positional Chinese `二三`. Different glyphs, same
  underlying tens-and-ones structure.
- **Non-positional / additive**: Greek alphabetic `κγ` (κ=20 + γ=3),
  Roman `XXIII` (10+10+1+1+1). No tens column; you sum letter values.
- **Mixed**: Babylonian cuneiform — positional at base 60, additive
  within each column. We treat this separately in Finding 3.

Helix/PCA ratio at each model's peak layer:

| script | Pythia-6.9B | Llama-3.1-8B | Gemma-4-E4B | Gemma-4-31B |
|---|---|---|---|---|
| Latin | **0.85** | **0.85** | **0.77** | **0.75** |
| Arabic-Indic | 0.55 | 0.58 | 0.66 | 0.69 |
| Persian | 0.55 | 0.70 | 0.72 | 0.72 |
| Devanagari | 0.53 | 0.61 | 0.78 | **0.79** |
| Chinese | 0.60 | 0.58 | 0.68 | 0.66 |
| Greek (additive) | 0.54 | 0.57 | 0.57 | 0.57 |
| Roman (additive) | 0.54 | 0.55 | 0.62 | 0.64 |

Three things matter here.

**The non-positional collapse is uniform.** Greek and Roman drop the
helix/PCA ratio to ~0.55–0.65 on every model. This is the script-
structure prediction: a helix with a "tens loop" only buys you
anything if the numeral system has a tens column. If you're stacking
and summing letter values, there's nothing periodic to wrap.

**Latin is uniform too.** All four models clear 0.75. The shape
itself isn't model-specific.

**The non-Latin positional cells split sharply along models.** Pythia
on Devanagari (0.53) is essentially indistinguishable from Pythia on
Greek (0.54); the helix is gone. Llama on Devanagari (0.61) is a
little better. Gemma-4-31B on Devanagari hits **0.79**, matching its
own Latin score of 0.75. Pythia's training is heavily English; Llama
saw more multilingual text; Gemma is the most multilingual of the
four. The pattern is clear: **positional structure is necessary for
the helix, but training exposure to the specific script is what
makes it appear.** The geometric encoding is built per-script during
pre-training; the architecture alone is not enough.

This is a refinement of the original paper's framing. It is not the
case that the helix is simply "the geometry of integers"; it is the
geometry of integers *in scripts the model has actually seen*.

### What happens in the non-positional cases (it's not all the same kind of failure)

"Greek and Roman collapse the helix" is true but misleading on its own.
It suggests both scripts fail in the same way — they don't. The
helix/PCA ratio looks similar (~0.55), but the underlying geometry is
qualitatively different.

Comparing each model's PC1 R² (the magnitude axis) and helix R² (the
periodic structure) **and the layer where each one peaks**:

| model | script | PC1 R² peak | helix R² peak | layer gap |
|---|---|---|---|---|
| Pythia-6.9B | Latin | 0.96 @ L5 | 0.53 @ L32 | — |
| Pythia-6.9B | Roman | **0.66 @ L9** | 0.36 @ L18 | **9 layers** |
| Pythia-6.9B | Greek | 0.10 @ L9 | 0.28 @ **L0** | (artifact) |
| Llama-3.1-8B | Roman | 0.51 @ L16 | 0.36 @ L17 | 1 |
| Llama-3.1-8B | Greek | 0.17 @ L5 | 0.27 @ **L0** | (artifact) |
| Gemma-4-E4B | Roman | 0.54 @ L4 | 0.61 @ L18 | **14 layers** |
| Gemma-4-31B | Roman | 0.49 @ L11 | 0.64 @ L31 | **20 layers** |

**Roman: the helix splits across layers.** On Latin, PC1 and helix R²
both peak at the same mid-stack layer — magnitude and periodicity
co-locate, so the trig basis literally fits a helix (line × circles).
On Roman, the two peak at *different* layers, often 9–20 apart.
There's a magnitude axis (PC1 R² climbs to 0.49–0.66, so the model
does roughly order Roman numerals 0..99), and there's periodic
structure (T=10 and T=5 from the X- and V-cycles), but they live in
different parts of the residual stream:

![Pythia-6.9B Roman — number-line plus letter-cycling, in different layers](out/EleutherAI__pythia-6.9b/roman/mean/fig2_fourier_pc1.png)

The FFT (top) shows clean peaks at T=10 (X cycle) and T=5 (V cycle)
— Roman's actual visual periodicity. PC1 vs `a` (bottom) is *not*
linear, it's a **piecewise function with jumps at the Roman
threshold values** (4→5, 9→10, 49→50, 89→90). The fitted line has
R²=0.23 because the underlying shape isn't a line, it's a staircase
with discontinuities at the I/V, IX/X, XL/L, XC/C boundaries.

Geometrically: Roman is encoded as **a number-line module at one
layer plus a letter-cycling module at another**, not as a fused
helix. The model has decomposed the additive system into its parts
and stored them separately, where Latin's positional digits live
together as a single object.

**Greek: barely a representation at all.**

PC1 R² peaks at 0.10–0.21 across every model. There is no clean
magnitude axis anywhere in the stack — the model doesn't order Greek
numerals as quantities. And the helix R² that *does* appear peaks
at **layer 0 or near-0** on three of the four models. Layer 0 is the
token-embedding output, before any transformer block has run.

![Pythia-6.9B Greek — tokenisation artifact at the embedding layer](out/EleutherAI__pythia-6.9b/greek/mean/fig2_fourier_pc1.png)

Look at the magnitude scale on the FFT: peaks at ~0.20, versus
Latin's ~30. The T=10 peak that *is* there is just "the last Greek
letter cycles through α, β, γ, …, θ every ten numbers" — a literal
byte-level artifact of how the tokenizer encodes alternating two-
letter words. PC1 R² in the bottom panel is **0.004**: the model
isn't representing Greek numerals as quantities at all. They're
two-letter sequences with no numeric semantics, treated like
ordinary short words. Whatever "helix R²" the basis fit reports is
fitting noise plus tokenizer-induced periodicity at the input
layer.

This is a stronger statement than "the helix collapses." For Greek,
there is no number representation at any layer to collapse from.

**The three regimes, geometrically:**

| script type | example | layer-wise geometry |
|---|---|---|
| Positional, trained | Latin, Devanagari on Gemma | linear axis + period circles **co-located** → true helix |
| Additive, visually ordered | Roman | linear axis (early) + weak period structure (late), in **different layers** → decomposed parallel modules, not a helix |
| Additive, no visual order | Greek alphabetic | neither axis nor learned periodicity; FFT structure is a tokenizer artifact at L0 |

The two non-positional scripts collapse the helix for completely
different reasons — Roman because magnitude and periodicity get
stored separately rather than fused, Greek because the model never
formed a quantitative representation at all. The standard "helix R²"
metric averages over both modes and reports them as similar; reading
PC1 R² and the *layer at which it peaks* alongside helix R² is what
separates them.

A natural follow-up: FFT only sees periodic structure, but Roman's
actual encoding looks like a **piecewise linear staircase with
discontinuities at glyph thresholds**, and Greek's might be **9
clusters of 10, one per tens-letter**. Neither shows up as a Fourier
peak. Surfacing them needs different diagnostics — 2D PCA scatters
to find clusters, additive regression `h(a) ≈ Σ count(symbol) ·
u_symbol` to test the tree hypothesis.

### What 2D PCA actually shows (and the PC1-is-tokenization surprise)

We added a basis-free diagnostic to every cell: a 2D PCA scatter of
the residual stream at the helix-peak layer, colored by `a`, with
each numeral labelled and consecutive integers connected by a faint
trajectory line. Eight scripts × four models = 32 panels. Reading
them together turned up something none of the previous diagnostics
caught.

**Roman: cluster-by-prefix is visible directly.** The trajectory
forms distinct regions of representation space corresponding to the
Roman-numeral prefix classes (I/V, X, L, XC), with discontinuities
right at the threshold integers:

![Pythia-6.9B Roman — clusters by first-letter prefix](out/EleutherAI__pythia-6.9b/roman/mean/fig4_pca_2d.png)

The labels make the discontinuities concrete: numerals starting with
the same prefix cluster together; the trajectory line zigzags
between clusters as the prefix changes at 4→5, 9→10, 49→50, 89→90.
Roman's representation isn't a manifold; it's a partitioning of
representation space into prefix classes with crude ordering within
each class. The FFT couldn't see this — there's nothing periodic to
detect — but the 2D PCA shows it directly.

**Greek: confirmed noise.** Pythia/Greek hits PC1 + PC2 = 15.5% of
total variance, less than half of any other cell. The points scatter
diffusely with no recognisable shape. **There is no learned
representation to find.** Whatever periodic structure the FFT picked
up at layer 0 was a tokenizer artifact.

**The Gemma surprise: PC1 isn't magnitude — it's *token shape*.**

The most striking result is what 2D PCA shows for Gemma-4-31B. On
Latin, helix/PCA = 0.75 and helix R² = 0.70 — strong by any measure.
But PC1 doesn't pick up the magnitude axis. PC1 picks up the
**tokenization structure**:

![Gemma-4-31B Latin — PC1 is tokenization, not magnitude](out/google__gemma-4-31B/latin/mean/fig4_pca_2d.png)

PC1 captures **70.9% of variance** here — much more than Pythia's
20.4% — but it separates the points into three regions: single-digit
numerals (`0`–`9`, far left), two-digit teens (`10`–`19`, top middle),
and the rest (`20`–`99`, dense cluster on the right). The split is
along *digit count* and *leading-digit class*, not magnitude. The
helix lives in lower-variance directions (PC4, PC5, etc.) — present
in the activations but not the dominant variance.

Same pattern on Devanagari, more extreme:

![Gemma-4-31B Devanagari — PC1 = 94.6%, still tokenization](out/google__gemma-4-31B/devanagari/mean/fig4_pca_2d.png)

PC1 captures **94.6% of variance** and it just splits single-digit
(`०`–`९`, left blob) from two-digit (`१०`–`९९`, right blob). PC2 is
2.1%. From PC1 alone you'd think Gemma's Devanagari representation
is a binary token-count detector — but helix R² = 0.79 says the
helix is *also* there, in much lower-variance directions.

And on Roman it's even more dramatic — PC1 = 96.6% — splitting
roughly at the L/XL boundary (50 vs everything-else).

**What this tells us methodologically.** In Pythia, PC1 R² and the
magnitude-vs-`a` linearity correspond cleanly — looking at PC1 is
basically looking at the number-line. In Gemma, PC1 R² and
linearity-in-`a` *come apart*: PC1 captures massive variance but
that variance is overwhelmingly tokenization (digit count, leading
digit class). The number-line still exists; it lives further down
in the spectrum. **A high PC1 R² is not by itself evidence of a
clean magnitude axis** — it's evidence that something accounts for
most of the variance, and you have to look at *what* PC1 separates
to know whether that something is magnitude.

This also reframes the depth-variance finding: the reason Gemma's
helix-peak layer differs so wildly from Pythia's is partly that
Gemma's residual streams are more *token-shape-dominated* at any
given layer — the helix shares space with strong tokenization
features and has to be detected as a smaller-variance signal.

The full set of 32 panels (`out/<model>/<script>/mean/fig4_pca_2d.png`)
makes these regimes immediately legible at a glance and is the
single most informative output we produced.

## Finding 3 — Babylonian: base 60 is there, but you have to look properly

This is the main scientific surprise. It also doubles as a clean
case study in why the standard measurement protocol can mislead.

Babylonian cuneiform is positional at base 60, additive within each
column. The number 23 is `𒌋𒌋𒁹𒁹𒁹` — two tens-wedges plus three
ones-wedges, all in one column. The number 60 is `𒁹` (a single
ones-wedge in a new column to the left, separated by a space, with a
late-period placeholder for the empty units column: `𒁹 𒑊`). 599 is
`𒁹𒁹𒁹𒁹𒁹𒁹𒁹𒁹𒁹 𒌋𒌋𒌋𒌋𒌋𒁹𒁹𒁹𒁹𒁹𒁹𒁹𒁹𒁹`. The system has all the
ingredients of positional notation, but at base 60 instead of base 10.

### What 0..99 says

Running the standard experiment on Babylonian — same 0..99 range,
same `[T=2, 5, 10, 100]` basis as the paper — gives this:

| model | PC1 R² (magnitude axis) | helix R² | helix/PCA |
|---|---|---|---|
| Pythia-6.9B | 0.06 | 0.71 | 0.74 |
| Llama-3.1-8B | 0.10 | 0.65 | 0.69 |
| Gemma-4-E4B | 0.40 | 0.75 | 0.77 |
| Gemma-4-31B | 0.51 | 0.76 | 0.77 |

The natural reading: Pythia and Llama have the *circles* but no
*number-line* (PC1 R² ≈ 0); they're just counting wedges. Gemma has
both; it's representing magnitude.

Here is Gemma-4-31B in that picture:

![Gemma-4-31B Babylonian at n=100 — the misleading view](out/google__gemma-4-31B/babylonian/mean/fig2_fourier_pc1.png)

The FFT (top) labels peaks at T=10 and T=100; **no T=60 peak**. The
PC1 panel (bottom) shows two clean clusters with a hard step between
them: numbers 0–59 on one side of the magnitude axis, numbers 60–99
on the other. The 60-boundary is encoded as a *categorical state
change*, not as a rotation.

This is actually expected: in the 0..99 window, 60 only wraps
**once**. There is no periodic signal at T=60 for an FFT to detect.
What looks like "no base-60 structure" is in fact "structure that
can't yet be measured."

### What 0..599 says

At `n_max = 600` — ten full 60-wraparounds — the picture changes:

| model | PC1 R² (100 → 600) | helix R² (100 → 600) | helix/PCA (100 → 600) |
|---|---|---|---|
| Pythia-6.9B | 0.06 → **0.55** | 0.71 → 0.66 | 0.74 → 0.69 |
| Llama-3.1-8B | 0.10 → **0.55** | 0.65 → 0.52 | 0.69 → 0.56 |
| Gemma-4-E4B | 0.40 → **0.69** | 0.75 → 0.61 | 0.77 → 0.64 |
| Gemma-4-31B | 0.51 → **0.64** | 0.76 → 0.50 | 0.77 → 0.52 |

**Every model now has a real magnitude axis** (PC1 R² ≈ 0.55–0.69).
Pythia and Llama's earlier "no number-line" was a measurement
artifact of the narrow window. They were representing Babylonian as
numbers all along; we just couldn't see it.

The helix R² *drops* at n=600, in the opposite direction to PC1.
That's the second measurement issue, and the bigger one.

Here's Pythia at n=600 (paper basis):

![Pythia-6.9B Babylonian at n=600 — T=60 emerges](out/EleutherAI__pythia-6.9b/babylonian/mean_n600/fig2_fourier_pc1.png)

A **clean, labelled peak at T ≈ 60** appears in the FFT, alongside
the T=10 sub-base and various harmonics. Gemma-4-E4B's FFT looks
similar: T=60 is the leftmost prominent peak.

For Llama and Gemma-4-31B, base-60 structure is still there, but
shows up via its *second harmonic* at T=30 (frequency 2/60 = 1/30):

![Gemma-4-31B Babylonian at n=600 — T=30 harmonic](out/google__gemma-4-31B/babylonian/mean_n600/fig2_fourier_pc1.png)

A pure sinusoidal period-60 signal would have only the fundamental;
strong even harmonics in the FFT suggest a more sawtooth-like
underlying signal (consistent with the "step at 60, reset within
60" structure of cuneiform's columns). Different models encode the
same underlying period in slightly different shapes.

The PC1 panels at n=600 also show the magnitude axis cleanly with
visible per-column sawtooth — every 60 numbers, the model "resets"
into a new column and the linear coordinate jumps.

So the base-60 structure is **present** in the FFT — but the helix
R² *drops* because the paper basis `[T=2, 5, 10, 100]` doesn't
include T=60. The regression has nothing to fit the period-60 signal
against, so it can't capture it as helix variance.

### What an extended basis says

The natural fix: add T=60 to the basis. I re-ran the four n=600
Babylonian sweeps with `periods = [2, 5, 10, 60, 100]` — same paper
basis, plus one extra trig pair (cos and sin of `2πa/60`) — and
compared:

| model | helix R² (paper) → (extended) | Δ | helix/PCA (paper) → (ext) | Δ |
|---|---|---|---|---|
| Pythia-6.9B | 0.661 → **0.747** | **+0.086** | 0.687 → 0.772 | +0.085 |
| Llama-3.1-8B | 0.515 → **0.622** | **+0.107** | 0.555 → 0.645 | +0.090 |
| Gemma-4-E4B | 0.611 → **0.698** | **+0.087** | 0.637 → 0.699 | +0.062 |
| Gemma-4-31B | 0.501 → **0.634** | **+0.133** | 0.523 → 0.636 | +0.113 |

Adding the single missing period recovers **7–13 percentage points
of variance** on every model. Same direction, same order of magnitude,
on four architectures.

A specific pattern: Llama-3.1-8B and Gemma-4-31B benefit the most
(+0.107 and +0.133). Those are precisely the two models whose FFT at
n=600 showed T=30 (the 2nd harmonic) more prominently than T=60
itself. Giving the regression the fundamental anchors its phase;
once T=60 is in the basis, the model's signal lands on it cleanly
even when the spectrum's strongest peak was harmonic.

Sanity check: PC1 R² across the layer sweep is unchanged by the
basis change (it doesn't depend on the trig basis at all). Good —
confirms we didn't accidentally change anything else.

The conclusion is unambiguous:

> **Base 60 is present in Babylonian representations on all four
> models, with the structure increasing in strength roughly as
> Pythia < Gemma-4-E4B < Llama < Gemma-4-31B (by Δ helix R²).**

The narrow-window protocol said "no" because it couldn't see it.
The paper-basis fit said "small" because it wasn't asking. Both
measurement adjustments point the same direction, which is the
strongest evidence we have that the structure is real.

## Method matters: two general lessons

Both of these go beyond this paper.

### M1. The basis has bandwidth

Helix R² is a fitted regression onto a fixed set of periods. It's
sensitive only to structure at those periods. If a model encodes
something at a period the basis doesn't include, helix R² will be
low *even if* the underlying structure is strong. The Babylonian
result shows this concretely: same data, same model, +0.13 helix R²
just from adding one period to the basis.

The fix: **always cross-check a basis-fit metric against a basis-free
one.** In our case that means reading the FFT panel: if a peak
appears at a period not in the basis, the helix R² is undermeasuring
the structure. Then add that period to the basis and refit. The
extended-basis number is the honest one.

### M2. The window must wrap

To detect a period `T` via FFT, the input range needs to be at least
several wraps of `T`. One wrap is a single discontinuity; ten wraps
is a clean peak. The standard 0..99 window is fine for T = 2, 5, 10
(50, 20, and 10 wraps respectively) and just barely workable for
T = 100 (one wrap, visible as a slow trend). It is *not* workable
for T = 60 (also one wrap, but in a phase-misaligned way that
produces noise rather than a trend). The fix is to widen the window
to many wraps of the candidate periods.

The general rule: **n_max must be a multiple of all the periods you
might want to detect, ideally large.** For Latin alone, 100 is fine;
for any extension hunting unfamiliar periods, go wider.

## So what's the ultimate finding?

Putting the threads together:

- Transformers really do form a geometric representation of numbers
  in their residual streams. The paper showed this on two models
  (Pythia and GPT-J); two more architecture families (Llama, Gemma)
  converge on the same shape on Latin digits, though at very
  different layers. The recipe for *finding* the helix has to be
  adapted per family; the helix itself is consistent.

- That representation transfers across glyph systems for any
  positional base-10 script the model has been trained on. The
  same geometric encoding lives at Devanagari `२३` and Latin `23`
  in multilingual models. The encoding is built per-script during
  pre-training; positional structure alone is necessary but not
  sufficient.

- It also reaches further than expected: into Babylonian cuneiform,
  with base-60 structure visible in the FFT and recoverable in the
  regression once you supply the right basis and range. None of the
  four models have likely seen much cuneiform during training, yet
  they all encode it as numbers with its actual base. This is a
  stronger generalisation claim than the paper makes, and it lands
  cleanly across four architectures.

- The standard measurement protocol — fixed trig basis, fixed 0..99
  window — has two known blind spots that can produce false
  negatives. Both can mislead in the same direction simultaneously
  (as they did for us on Babylonian at first). The fixes are simple
  but they have to be applied: widen the range to many wraps,
  cross-check the basis fit against the basis-free FFT, and add
  missing periods to the basis if the FFT shows them.

What this experiment changes in my own mental model of LLMs is the
*generality* of the helix. The paper showed it on one model and one
numeral system. The natural assumption — and the one I started with
— was that it was a property of that model's training. What seems
truer now is that **the helix is a property of how transformers
encode anything they represent as a quantity**, parameterised by
which scripts they have seen and which periods are natural to those
scripts. Different models pick up different periods at different
depths, but the geometric strategy is the same.

The causal question — *whether the helix subspace is in fact the
representation downstream computation reads* — is not what this post
sets out to answer. That is the natural follow-up, and it requires
careful intervention at the position the model reads to produce the
answer (the `=` position), not at the operand position. We leave it
to a follow-up.

## Reproducing

```bash
# Default: Pythia-6.9B, Latin digits, paper basis [2,5,10,100], n=100
uv run main.py

# Cross-script
uv run main.py --model meta-llama/Llama-3.1-8B --script devanagari
uv run main.py --model google/gemma-4-31B --script babylonian --sweep

# Wider window
uv run main.py --script babylonian --n_max 600 --sweep

# Extended basis with T=60
uv run main.py --script babylonian --n_max 600 --periods 2,5,10,60,100 --sweep

# Or the full sweep (4 models × 8 scripts × 2 windows × 2 bases = 40 combos,
# idempotent — each combo is skipped if its fig_layer_sweep.png exists)
./run.sh
```

All of this lives in
[github.com/...](https://github.com/...). Hardware: tested on an
Apple M5 Max with 128 GB unified memory (MPS). Falls back to CUDA,
then CPU. Total wall-time for the full sweep: ~6–8 hours.

## Citation

```
@article{kantamneni2025helix,
  title  = {Language Models Use Trigonometry to Do Addition},
  author = {Kantamneni, Subhash and Tegmark, Max},
  journal= {arXiv preprint arXiv:2502.00873},
  year   = {2025}
}

@inproceedings{nanda2023progress,
  title     = {Progress Measures for Grokking via Mechanistic Interpretability},
  author    = {Nanda, Neel and Chan, Lawrence and Lieberum, Tom and Smith, Jess and Steinhardt, Jacob},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023},
  eprint    = {2301.05217}
}

@article{power2022grokking,
  title  = {Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets},
  author = {Power, Alethea and Burda, Yuri and Edwards, Harri and Babuschkin, Igor and Misra, Vedant},
  journal= {arXiv preprint arXiv:2201.02177},
  year   = {2022}
}
```
