# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10"]
# ///
"""Stack the four Latin layer-sweep PNGs into one comparison image.

One row per model, with a labelled header strip above each row showing
the model id and its total transformer-layer count. The four existing
fig_layer_sweep.png files (PC1 R², helix R², helix/PCA, vs layer) are
pasted unchanged so the x-axes still reflect each model's true layer
count -- which is the depth-variance story we want to see.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "out" / "_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("EleutherAI__pythia-6.9b",   "Pythia-6.9B  (32 layers)"),
    ("meta-llama__Llama-3.1-8B",  "Llama-3.1-8B  (32 layers)"),
    ("google__gemma-4-E4B",       "Gemma-4-E4B  (~30 layers)"),
    ("google__gemma-4-31B",       "Gemma-4-31B  (61 layers)"),
]

HEADER_H = 70
PAD = 12


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    panels = []
    for model_dir, _label in MODELS:
        p = ROOT / "out" / model_dir / "latin" / "mean" / "fig_layer_sweep.png"
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
        panels.append(Image.open(p))

    w = max(im.width for im in panels)
    row_h = HEADER_H + max(im.height for im in panels) + PAD
    total_h = PAD + row_h * len(panels)

    canvas = Image.new("RGB", (w + 2 * PAD, total_h), "white")
    font = load_font(34)
    draw = ImageDraw.Draw(canvas)

    y = PAD
    for (model_dir, label), im in zip(MODELS, panels):
        # header strip
        draw.rectangle(
            [PAD, y, PAD + w, y + HEADER_H],
            fill=(240, 240, 245),
        )
        draw.text((PAD + 24, y + 16), label, fill="black", font=font)
        # paste the sweep image directly under the header
        canvas.paste(im, (PAD + (w - im.width) // 2, y + HEADER_H))
        y += row_h

    out_path = OUT_DIR / "fig_latin_sweep_4models.png"
    canvas.save(out_path, "PNG", optimize=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
