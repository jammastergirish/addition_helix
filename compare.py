# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10"]
# ///
"""Stack the per-model Latin layer-sweep PNGs into one comparison image.

One row per model, with a labelled header strip above each row showing
the model id and its total transformer-layer count. Layer counts are
read from each model's `fig_layer_sweep.json` (written by main.py) so
they stay in sync with what the sweep actually saw — no stale hardcoded
numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "out" / "_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (output dir, display name) — layer count is appended at runtime.
MODELS = [
    ("EleutherAI__pythia-6.9b",    "Pythia-6.9B"),
    ("EleutherAI__gpt-j-6b",       "GPT-J-6B"),
    ("meta-llama__Llama-3.1-8B",   "Llama-3.1-8B"),
    ("google__gemma-4-E4B",        "Gemma-4-E4B"),
    ("google__gemma-4-31B",        "Gemma-4-31B"),
    ("allenai__Olmo-3-1125-32B",   "OLMo-3-32B"),
    ("Qwen__Qwen2.5-7B",           "Qwen2.5-7B"),
    ("Qwen__Qwen2.5-32B",          "Qwen2.5-32B"),
]


def label_for(model_dir: str, base_name: str) -> str:
    """Read n_layers out of the model's Latin layer-sweep JSON and tack
    it onto the display name. Falls back to the bare name if the JSON
    isn't present yet."""
    sweep_json = ROOT / "out" / model_dir / "latin" / "mean" / "fig_layer_sweep.json"
    if not sweep_json.exists():
        return base_name
    try:
        meta = json.loads(sweep_json.read_text())["meta"]
        return f"{base_name}  ({meta['n_layers']} layers)"
    except (KeyError, json.JSONDecodeError):
        return base_name

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
    labels = []
    for model_dir, base_name in MODELS:
        p = ROOT / "out" / model_dir / "latin" / "mean" / "fig_layer_sweep.png"
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
        panels.append(Image.open(p))
        labels.append(label_for(model_dir, base_name))

    w = max(im.width for im in panels)
    row_h = HEADER_H + max(im.height for im in panels) + PAD
    total_h = PAD + row_h * len(panels)

    canvas = Image.new("RGB", (w + 2 * PAD, total_h), "white")
    font = load_font(34)
    draw = ImageDraw.Draw(canvas)

    y = PAD
    for im, label in zip(panels, labels):
        # header strip
        draw.rectangle(
            [PAD, y, PAD + w, y + HEADER_H],
            fill=(240, 240, 245),
        )
        draw.text((PAD + 24, y + 16), label, fill="black", font=font)
        # paste the sweep image directly under the header
        canvas.paste(im, (PAD + (w - im.width) // 2, y + HEADER_H))
        y += row_h

    out_path = OUT_DIR / f"fig_latin_sweep_{len(MODELS)}models.png"
    canvas.save(out_path, "PNG", optimize=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
