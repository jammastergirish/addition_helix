# www/ -- the blogpost as a site

React + Vite + Tailwind. Every chart loads from `../out/*.json` produced by
`main.py` (PNG siblings are kept for the LaTeX paper).

## Dev

```bash
cd www
npm install
npm run dev   # http://localhost:5173 ; serves ../out at /data/
```

If `_index.json` is missing the site renders with "data not loaded"
placeholders -- run `uv run aggregate.py` at the repo root first.

## Build (static)

```bash
npm run build               # -> www/dist
# www/dist/data/ is empty -- copy the JSONs in before deploying:
rsync -av --include='*/' --include='*.json' --exclude='*' ../out/ dist/data/
```

Deploy `dist/` to any static host.

## How charts get their data

1. `main.py` writes `<fig>.json` next to every `<fig>.png` (one per
   `out/<model>/<script>/<pool>/`).
2. `aggregate.py` walks those and writes `out/_index.json`
   (the master cell index the site reads first) and `out/_l0_share.csv`.
3. The Vite dev server serves `../out/` at `/data/`.
4. React loads `/data/_index.json`, finds the right cell for each section,
   and lazy-loads its per-figure JSON.

## Component map

- `lib/types.ts`   -- JSON schema mirrors. Keep in sync with `main.py`'s `dump_json` calls.
- `lib/data.ts`    -- `loadJson`, `findCell`, model/script orderings + labels.
- `lib/svg.ts`     -- shared d3 helpers + the rho heatmap colour scale.
- `ChartFrame.tsx` -- lazy-load wrapper with graceful "not available" fallback.
- `charts/`        -- one component per figure type (FFT, PC1, circles, 3D helix, PCA 2D, layer sweep, rho heatmap).
- `sections/`      -- one component per blogpost section. Hero -> Findings -> WhatTheHelixIs -> HowWeMeasure -> Finding1..4 -> Methods -> Conclusion -> Reproducing.
