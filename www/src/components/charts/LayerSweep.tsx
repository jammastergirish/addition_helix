import { useMemo, useState } from "react";
import * as d3 from "d3";
import type { LayerSweepDoc } from "../../lib/types";
import { fmtPct } from "../../lib/svg";
import { MODEL_COLOR, MODEL_ORDER } from "../../lib/data";

// Fallback palette indexed by position, used when an overlay isn't keyed by a
// known HF model id. Mirrors MODEL_COLOR in MODEL_ORDER order.
const PALETTE = MODEL_ORDER.map((m) => MODEL_COLOR[m]);
function defaultColor(i: number): string {
  return PALETTE[i % PALETTE.length];
}

interface Props {
  /** One or more layer-sweep docs to overlay. */
  docs: { label: string; doc: LayerSweepDoc; color?: string }[];
  /** Which metric to plot. */
  metric?: "pc1_r2" | "helix_r2" | "helix_over_pca";
  height?: number;
}

const METRIC_LABEL = {
  pc1_r2:         { title: "PC1 R²",                  hint: "linear magnitude 'spine'", c: "#0369a1" },
  helix_r2:       { title: "helix R²",                hint: "trig basis fit",           c: "#c2410c" },
  helix_over_pca: { title: "helix / K-d PCA",         hint: "subspace dominance",       c: "#15803d" },
};

export function LayerSweep({ docs, metric = "helix_r2", height = 280 }: Props) {
  const [hover, setHover] = useState<{ layer: number; x: number; y: number } | null>(null);

  const meta = METRIC_LABEL[metric];
  // X axis: use the maximum layer count across the overlaid models so the
  // shape comparison stays honest (depth varies wildly across families).
  const maxLayers = useMemo(
    () => Math.max(...docs.map((d) => d.doc.layers.length)) - 1,
    [docs],
  );

  const W = 720, H = height;
  const margin = { top: 14, right: 18, bottom: 36, left: 44 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;

  const x = d3.scaleLinear().domain([0, maxLayers]).range([0, innerW]);
  const y = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);
  const line = d3.line<{ L: number; v: number }>()
    .x((d) => x(d.L))
    .y((d) => y(d.v))
    .curve(d3.curveMonotoneX);

  const yTicks = [0, 0.25, 0.5, 0.75, 1];
  const xTicks = x.ticks(Math.min(8, maxLayers));

  return (
    <div className="w-full">
      <div className="mb-2 flex items-baseline justify-between">
        <div>
          <div className="text-sm font-semibold">{meta.title}</div>
          <div className="text-xs text-ink-mute">{meta.hint}</div>
        </div>
        <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-ink-mute">
          {docs.map((d, i) => (
            <span key={d.label} className="inline-flex items-center gap-1.5">
              <span className="inline-block h-[3px] w-4 rounded" style={{ background: d.color ?? defaultColor(i) }} />
              {d.label}
            </span>
          ))}
        </div>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" onMouseLeave={() => setHover(null)}>
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          {yTicks.map((t) => (
            <g key={t} transform={`translate(0, ${y(t)})`}>
              <line x1={0} x2={innerW} stroke="#e5e7eb" strokeWidth={1} />
              <text x={-8} y={3} fontSize={10} textAnchor="end" fill="#6b7280">{t}</text>
            </g>
          ))}
          {xTicks.map((t) => (
            <g key={t} transform={`translate(${x(t)}, ${innerH})`}>
              <line y1={0} y2={4} stroke="#9ca3af" />
              <text y={16} fontSize={10} textAnchor="middle" fill="#6b7280">{t}</text>
            </g>
          ))}
          <text
            transform={`translate(${innerW / 2}, ${innerH + 30})`}
            textAnchor="middle"
            fontSize={11}
            fill="#374151"
          >
            layer index L
          </text>

          {docs.map((d, i) => {
            const arr = d.doc[metric];
            const pts = arr.map((v, L) => ({ L, v }));
            const c = d.color ?? defaultColor(i);
            const peakL = d.doc.peaks[metric].layer;
            const peakV = d.doc.peaks[metric].value;
            return (
              <g key={d.label}>
                <path d={line(pts) ?? ""} fill="none" stroke={c} strokeWidth={1.8} opacity={0.9} />
                <circle cx={x(peakL)} cy={y(peakV)} r={3.5} fill={c} stroke="white" strokeWidth={1.5} />
              </g>
            );
          })}

          {/* Mouse capture */}
          <rect
            x={0} y={0} width={innerW} height={innerH}
            fill="transparent"
            onMouseMove={(e) => {
              const rect = (e.target as SVGRectElement).getBoundingClientRect();
              const px = e.clientX - rect.left;
              const L = Math.round(x.invert(px));
              if (L >= 0 && L <= maxLayers) setHover({ layer: L, x: x(L), y: 0 });
            }}
          />

          {hover && (
            <g transform={`translate(${hover.x}, 0)`}>
              <line y1={0} y2={innerH} stroke="#94a3b8" strokeWidth={1} strokeDasharray="3 3" />
            </g>
          )}
        </g>
      </svg>

      {hover && (
        <div className="mt-1 grid grid-cols-2 gap-x-6 text-xs text-ink-mute">
          <div><strong className="text-ink/80">layer {hover.layer}</strong></div>
          <div>
            {docs.map((d, i) => {
              const v = d.doc[metric][hover.layer];
              return v == null ? null : (
                <span key={d.label} className="mr-3 inline-flex items-center gap-1">
                  <span className="inline-block h-2 w-2 rounded-full" style={{ background: d.color ?? defaultColor(i) }} />
                  {d.label}: <strong className="text-ink/80 ml-0.5">{fmtPct(v)}</strong>
                </span>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

