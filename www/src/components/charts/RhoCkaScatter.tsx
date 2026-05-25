import { useEffect, useState } from "react";
import * as d3 from "d3";
import { loadJson } from "../../lib/data";
import { MODEL_ORDER, MODEL_LABEL, SCRIPT_LABEL } from "../../lib/data";

interface Cell {
  model: string;
  script: string;
  peak_layer: number;
  rho: number | null;
  cka_helix: number | null;
  trivial?: boolean;
  error?: string;
}

interface Doc {
  kind: "subspace_alignment_cka";
  results: Cell[];
}

const MODEL_COLOR: Record<string, string> = {
  "EleutherAI/pythia-6.9b":    "#0369a1",
  "EleutherAI/gpt-j-6b":       "#4338ca",
  "meta-llama/Llama-3.1-8B":   "#c2410c",
  "google/gemma-4-E4B":        "#15803d",
  "google/gemma-4-31B":        "#7c3aed",
  "allenai/Olmo-3-1125-32B":   "#be185d",
  "Qwen/Qwen2.5-7B":           "#0891b2",
  "Qwen/Qwen2.5-32B":          "#a16207",
};

/**
 * 2D scatter of (ρ, CKA-helix) for every (model, script) cell. Four
 * quadrants give the four mechanistic regimes:
 *
 *    high CKA |  amplifies (LL)         passes through (LR)
 *             |  depth refines L=0      depth doesn't touch L=0
 *    ---------+-----------------------------------------
 *    low CKA  |  builds new helix (UL)  REBUILDS w/ same score (UR)
 *             |  rare                   "high ρ" that hides real depth work
 *             +-----------------------------------------
 *                low ρ                     high ρ
 *               (depth raises score)       (score unchanged)
 *
 * Cells with peak_layer = 0 are trivial (CKA(L=0,L=0)=1) and shown faded.
 */
export function RhoCkaScatter({ height = 480 }: { height?: number }) {
  const [doc, setDoc] = useState<Doc | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState<Cell | null>(null);

  useEffect(() => {
    loadJson<Doc>("_subspace_alignment.json")
      .then(setDoc)
      .catch((e) => setError(String(e)));
  }, []);

  if (error) {
    return (
      <div className="flex h-72 items-center justify-center text-sm text-ink-mute">
        <div className="text-center">
          <div className="font-medium text-ink/70">CKA data not loaded</div>
          <div className="mt-1 text-xs">Run <code>uv run subspace_align.py --scripts all</code>.</div>
        </div>
      </div>
    );
  }
  if (!doc) {
    return <div className="flex h-72 items-center justify-center text-sm text-ink-mute">loading…</div>;
  }

  const cells = doc.results.filter((r) => !r.error && r.rho != null && r.cka_helix != null);

  const W = 760, H = height;
  const margin = { top: 18, right: 20, bottom: 44, left: 56 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;

  const x = d3.scaleLinear().domain([0, 1.02]).range([0, innerW]);
  const y = d3.scaleLinear().domain([0, 1.02]).range([innerH, 0]);

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" onMouseLeave={() => setHover(null)}>
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          {/* Quadrant background tints */}
          <rect x={x(0)} y={y(1.02)} width={x(0.55) - x(0)} height={y(0.6) - y(1.02)} fill="#ecfdf5" opacity={0.7} />
          <rect x={x(0.55)} y={y(1.02)} width={x(1.02) - x(0.55)} height={y(0.6) - y(1.02)} fill="#fff7ed" opacity={0.6} />
          <rect x={x(0)} y={y(0.6)} width={x(0.55) - x(0)} height={y(0) - y(0.6)} fill="#f8fafc" opacity={0.7} />
          <rect x={x(0.55)} y={y(0.6)} width={x(1.02) - x(0.55)} height={y(0) - y(0.6)} fill="#fef2f2" opacity={0.7} />

          {/* Quadrant labels */}
          <text x={x(0.05)} y={y(0.97)} fontSize={11} fill="#047857" fontWeight={600}>depth amplifies</text>
          <text x={x(0.05)} y={y(0.92)} fontSize={9} fill="#065f46">low ρ · high CKA · depth refines L=0</text>

          <text x={x(0.95)} y={y(0.97)} fontSize={11} fill="#c2410c" fontWeight={600} textAnchor="end">passes through</text>
          <text x={x(0.95)} y={y(0.92)} fontSize={9} fill="#7c2d12" textAnchor="end">high ρ · high CKA · depth doesn't touch L=0</text>

          <text x={x(0.05)} y={y(0.04)} fontSize={11} fill="#64748b" fontWeight={600}>depth builds new</text>
          <text x={x(0.05)} y={y(0.09)} fontSize={9} fill="#475569" transform={`translate(0, -12)`}>low ρ · low CKA · rare in practice</text>

          <text x={x(0.95)} y={y(0.04)} fontSize={11} fill="#b91c1c" fontWeight={600} textAnchor="end">rebuilt geometry, same score</text>
          <text x={x(0.95)} y={y(0.09)} fontSize={9} fill="#7f1d1d" textAnchor="end" transform={`translate(0, -12)`}>high ρ · low CKA · "inherited" mislabel</text>

          {/* Quadrant divider lines */}
          <line x1={x(0.55)} x2={x(0.55)} y1={y(0)} y2={y(1.02)} stroke="#475569" strokeDasharray="3 3" opacity={0.4} />
          <line x1={x(0)} x2={x(1.02)} y1={y(0.6)} y2={y(0.6)} stroke="#475569" strokeDasharray="3 3" opacity={0.4} />

          {/* Axes */}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <g key={`x${t}`}>
              <line x1={x(t)} x2={x(t)} y1={y(0)} y2={y(0) + 4} stroke="#94a3b8" />
              <text x={x(t)} y={y(0) + 16} fontSize={10} fill="#64748b" textAnchor="middle">{t.toFixed(2)}</text>
            </g>
          ))}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <g key={`y${t}`}>
              <line x1={-4} x2={0} y1={y(t)} y2={y(t)} stroke="#94a3b8" />
              <text x={-8} y={y(t) + 3} fontSize={10} fill="#64748b" textAnchor="end">{t.toFixed(2)}</text>
            </g>
          ))}
          <text x={innerW / 2} y={innerH + 32} fontSize={11} fill="#374151" textAnchor="middle">
            ρ = R²_helix(L=0) / R²_helix(peak)   →
          </text>
          <text transform={`translate(${-40}, ${innerH / 2}) rotate(-90)`} fontSize={11} fill="#374151" textAnchor="middle">
            CKA(B·W_L=0, B·W_peak)   →
          </text>

          {/* Points */}
          {cells.map((c, i) => {
            const cx = x(c.rho!);
            const cy = y(c.cka_helix!);
            const color = MODEL_COLOR[c.model] ?? "#999";
            const trivial = !!c.trivial || c.peak_layer === 0;
            return (
              <g key={i}>
                <circle
                  cx={cx} cy={cy} r={5}
                  fill={color}
                  fillOpacity={trivial ? 0.25 : 0.78}
                  stroke="white" strokeWidth={1}
                  onMouseEnter={() => setHover(c)}
                  style={{ cursor: "pointer" }}
                />
              </g>
            );
          })}

          {/* Hover crosshair */}
          {hover && hover.rho != null && hover.cka_helix != null && (
            <g pointerEvents="none">
              <line x1={x(hover.rho)} x2={x(hover.rho)} y1={y(0)} y2={y(1.02)} stroke="#1f2937" strokeDasharray="2 2" opacity={0.4} />
              <line x1={x(0)} x2={x(1.02)} y1={y(hover.cka_helix)} y2={y(hover.cka_helix)} stroke="#1f2937" strokeDasharray="2 2" opacity={0.4} />
              <circle cx={x(hover.rho)} cy={y(hover.cka_helix)} r={7} fill="none" stroke="#1f2937" strokeWidth={1.5} />
            </g>
          )}
        </g>
      </svg>

      {/* Legend + hover info */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-xs text-ink-mute">
        <div className="flex flex-wrap gap-x-3 gap-y-1">
          {MODEL_ORDER.map((m) => (
            <span key={m} className="inline-flex items-center gap-1.5">
              <span className="inline-block h-3 w-3 rounded-full" style={{ background: MODEL_COLOR[m] }} />
              {MODEL_LABEL[m]}
            </span>
          ))}
        </div>
        <span className="text-[11px]">faded = peak at L=0 (trivial CKA)</span>
      </div>

      {hover && (
        <div className="mt-2 rounded border border-ink/15 bg-white px-3 py-2 text-xs text-ink">
          <strong>{MODEL_LABEL[hover.model]}</strong> / {SCRIPT_LABEL[hover.script]} ·
          peak L{hover.peak_layer} ·
          ρ = {hover.rho?.toFixed(2)} ·
          CKA(helix) = {hover.cka_helix?.toFixed(2)}
        </div>
      )}
    </div>
  );
}
