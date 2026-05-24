import { useMemo } from "react";
import type { IndexDoc } from "../../lib/types";
import {
  MODEL_ORDER, MODEL_LABEL,
  SCRIPT_ORDER, SCRIPT_LABEL,
} from "../../lib/data";

interface Props { index: IndexDoc | null; }

type Cls = "depth-built" | "depth-amplified" | "inherited" | "weak";

const CLS_COLOR: Record<Cls, string> = {
  "depth-built":     "#0f766e", // teal-700  — depth builds the geometry
  "depth-amplified": "#5eead4", // teal-300  — partial / amplification
  "inherited":       "#fb923c", // orange-400 — embedding-inherited
  "weak":            "#e5e7eb", // gray-200   — basis fits little
};

const CLS_LABEL: Record<Cls, string> = {
  "depth-built":     "depth-built",
  "depth-amplified": "depth-amplified",
  "inherited":       "inherited",
  "weak":            "weak",
};

/**
 * Classifies each (model, script) cell at paper defaults (n=100,
 * basis [2,5,10,100]) using joint ρ + helix/PCA thresholds:
 *
 *   weak              helix/PCA < 0.55                (basis fits little)
 *   depth-built       ρ ≤ 0.55  AND helix/PCA ≥ 0.70
 *   depth-amplified   0.55 < ρ ≤ 0.70  AND helix/PCA ≥ 0.60
 *   inherited         everything else (≈ high ρ)
 *
 * Companion to the ρ heatmap: ρ alone hides the quality dimension —
 * a low ρ in a cell where helix R² is barely above noise doesn't
 * indicate a depth-built helix, just that the basis fits nothing.
 */
function classify(rho: number | null, qual: number | null): Cls {
  if (rho == null || qual == null) return "weak";
  if (qual < 0.55) return "weak";
  if (rho <= 0.55 && qual >= 0.70) return "depth-built";
  if (rho <= 0.70 && qual >= 0.60) return "depth-amplified";
  return "inherited";
}

export function ClassificationGrid({ index }: Props) {
  const grid = useMemo(() => {
    const m = new Map<string, { cls: Cls; rho: number | null; qual: number | null }>();
    if (!index) return m;
    for (const c of index.cells) {
      if (c.n_max !== 100) continue;
      if (c.periods.length !== 4 || c.periods.some((p, i) => p !== [2,5,10,100][i])) continue;
      m.set(`${c.model}|${c.script}`, {
        cls:  classify(c.rho, c.helix_over_pca_peak),
        rho:  c.rho,
        qual: c.helix_over_pca_peak,
      });
    }
    return m;
  }, [index]);

  // Summary counts for the legend
  const counts = useMemo(() => {
    const c: Record<Cls, number> = { "depth-built": 0, "depth-amplified": 0, "inherited": 0, "weak": 0 };
    for (const v of grid.values()) c[v.cls]++;
    return c;
  }, [grid]);

  const cellSize = 44;
  const gap = 3;
  const labelW = 130;

  return (
    <div className="w-full overflow-x-auto">
      <div className="inline-block">
        {/* Column headers */}
        <div style={{ marginLeft: labelW }} className="mb-1 flex">
          {SCRIPT_ORDER.map((s, i) => (
            <div
              key={s}
              style={{ width: cellSize, marginRight: i === SCRIPT_ORDER.length - 1 ? 0 : gap }}
              className="text-center text-[10px] font-medium text-ink-mute leading-tight truncate"
            >
              {SCRIPT_LABEL[s].split(" ")[0]}
            </div>
          ))}
        </div>

        {/* Rows */}
        {MODEL_ORDER.map((model) => (
          <div key={model} className="flex items-stretch" style={{ marginBottom: gap }}>
            <div
              style={{ width: labelW, height: cellSize }}
              className="flex items-center pr-3 text-right text-[11px] font-medium text-ink"
            >
              {MODEL_LABEL[model]}
            </div>
            <div className="flex">
              {SCRIPT_ORDER.map((s, i) => {
                const v = grid.get(`${model}|${s}`);
                const cls = v?.cls ?? "weak";
                const c = CLS_COLOR[cls];
                return (
                  <div
                    key={s}
                    style={{
                      width: cellSize, height: cellSize,
                      marginRight: i === SCRIPT_ORDER.length - 1 ? 0 : gap,
                      background: c,
                      outline: "1px solid rgba(0,0,0,0.06)",
                      outlineOffset: -1,
                    }}
                    className="transition-transform hover:scale-[1.06]"
                    title={
                      v
                        ? `${MODEL_LABEL[model]} / ${SCRIPT_LABEL[s]} — ${CLS_LABEL[cls]}\nρ = ${v.rho?.toFixed(2) ?? "—"}, helix/PCA = ${v.qual?.toFixed(2) ?? "—"}`
                        : `${MODEL_LABEL[model]} / ${SCRIPT_LABEL[s]} — no data`
                    }
                  />
                );
              })}
            </div>
          </div>
        ))}

        {/* Legend */}
        <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1 text-[11px] text-ink-mute">
          {(["depth-built", "depth-amplified", "inherited", "weak"] as Cls[]).map((cls) => (
            <span key={cls} className="inline-flex items-center gap-1.5">
              <span className="inline-block h-3 w-3 rounded-sm" style={{ background: CLS_COLOR[cls] }} />
              <span>{CLS_LABEL[cls]}</span>
              <span className="text-ink/40">({counts[cls]})</span>
            </span>
          ))}
        </div>

        <p className="mt-2 text-[11px] text-ink-mute">
          Criteria: <span className="font-mono">depth-built</span> = ρ ≤ 0.55 ∧ helix/PCA ≥ 0.70;{" "}
          <span className="font-mono">depth-amplified</span> = ρ ≤ 0.70 ∧ helix/PCA ≥ 0.60;{" "}
          <span className="font-mono">weak</span> = helix/PCA &lt; 0.55;{" "}
          <span className="font-mono">inherited</span> = remainder (high ρ).
        </p>
      </div>
    </div>
  );
}
