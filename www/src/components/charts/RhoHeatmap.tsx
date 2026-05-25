import { useMemo, useState } from "react";
import type { IndexDoc } from "../../lib/types";
import { MODEL_ORDER, MODEL_LABEL, SCRIPT_ORDER, SCRIPT_LABEL } from "../../lib/data";
import { fmtPct, rhoColor } from "../../lib/svg";

interface Props { index: IndexDoc | null; }

/**
 * The L=0 vs peak ρ heatmap — the blogpost's Finding 4. ρ ≈ 1 means
 * "the helix is already there at the embedding layer; the transformer
 * adds essentially nothing." ρ small means depth is doing real work.
 *
 * One cell per (model, script) at the paper-default n_max=100, periods=[2,5,10,100].
 */
export function RhoHeatmap({ index }: Props) {
  const [hover, setHover] = useState<{ model: string; script: string } | null>(null);

  const grid = useMemo(() => {
    const m = new Map<string, number | null>();
    if (!index) return m;
    for (const c of index.cells) {
      if (c.n_max !== 100) continue;
      if (c.periods.length !== 4 || c.periods.some((p, i) => p !== [2,5,10,100][i])) continue;
      m.set(`${c.model}|${c.script}`, c.rho);
    }
    return m;
  }, [index]);

  const cellSize = 64;
  const gap = 4;
  const labelW = 130;

  return (
    <div className="w-full overflow-x-auto">
      <div className="inline-block">
        {/* Column (script) headers */}
        <div style={{ marginLeft: labelW }} className="mb-1 flex" >
          {SCRIPT_ORDER.map((s, i) => (
            <div
              key={s}
              style={{ width: cellSize, marginRight: i === SCRIPT_ORDER.length - 1 ? 0 : gap }}
              className="text-center text-[11px] font-medium text-ink-mute leading-tight"
            >
              <div className="truncate">{SCRIPT_LABEL[s]}</div>
            </div>
          ))}
        </div>

        {/* Rows */}
        {MODEL_ORDER.map((model) => (
          <div key={model} className="flex items-stretch" style={{ marginBottom: gap }}>
            <div
              style={{ width: labelW, height: cellSize }}
              className="flex items-center pr-3 text-right text-[12px] font-medium text-ink"
            >
              {MODEL_LABEL[model]}
            </div>
            <div className="flex">
              {SCRIPT_ORDER.map((s, i) => {
                const v = grid.get(`${model}|${s}`) ?? null;
                const c = rhoColor(v);
                const isHi = hover && hover.model === model && hover.script === s;
                return (
                  <div
                    key={s}
                    onMouseEnter={() => setHover({ model, script: s })}
                    onMouseLeave={() => setHover(null)}
                    style={{
                      width: cellSize, height: cellSize,
                      marginRight: i === SCRIPT_ORDER.length - 1 ? 0 : gap,
                      background: c,
                      outline: isHi ? "2px solid #1f2937" : "1px solid rgba(0,0,0,0.06)",
                      outlineOffset: -1,
                    }}
                    className="flex flex-col items-center justify-center text-center transition-transform hover:scale-[1.04]"
                    title={`${MODEL_LABEL[model]} / ${SCRIPT_LABEL[s]} — ρ = ${fmtPct(v)}`}
                  >
                    <div className="text-[14px] font-semibold leading-none text-ink/85">
                      {v == null ? "—" : v.toFixed(2)}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}

        {/* Legend */}
        <div className="mt-4 flex items-center gap-3 text-xs text-ink-mute">
          <span>depth builds it</span>
          <div className="h-3 w-48 rounded" style={{
            background:
              "linear-gradient(to right, #0f766e, #5eead4, #f5f3ed, #fb923c, #9a3412)",
          }} />
          <span>input pipeline supplies it</span>
          <span className="ml-4">ρ = helix R²(L=0) / helix R²(peak)</span>
        </div>
      </div>
    </div>
  );
}
