import * as d3 from "d3";
import type { FourierPC1Doc } from "../../lib/types";
import { VIRIDIS } from "../../lib/svg";

interface Props { data: FourierPC1Doc; height?: number; }

export function PC1Scatter({ data, height = 260 }: Props) {
  const W = 720, H = height;
  const margin = { top: 14, right: 18, bottom: 36, left: 48 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;

  const ns = data.pc1.numbers;
  const vs = data.pc1.values;
  const x = d3.scaleLinear().domain(d3.extent(ns) as [number, number]).range([0, innerW]).nice();
  const y = d3.scaleLinear().domain(d3.extent(vs) as [number, number]).range([innerH, 0]).nice();
  const aMax = Math.max(1, d3.max(ns) ?? 1);

  const fit = (a: number) => data.pc1.fit_slope * a + data.pc1.fit_intercept;
  const xTicks = x.ticks(6);
  const yTicks = y.ticks(5);

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          {yTicks.map((t) => (
            <g key={t} transform={`translate(0, ${y(t)})`}>
              <line x1={0} x2={innerW} stroke="#e5e7eb" />
              <text x={-8} y={3} fontSize={10} textAnchor="end" fill="#6b7280">{t.toFixed(1)}</text>
            </g>
          ))}
          {xTicks.map((t) => (
            <g key={t} transform={`translate(${x(t)}, ${innerH})`}>
              <line y1={0} y2={4} stroke="#9ca3af" />
              <text y={16} fontSize={10} textAnchor="middle" fill="#6b7280">{t}</text>
            </g>
          ))}
          <text transform={`translate(${innerW / 2}, ${innerH + 30})`} textAnchor="middle" fontSize={11} fill="#374151">
            integer a
          </text>
          <text transform={`translate(${-34}, ${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={11} fill="#374151">
            PC1 projection
          </text>

          <line
            x1={x(ns[0])} y1={y(fit(ns[0]))}
            x2={x(ns[ns.length - 1])} y2={y(fit(ns[ns.length - 1]))}
            stroke="#dc2626" strokeDasharray="4 4" strokeWidth={1.6}
          />

          {ns.map((a, i) => (
            <circle
              key={a}
              cx={x(a)} cy={y(vs[i])}
              r={3.2}
              fill={VIRIDIS(a / aMax)}
              opacity={0.9}
            />
          ))}

          <text x={innerW - 6} y={14} textAnchor="end" fontSize={11} fill="#7c2d12">
            R² = {data.pc1.r2.toFixed(3)}
          </text>
        </g>
      </svg>
    </div>
  );
}
