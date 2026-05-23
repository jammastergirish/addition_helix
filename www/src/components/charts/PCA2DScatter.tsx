import * as d3 from "d3";
import type { PCA2DDoc } from "../../lib/types";
import { VIRIDIS } from "../../lib/svg";

interface Props { data: PCA2DDoc; height?: number; }

export function PCA2DScatter({ data, height = 480 }: Props) {
  const W = 720, H = height;
  const margin = { top: 14, right: 18, bottom: 36, left: 48 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;

  const xs = data.coords.map((p) => p[0]);
  const ys = data.coords.map((p) => p[1]);
  const x = d3.scaleLinear().domain(d3.extent(xs) as [number, number]).range([0, innerW]).nice();
  const y = d3.scaleLinear().domain(d3.extent(ys) as [number, number]).range([innerH, 0]).nice();
  const nMax = Math.max(...data.numbers, 1);
  const line = d3.line<[number, number]>().x((p) => x(p[0])).y((p) => y(p[1]));

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          {x.ticks(5).map((t) => (
            <g key={t} transform={`translate(${x(t)}, 0)`}>
              <line y1={0} y2={innerH} stroke="#f1f5f9" />
              <text y={innerH + 14} fontSize={10} textAnchor="middle" fill="#6b7280">{t.toFixed(1)}</text>
            </g>
          ))}
          {y.ticks(5).map((t) => (
            <g key={t} transform={`translate(0, ${y(t)})`}>
              <line x1={0} x2={innerW} stroke="#f1f5f9" />
              <text x={-8} y={3} fontSize={10} textAnchor="end" fill="#6b7280">{t.toFixed(1)}</text>
            </g>
          ))}
          <text transform={`translate(${innerW / 2}, ${innerH + 30})`} textAnchor="middle" fontSize={11} fill="#374151">
            PC1 ({(data.explained_variance_ratio[0] * 100).toFixed(1)}%)
          </text>
          <text transform={`translate(${-36}, ${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={11} fill="#374151">
            PC2 ({(data.explained_variance_ratio[1] * 100).toFixed(1)}%)
          </text>

          {/* Faint trajectory between consecutive integers */}
          <path d={line(data.coords) ?? ""} fill="none" stroke="#94a3b8" strokeWidth={0.6} opacity={0.5} />

          {data.coords.map((p, i) => (
            <g key={i}>
              <circle cx={x(p[0])} cy={y(p[1])} r={4} fill={VIRIDIS(data.numbers[i] / nMax)} opacity={0.85} />
              <text x={x(p[0])} y={y(p[1]) - 6} fontSize={8} textAnchor="middle" fill="#1f2937">{data.labels[i]}</text>
            </g>
          ))}
        </g>
      </svg>
    </div>
  );
}
