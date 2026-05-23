import * as d3 from "d3";
import type { FourierPC1Doc } from "../../lib/types";

interface Props { data: FourierPC1Doc; height?: number; }

export function FFTSpectrum({ data, height = 260 }: Props) {
  const W = 720, H = height;
  const margin = { top: 14, right: 18, bottom: 36, left: 44 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;

  // Skip the DC bin when computing y-scale (so it doesn't squash everything).
  const freqs = data.fft.freqs;
  const mags = data.fft.magnitudes;
  const yMax = d3.max(mags.slice(1)) ?? 1;

  const x = d3.scaleLinear().domain([0, 0.51]).range([0, innerW]);
  const y = d3.scaleLinear().domain([0, yMax * 1.05]).range([innerH, 0]).nice();
  const line = d3.line<{ f: number; m: number }>()
    .x((d) => x(d.f))
    .y((d) => y(d.m))
    .curve(d3.curveLinear);
  const pts = freqs.map((f, i) => ({ f, m: mags[i] }));

  const yTicks = y.ticks(4);
  const xTicks = [0, 0.1, 0.2, 0.3, 0.4, 0.5];

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          {yTicks.map((t) => (
            <g key={t} transform={`translate(0, ${y(t)})`}>
              <line x1={0} x2={innerW} stroke="#e5e7eb" />
              <text x={-8} y={3} fontSize={10} textAnchor="end" fill="#6b7280">{t.toFixed(2)}</text>
            </g>
          ))}
          {xTicks.map((t) => (
            <g key={t} transform={`translate(${x(t)}, ${innerH})`}>
              <line y1={0} y2={4} stroke="#9ca3af" />
              <text y={16} fontSize={10} textAnchor="middle" fill="#6b7280">{t.toFixed(1)}</text>
            </g>
          ))}
          <text transform={`translate(${innerW / 2}, ${innerH + 30})`} textAnchor="middle" fontSize={11} fill="#374151">
            frequency (cycles per integer)
          </text>

          {/* Reference period lines (the basis we fit) */}
          {data.periods_ref.map((T) => (
            <g key={T} transform={`translate(${x(1 / T)}, 0)`}>
              <line y1={0} y2={innerH} stroke="#cbd5e1" strokeDasharray="2 3" />
              <text x={4} y={innerH - 4} fontSize={9} fill="#64748b">T={T}</text>
            </g>
          ))}

          <path d={line(pts) ?? ""} fill="none" stroke="#c2410c" strokeWidth={1.6} />

          {/* Detected peaks */}
          {data.fft.auto_peaks.slice(0, 5).map((p, i) => (
            <g key={i} transform={`translate(${x(p.freq)}, ${y(p.magnitude)})`}>
              <circle r={4} fill="#c2410c" stroke="white" strokeWidth={1.5} />
              <g transform={`translate(0, ${-12})`}>
                <rect x={-18} y={-13} width={36} height={14} rx={3} fill="#fff7ed" stroke="#fdba74" />
                <text textAnchor="middle" y={-3} fontSize={9} fill="#7c2d12">T≈{p.period.toFixed(1)}</text>
              </g>
            </g>
          ))}
        </g>
      </svg>
    </div>
  );
}
