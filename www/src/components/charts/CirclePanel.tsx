import * as d3 from "d3";
import type { CirclesDoc } from "../../lib/types";
import { VIRIDIS } from "../../lib/svg";

interface Props { data: CirclesDoc; }

export function CirclePanel({ data }: Props) {
  const ns = data.numbers;
  const labels = data.labels;
  const aMax = Math.max(1, Math.max(...ns));

  const SIZE = 200;
  const M = 16;

  return (
    <div className="w-full">
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {data.circles.map((c) => {
          const xs = c.coords.map((p) => p[0]);
          const ys = c.coords.map((p) => p[1]);
          const x = d3.scaleLinear().domain(d3.extent(xs) as [number, number]).range([M, SIZE - M]).nice();
          const y = d3.scaleLinear().domain(d3.extent(ys) as [number, number]).range([SIZE - M, M]).nice();
          const seen = new Set<number>();
          return (
            <figure key={c.T} className="rounded-md border border-ink/10 bg-paper-warm/40 p-2">
              <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="w-full">
                {c.coords.map(([px, py], i) => (
                  <circle key={i} cx={x(px)} cy={y(py)} r={3.5} fill={VIRIDIS(ns[i] / aMax)} opacity={0.9} />
                ))}
                {/* Label one number per residue class for small T */}
                {c.T <= 10 && c.coords.map(([px, py], i) => {
                  const r = c.residues[i];
                  if (seen.has(r)) return null;
                  seen.add(r);
                  return (
                    <text
                      key={`l-${i}`}
                      x={x(px)} y={y(py) - 5}
                      textAnchor="middle"
                      fontSize={9}
                      fill="#1f2937"
                    >
                      {labels[i]}
                    </text>
                  );
                })}
              </svg>
              <figcaption className="mt-1 text-center text-xs font-mono text-ink-mute">
                T = {c.T}
              </figcaption>
            </figure>
          );
        })}
      </div>

      {/* Linear "number line" strip */}
      <div className="mt-4 rounded-md border border-ink/10 bg-paper-warm/40 p-2">
        <NumberLine line={data.line.coords} numbers={ns} labels={labels} />
        <div className="mt-1 text-center text-xs font-mono text-ink-mute">linear axis (the "rise")</div>
      </div>
    </div>
  );
}

function NumberLine({ line, numbers, labels }: { line: number[]; numbers: number[]; labels: string[]; }) {
  const W = 720, H = 70, M = 16;
  const x = d3.scaleLinear().domain(d3.extent(line) as [number, number]).range([M, W - M]).nice();
  const aMax = Math.max(1, Math.max(...numbers));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      <line x1={M} x2={W - M} y1={H / 2} y2={H / 2} stroke="#cbd5e1" />
      {line.map((v, i) => (
        <circle key={i} cx={x(v)} cy={H / 2} r={4} fill={VIRIDIS(numbers[i] / aMax)} opacity={0.85} />
      ))}
      {line.map((v, i) => (i % 5 === 0 ? (
        <text key={`l-${i}`} x={x(v)} y={H / 2 - 8} textAnchor="middle" fontSize={9} fill="#374151">
          {labels[i]}
        </text>
      ) : null))}
    </svg>
  );
}
