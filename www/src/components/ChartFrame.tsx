import { useEffect, useState } from "react";
import { loadJson } from "../lib/data";

interface Props<T> {
  src: string | null | undefined;
  caption?: React.ReactNode;
  children: (data: T) => React.ReactNode;
  /** Height of the placeholder/loading state (in px). */
  minHeight?: number;
}

/**
 * Lazy-loads JSON from /data/<src>, renders a graceful placeholder while
 * loading or when the file is missing (e.g. the user hasn't re-run main.py
 * yet). Centralises the "data not available" message so every chart looks
 * the same when it has no data.
 */
export function ChartFrame<T>({ src, caption, children, minHeight = 320 }: Props<T>) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setData(null);
    setError(null);
    if (!src) {
      setError("not available");
      return;
    }
    let alive = true;
    loadJson<T>(src)
      .then((d) => { if (alive) setData(d); })
      .catch((e) => { if (alive) setError(String(e)); });
    return () => { alive = false; };
  }, [src]);

  return (
    // `wide` opts this figure into the outer breakout grid track, so charts
    // get more horizontal room than the surrounding body text.
    <figure className="wide my-8">
      <div
        className="rounded-lg border border-ink/10 bg-white shadow-[0_1px_2px_rgba(0,0,0,0.04)] p-4"
        style={{ minHeight }}
      >
        {data ? (
          children(data)
        ) : error ? (
          <div className="flex h-full min-h-[200px] items-center justify-center text-sm text-ink-mute">
            <div className="text-center">
              <div className="font-medium text-ink/70">Chart data not loaded</div>
              <div className="mt-1 text-xs">
                Source: <code className="text-ink/60">{src ?? "(none)"}</code>
              </div>
              <div className="mt-2 text-xs">
                Run <code>./run.sh &amp;&amp; uv run aggregate.py</code> to populate it.
              </div>
            </div>
          </div>
        ) : (
          <div className="flex h-full min-h-[200px] items-center justify-center text-sm text-ink-mute">
            loading…
          </div>
        )}
      </div>
      {caption && (
        <figcaption className="mx-auto mt-3 max-w-prose text-sm text-ink-mute leading-snug">
          {caption}
        </figcaption>
      )}
    </figure>
  );
}
