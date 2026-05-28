import { useEffect, useState } from "react";
import type { IndexDoc } from "./lib/types";
import { loadIndex } from "./lib/data";
import { Hero }              from "./components/sections/Hero";
import { PartHeader }        from "./components/PartHeader";
import { NumeralExplainer }  from "./components/sections/NumeralExplainer";
import { Findings }          from "./components/sections/Findings";
import { WhatTheHelixIs }    from "./components/sections/WhatTheHelixIs";
import { TheDiagnostic }     from "./components/sections/TheDiagnostic";
import { Finding1Layers }    from "./components/sections/Finding1Layers";
import { Finding2Scripts }   from "./components/sections/Finding2Scripts";
import { Finding3Babylon }   from "./components/sections/Finding3Babylon";
import { FindingBases }      from "./components/sections/FindingBases";
import { Finding4L0 }        from "./components/sections/Finding4L0";
import { Finding5CKA }       from "./components/sections/Finding5CKA";
import { Conclusion }        from "./components/sections/Conclusion";
import { Reproducing }       from "./components/sections/Reproducing";

export default function App() {
  const [index, setIndex] = useState<IndexDoc | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadIndex().then(setIndex).catch((e) => setError(String(e)));
  }, []);

  return (
    <div className="relative z-10">
      <Hero />

      <main className="layout pb-32">
        {error && (
          <div className="my-8 rounded border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900">
            <strong>No sweep data yet.</strong>{" "}
            The site is rendering with placeholder charts because{" "}
            <code>out/_index.json</code> hasn't been generated. Run{" "}
            <code>./run.sh</code> (the full sweep, ~6–8 h) followed by{" "}
            <code>uv run aggregate.py</code> at the repo root to populate it.
            Charts will hot-fill in dev once the JSONs appear.
          </div>
        )}

        <Findings />

        <PartHeader n={1} title="Setup" />
        <NumeralExplainer />
        <WhatTheHelixIs index={index} />

        <PartHeader n={2} title="What can go wrong with helix R²" />
        <TheDiagnostic />

        <PartHeader n={3} title="Reading the matrix" />
        <Finding1Layers index={index} />
        <Finding2Scripts index={index} />
        <Finding4L0 index={index} />
        <Finding5CKA />

        <PartHeader n={4} title="Case studies" />
        <Finding3Babylon index={index} />
        <FindingBases index={index} />

        <PartHeader n={5} title="Implications" />
        <Conclusion />

        <Reproducing />
      </main>

      <footer className="border-t border-ink/10 bg-paper-warm">
        <div className="layout py-10 text-sm text-ink-mute">
          <p>
            A from-scratch extension of{" "}
            <a className="text-accent underline" href="https://arxiv.org/abs/2502.00873">
              Kantamneni &amp; Tegmark (2025)
            </a>.
          </p>
        </div>
      </footer>
    </div>
  );
}
