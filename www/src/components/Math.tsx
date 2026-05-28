// Thin wrappers around react-katex so the rest of the codebase doesn't
// import from it directly. Inline math goes flush in a paragraph;
// block math centres on its own line with vertical spacing.

import { InlineMath, BlockMath } from "react-katex";

export function M({ children }: { children: string }) {
  return <InlineMath math={children} />;
}

export function MM({ children }: { children: string }) {
  return (
    <div className="my-3 overflow-x-auto">
      <BlockMath math={children} />
    </div>
  );
}
