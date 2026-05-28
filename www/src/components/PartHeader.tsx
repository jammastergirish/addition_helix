interface Props {
  n: number;
  title: string;
}

/**
 * Visual divider between Parts. An oversized zero-padded Part number in
 * the display face anchors the divider; a kicker label and the title sit
 * beneath it, flanked by a thin accent rule. Purely a visual cue, not a
 * semantic heading level (sections keep their own h2).
 */
export function PartHeader({ n, title }: Props) {
  return (
    <div className="my-24 flex flex-col items-center text-center first:mt-12">
      <div className="font-display text-[4rem] md:text-[5.5rem] font-black leading-[0.85] tracking-[-0.03em] text-accent/15 select-none">
        {String(n).padStart(2, "0")}
      </div>
      <div className="-mt-3 flex items-center gap-3">
        <span className="h-px w-8 bg-accent/40" />
        <span className="kicker">Part {n}</span>
        <span className="h-px w-8 bg-accent/40" />
      </div>
      <h2 className="mt-3 font-display text-2xl md:text-[2rem] font-extrabold tracking-[-0.02em] text-ink leading-[1.05]">
        {title}
      </h2>
    </div>
  );
}
