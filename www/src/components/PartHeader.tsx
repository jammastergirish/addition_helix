interface Props {
  n: number;
  title: string;
}

/**
 * Visual divider between Parts. Renders as a small uppercase "Part N"
 * label flanked by horizontal rules, with a slightly larger title
 * underneath. Sections retain their own h2; PartHeader is purely
 * a visual cue, not a semantic heading level.
 */
export function PartHeader({ n, title }: Props) {
  return (
    <div className="my-24 text-center first:mt-12">
      <div className="inline-flex items-center gap-3 text-xs uppercase tracking-[0.2em] text-ink-mute">
        <span className="h-px w-12 bg-ink/20" />
        Part {n}
        <span className="h-px w-12 bg-ink/20" />
      </div>
      <div className="mt-2 font-sans text-xl md:text-2xl font-medium tracking-tight text-ink/90">
        {title}
      </div>
    </div>
  );
}
