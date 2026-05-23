// Tiny helpers for the d3-rendered SVG charts. We use d3 only for scales /
// shape generators; React owns the DOM, which keeps re-renders cheap.

import * as d3 from "d3";

export const VIRIDIS = (t: number) => d3.interpolateViridis(t);

export interface Margin { top: number; right: number; bottom: number; left: number; }

export const DEFAULT_MARGIN: Margin = { top: 16, right: 24, bottom: 36, left: 48 };

export function tickValues(scale: d3.ScaleLinear<number, number>, n = 5): number[] {
  return scale.ticks(n);
}

// Color stops for the rho heatmap. Diverging palette so 1.0 ("all tokeniser")
// is visually distinct from 0.0 ("all depth"). Mid = 0.5.
export function rhoColor(rho: number | null | undefined): string {
  if (rho == null || !isFinite(rho)) return "#e5e7eb";
  // Below 0.5 is "depth-built" (the good case): teal -> mid grey.
  // Above 0.5 is "tokeniser-inherited": mid grey -> rust.
  const t = Math.max(0, Math.min(1, rho));
  return d3.interpolateRgbBasis([
    "#0f766e", // teal-700
    "#5eead4", // teal-300
    "#f5f3ed", // paper
    "#fb923c", // orange-400
    "#9a3412", // orange-800
  ])(t);
}

export function fmtPct(v: number | null | undefined, digits = 2): string {
  if (v == null || !isFinite(v)) return "—";
  return v.toFixed(digits);
}
