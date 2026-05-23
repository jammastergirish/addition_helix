// Mirrors the JSON shapes that main.py / aggregate.py emit.
// Keep in sync with `dump_json` calls in main.py and the cell payload in
// aggregate.py.

export type Script =
  | "latin" | "arabic" | "persian" | "devanagari"
  | "chinese" | "greek" | "roman" | "babylonian";

export interface RunMeta {
  model: string;
  script: Script;
  pool: string;
  pool_dir: string;
  n_max: number;
  periods: number[];
  n_layers: number;
  layer: number;
  dtype: string;
  sweep: boolean;
}

export interface CellPaths {
  dir: string;
  layer_sweep: string | null;
  fourier_pc1: string | null;
  circles: string | null;
  helix_3d: string | null;
  pca_2d: string | null;
  meta: string | null;
}

export interface IndexCell {
  model: string;
  script: Script;
  pool: string;
  n_max: number;
  periods: number[];
  n_layers: number | null;
  helix_r2_l0: number | null;
  helix_r2_peak: number | null;
  peak_layer: number | null;
  rho: number | null;
  pc1_r2_peak: number | null;
  pc1_r2_peak_layer: number | null;
  helix_over_pca_peak: number | null;
  helix_over_pca_peak_layer: number | null;
  paths: CellPaths;
}

export interface IndexDoc {
  generated_at: string;
  n_cells: number;
  cells: IndexCell[];
}

export interface LayerSweepDoc {
  kind: "layer_sweep";
  meta: RunMeta;
  n_basis: number;
  layers: number[];
  pc1_r2: number[];
  helix_r2: number[];
  pca_kd_r2: number[];
  helix_over_pca: number[];
  peaks: {
    pc1_r2:         { value: number; layer: number };
    helix_r2:       { value: number; layer: number };
    helix_over_pca: { value: number; layer: number };
  };
  l0_share: {
    helix_r2_l0:   number;
    helix_r2_peak: number;
    peak_layer:    number;
    rho:           number | null;
  };
}

export interface FourierPC1Doc {
  kind: "fourier_pc1";
  meta: RunMeta;
  periods_ref: number[];
  fft: {
    freqs: number[];
    magnitudes: number[];
    auto_peaks: { freq: number; period: number; magnitude: number }[];
  };
  pc1: {
    numbers: number[];
    values: number[];
    fit_slope: number;
    fit_intercept: number;
    r2: number;
    explained_variance_ratio: number[];
  };
}

export interface CirclesDoc {
  kind: "circles_and_line";
  meta: RunMeta;
  numbers: number[];
  labels: string[];
  circles: { T: number; coords: [number, number][]; residues: number[] }[];
  line: { coords: number[] };
}

export interface Helix3DDoc {
  kind: "helix_3d";
  meta: RunMeta;
  T: number;
  numbers: number[];
  labels: string[];
  coords: [number, number, number][];
  axes: string[];
}

export interface PCA2DDoc {
  kind: "pca_2d";
  meta: RunMeta;
  numbers: number[];
  labels: string[];
  coords: [number, number][];
  explained_variance_ratio: number[];
}

export interface MetaDoc {
  kind: "meta";
  meta: RunMeta;
  helix_r2: number;
  pca_kd_r2: number;
  helix_over_pca: number | null;
  n_basis: number;
  T_helix: number;
  n_numbers: number;
}
