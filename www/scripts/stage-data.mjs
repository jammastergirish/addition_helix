#!/usr/bin/env node
/**
 * Copy every *.json, *.csv, and the small comparison-image PNGs under
 * out/_compare/ into dist/data/. Runs after `vite build` via the
 * `build` script in package.json, so a single `npm run build` produces
 * a fully deployable dist/ with all chart data baked in.
 *
 * Skips per-cell PNGs (those are the matplotlib figures that the React
 * site re-renders interactively from the JSON), log files, and model
 * weights.
 */
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SRC  = path.resolve(__dirname, "..", "..", "out");
const DEST = path.resolve(__dirname, "..", "dist", "data");

const KEEP_EXT = new Set([".json", ".csv"]);
const SKIP_DIRS = new Set(["_logs"]);
// PNGs are only kept inside the _compare/ directory (the cross-model
// comparison images). Per-cell figure PNGs are skipped — the React
// charts re-render them from the JSON.
const KEEP_PNG_DIR = "_compare";

async function* walk(dir) {
  let entries;
  try {
    entries = await fs.readdir(dir, { withFileTypes: true });
  } catch (e) {
    if (e.code === "ENOENT") return;
    throw e;
  }
  for (const e of entries) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) {
      if (SKIP_DIRS.has(e.name)) continue;
      yield* walk(p);
    } else if (e.isFile()) {
      const ext = path.extname(e.name);
      if (KEEP_EXT.has(ext)) {
        yield p;
      } else if (ext === ".png" && dir.split(path.sep).includes(KEEP_PNG_DIR)) {
        yield p;
      }
    }
  }
}

async function main() {
  try {
    await fs.access(SRC);
  } catch {
    console.warn(`stage-data: ${SRC} doesn't exist — nothing to stage. ` +
                 `Run ./run.sh && uv run aggregate.py at the repo root first.`);
    return;
  }

  // Fresh dist/data each build so deletions in out/ propagate cleanly.
  await fs.rm(DEST, { recursive: true, force: true });
  await fs.mkdir(DEST, { recursive: true });

  let count = 0;
  let bytes = 0;
  for await (const src of walk(SRC)) {
    const rel = path.relative(SRC, src);
    const dst = path.join(DEST, rel);
    await fs.mkdir(path.dirname(dst), { recursive: true });
    await fs.copyFile(src, dst);
    bytes += (await fs.stat(src)).size;
    count++;
  }
  const mb = (bytes / (1024 * 1024)).toFixed(2);
  console.log(`stage-data: copied ${count} files (${mb} MB) → dist/data/`);
}

main().catch((e) => {
  console.error("stage-data failed:", e);
  process.exit(1);
});
