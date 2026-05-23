import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import fs from "node:fs";
// In dev: serve `../out/` at `/data/` so the React app can fetch the JSON
// files main.py produces, with no copy step. For production builds, the
// helper script `scripts/stage-data.sh` (or the `build` script in
// package.json calling it) copies the *.json files into public/data/.
var OUT_DIR = path.resolve(__dirname, "..", "out");
export default defineConfig({
    plugins: [
        react(),
        {
            name: "serve-out-as-data",
            configureServer: function (server) {
                server.middlewares.use("/data", function (req, res, next) {
                    if (!req.url)
                        return next();
                    // Strip query, decode percent-escapes.
                    var rel = decodeURIComponent(req.url.split("?")[0]);
                    var filePath = path.join(OUT_DIR, rel);
                    // Stay inside OUT_DIR.
                    if (!filePath.startsWith(OUT_DIR)) {
                        res.statusCode = 403;
                        return res.end("forbidden");
                    }
                    fs.stat(filePath, function (err, stat) {
                        if (err || !stat.isFile()) {
                            // Don't fall through to the SPA handler -- that would return
                            // index.html (HTML) for a request the caller expects to be
                            // JSON, producing confusing parse errors. Return a clean 404
                            // so the chart loader can render the "not available" state.
                            res.statusCode = 404;
                            return res.end("not found: " + rel);
                        }
                        res.setHeader("Content-Type", filePath.endsWith(".json") ? "application/json" :
                            filePath.endsWith(".png") ? "image/png" :
                                filePath.endsWith(".csv") ? "text/csv" :
                                    "application/octet-stream");
                        fs.createReadStream(filePath).pipe(res);
                    });
                });
            },
        },
    ],
    resolve: {
        alias: {
            "@": path.resolve(__dirname, "src"),
        },
    },
    server: {
        port: 5173,
        open: true,
    },
    build: {
        target: "es2020",
        outDir: "dist",
        sourcemap: false,
    },
});
