# Interactive diagnostic dashboard

A client-side React/Vite app that walks through the pipeline's diagnostic
machinery — trace inspection with First-Divergence-Point highlighting, the
contingency-matrix/χ² view, the failure-signature taxonomy, judge
calibration, and the phase-by-phase pipeline layout.

**The example traces are illustrative, not the real study data** — see the
banner in the running app. For the actual findings, numbers, and
reproduction commands, see the repo root [`README.md`](../README.md) and
[`RESULTS.md`](../research/kv-cache-reasoning-divergence-study/RESULTS.md).
Everything here runs client-side (no backend, no API keys) so it's static
and deployable to GitHub Pages as-is.

## Run locally

```bash
cd webapp
npm install
npm run dev       # http://localhost:3000, via server.ts + Vite middleware
```

`npm run dev` starts a small Express server (`server.ts`) that wraps Vite in
middleware mode; the mock `/api/*` routes it exposes exist only for local
convenience and aren't called by the frontend (nothing here depends on
them, including in production).

## Build

```bash
npm run build      # vite build + bundles server.ts to dist/server.cjs
npm start           # serve dist/ via the bundled Express server
```

For a pure static build (what GitHub Pages actually serves):

```bash
npx vite build --base=/kv-cache-quantization-reasoning-study/
```

Deployment to GitHub Pages is automated by
[`.github/workflows/deploy-pages.yml`](../.github/workflows/deploy-pages.yml)
on every push to `main` that touches `webapp/`.
