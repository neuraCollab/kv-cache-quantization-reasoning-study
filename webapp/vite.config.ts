import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// https://vite.dev/config/
export default defineConfig({
  // GitHub Pages serves this at /<repo-name>/, not /. Overridden by
  // `vite build --base=...` in .github/workflows/deploy-pages.yml; this
  // default only matters for `vite preview` / local `npm run build`.
  base: process.env.VITE_BASE_PATH || '/',
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    host: '0.0.0.0',
    port: 3000,
  },
});
