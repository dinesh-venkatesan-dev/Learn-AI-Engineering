import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// `npm run build` writes dist/, which the Python app serves at http://localhost:8300.
// `npm run dev` (port 5178) is for editing the UI; it forwards /api to the Python app.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5178, proxy: { '/api': 'http://127.0.0.1:8300' } },
})
