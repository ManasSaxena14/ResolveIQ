/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// During development the API runs on uvicorn (make api); Vite forwards API calls to it.
const API = process.env.RESOLVEIQ_API_URL ?? 'http://127.0.0.1:8000'
const API_PATHS = ['/analyze', '/batch', '/feedback', '/health', '/requests', '/stats']

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(API_PATHS.map((path) => [path, API])),
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    // File sync tools (iCloud) leave copies such as "App 2.tsx"; never test them.
    exclude: ['**/node_modules/**', '**/dist/**', '**/* [0-9]*'],
    css: false,
  },
})
