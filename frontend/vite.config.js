import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  server: { host: true, port: 5173 },
  build: {
    rolldownOptions: {
      output: {
        // keep the charting library in its own cacheable chunk
        advancedChunks: { groups: [{ name: 'charts', test: /node_modules[\\/](recharts|d3-|victory)/ }] },
      },
    },
  },
  test: { environment: 'jsdom', setupFiles: './src/test/setup.js', globals: true },
})
