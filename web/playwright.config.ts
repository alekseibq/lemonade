import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30_000,
  use: { baseURL: 'http://127.0.0.1:5605', ...devices['Desktop Chrome'] },
  webServer: [
    {
      command: 'PYTHONPATH=.. ../.venv/bin/uvicorn lemonade.api:app --host 127.0.0.1 --port 8000',
      url: 'http://127.0.0.1:8000/health',
      reuseExistingServer: !process.env.CI,
    },
    {
      command: 'npm run dev',
      url: 'http://127.0.0.1:5605',
      reuseExistingServer: !process.env.CI,
    },
  ],
})
