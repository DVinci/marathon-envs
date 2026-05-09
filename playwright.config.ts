import { defineConfig } from '@playwright/test';
const isCI = !!process.env.CI;

export default defineConfig({
  testDir: './tests',
  timeout: 30_000,
  reporter: isCI ? 'github' : 'list',
  projects: [
    { name: 'node-tests', testMatch: ['**/*.spec.ts'], use: {} },
  ],
});
