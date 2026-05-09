import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

const ROOT = path.resolve(__dirname, '..');

// Only run link checks when explicitly enabled (CI weekly job or LINK_CHECK=true)
const LINK_CHECK = process.env.LINK_CHECK === 'true';

const MD_FILES = [
  'README.md',
  'CLAUDE.md',
  'Training.md',
  'Changes.md',
].map(f => path.join(ROOT, f));

function extractUrls(content: string): string[] {
  const urlRegex = /https?:\/\/[^\s)<>"']+/g;
  return [...new Set(content.match(urlRegex) ?? [])];
}

async function checkUrl(url: string): Promise<{ ok: boolean; status: number | null }> {
  try {
    const res = await fetch(url, {
      method: 'HEAD',
      signal: AbortSignal.timeout(10_000),
      headers: { 'User-Agent': 'marathon-envs-link-checker/1.0' },
    });
    // Some servers reject HEAD; retry with GET
    if (res.status === 405) {
      const res2 = await fetch(url, {
        method: 'GET',
        signal: AbortSignal.timeout(10_000),
        headers: { 'User-Agent': 'marathon-envs-link-checker/1.0' },
      });
      return { ok: res2.ok, status: res2.status };
    }
    return { ok: res.ok, status: res.status };
  } catch {
    return { ok: false, status: null };
  }
}

test.describe('External link validation', () => {
  test.skip(!LINK_CHECK, 'Set LINK_CHECK=true to run link checks');

  for (const mdFile of MD_FILES) {
    const fileName = path.basename(mdFile);

    test(`all URLs in ${fileName} are reachable`, async () => {
      if (!fs.existsSync(mdFile)) return;
      const content = fs.readFileSync(mdFile, 'utf8');
      const urls = extractUrls(content);

      const failures: string[] = [];
      for (const url of urls) {
        const { ok, status } = await checkUrl(url);
        if (!ok) {
          failures.push(`${url} → ${status ?? 'network error'}`);
        }
      }

      expect(failures, `Broken links in ${fileName}:\n${failures.join('\n')}`).toHaveLength(0);
    });
  }
});

test.describe('Markdown file existence', () => {
  test('all expected documentation files exist', () => {
    for (const mdFile of MD_FILES) {
      expect(fs.existsSync(mdFile), `Missing: ${path.basename(mdFile)}`).toBe(true);
    }
  });
});
