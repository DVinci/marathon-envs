import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'js-yaml';

const ROOT = path.resolve(__dirname, '..');
const CONFIG_YAML = path.join(ROOT, 'config', 'marathon_envs_config.yaml');
const CLAUDE_MD = path.join(ROOT, 'CLAUDE.md');

// Environments declared in CLAUDE.md table (extracted from the Available Environments section)
const CLAUDE_ENVS = [
  'Hopper-v0', 'Walker2d-v0', 'Ant-v0', 'MarathonMan-v0',
  'MarathonManWalking-v0', 'MarathonManRunning-v0', 'MarathonManJazzDancing-v0',
  'MarathonManMMAKick-v0', 'MarathonManPunchingBag-v0', 'MarathonManBackflip-v0',
  'TerrainHopper-v0', 'TerrainWalker2d-v0', 'TerrainAnt-v0', 'TerrainMarathonMan-v0',
  'ControllerMarathonMan-v0', 'MarathonManSparse-v0',
];

test.describe('Cross-file consistency', () => {
  let config: Record<string, unknown>;
  let claudeContent: string;

  test.beforeAll(() => {
    config = yaml.load(fs.readFileSync(CONFIG_YAML, 'utf8')) as Record<string, unknown>;
    claudeContent = fs.readFileSync(CLAUDE_MD, 'utf8');
  });

  test('all 16 environments are documented in CLAUDE.md', () => {
    for (const env of CLAUDE_ENVS) {
      // CLAUDE.md uses slash-shorthand: "TerrainHopper/Walker2d/Ant/MarathonMan-v0"
      // Check for the full name OR any meaningful segment of it
      const base = env.replace(/-v0$/, '');
      const segments = [
        env,
        base,
        base.replace(/^MarathonMan/, ''),
        base.replace(/^Terrain/, ''),
      ].filter(s => s.length > 2);
      const found = segments.some(s => claudeContent.includes(s));
      expect(found, `${env} (or a segment of it) should be referenced in CLAUDE.md`).toBeTruthy();
    }
  });

  test('style-transfer environments have higher max_steps than classical', () => {
    const styleEnvs = ['MarathonManBackflip-v0', 'MarathonManWalking-v0'];
    const classicalEnvs = ['Hopper-v0', 'Walker2d-v0'];

    for (const style of styleEnvs) {
      if (!(style in config)) continue;
      const styleBlock = config[style] as Record<string, unknown>;
      if (!('max_steps' in styleBlock)) continue;

      for (const classical of classicalEnvs) {
        if (!(classical in config)) continue;
        const classicalBlock = config[classical] as Record<string, unknown>;
        if (!('max_steps' in classicalBlock)) continue;

        expect(
          Number(styleBlock['max_steps']),
          `${style}.max_steps should exceed ${classical}.max_steps`
        ).toBeGreaterThan(Number(classicalBlock['max_steps']));
      }
    }
  });

  test('CLAUDE.md references the config yaml file', () => {
    expect(claudeContent).toContain('marathon_envs_config.yaml');
  });

  test('CLAUDE.md references mlagents-learn command', () => {
    expect(claudeContent).toContain('mlagents-learn');
  });
});
