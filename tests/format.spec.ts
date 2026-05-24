import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'js-yaml';

const ROOT = path.resolve(__dirname, '..');
const CONFIG_YAML = path.join(ROOT, 'config', 'marathon_envs_config.yaml');

const REQUIRED_DEFAULT_KEYS = [
  'trainer', 'batch_size', 'buffer_size', 'learning_rate',
  'max_steps', 'normalize', 'num_epoch', 'hidden_units',
];

const KNOWN_ENVIRONMENTS = [
  'MarathonManBackflip-v0',
  'MarathonManWalking-v0',
  'MarathonManRunning-v0',
  'MarathonManJazzDancing-v0',
  'MarathonManMMAKick-v0',
  'MarathonManPunchingBag-v0',
  'MarathonMan-v0',
  'MarathonManSparse-v0',
  'ControllerMarathonMan-v0',
  'Hopper-v0',
  'Walker2d-v0',
  'Ant-v0',
  'TerrainHopper-v0',
  'TerrainWalker2d-v0',
  'TerrainAnt-v0',
  'TerrainMarathonMan-v0',
];

test.describe('YAML config format', () => {
  let config: Record<string, unknown>;

  test.beforeAll(() => {
    const raw = fs.readFileSync(CONFIG_YAML, 'utf8');
    config = yaml.load(raw) as Record<string, unknown>;
  });

  test('config file parses without error', () => {
    expect(config).toBeTruthy();
    expect(typeof config).toBe('object');
  });

  test('default block contains all required keys', () => {
    const defaults = config['default'] as Record<string, unknown>;
    expect(defaults).toBeTruthy();
    for (const key of REQUIRED_DEFAULT_KEYS) {
      expect(defaults, `default.${key} should exist`).toHaveProperty(key);
    }
  });

  test('all top-level keys are either "default" or a known environment', () => {
    const keys = Object.keys(config).filter(k => k !== 'default');
    for (const key of keys) {
      expect(KNOWN_ENVIRONMENTS, `Unknown environment key: ${key}`).toContain(key);
    }
  });

  test('numeric hyperparameters are positive where present', () => {
    for (const [env, block] of Object.entries(config)) {
      if (env === 'default') continue;
      const b = block as Record<string, unknown>;
      for (const field of ['batch_size', 'buffer_size', 'max_steps', 'learning_rate']) {
        if (field in b) {
          expect(Number(b[field]), `${env}.${field} must be > 0`).toBeGreaterThan(0);
        }
      }
    }
  });

  test('trainer field is "ppo" when present', () => {
    const defaults = config['default'] as Record<string, unknown>;
    expect(defaults['trainer']).toBe('ppo');
  });
});
