import { expect, test } from 'claude-code/testing'

const ART = '/repo/artifacts/2026-11-24-qwen3'
const STATUS = {
  schema: 1, run: '2026-11-24-qwen3', artifact_dir: ART, ledger: `${ART}/ledger.md`,
  updated: '2026-11-24 23:47:59', run_script: 'examples/x/run.sh', state: 'running',
  current: { step: 'step7[G1]', since: '2026-11-24 23:40:58', job: '4242 rc1' },
  steps: [{ step: 'step1', total: '0m42s' }, { step: 'step2', skipped: 'folded' }],
  reproducer: {
    tarball: `${ART}/repro-G1.tar.gz`, sha256: '77ab3035' + '0'.repeat(56), size: 90411477,
    at: '2026-11-24 23:40:32', group: 'G1', verdict: 'KERNEL_BUG',
    rate: 'REPRODUCED at concurrency 1 -- deterministic: 100/100 solo runs', host: 'c16-18',
    handover_at: '2026-11-24 23:40:57',
  },
  root_cause: null, fix: null,
  line: '2026-11-24-qwen3: repro repro-G1.tar.gz 100/100 | step7[G1] job 4242 rc1',
}

const PANE = {
  plugin: 'numeric-doctor',
  component: 'Pane',
  requestId: 'numeric-doctor',
  props: {
    title: 'Numeric Doctor', isFocused: false, bodyColumns: 70, placement: 'dock',
    scroll: { offset: 0, bodyRows: 40 }, view: {},
  },
} as const

test('the pane shows the reproducer from status.json, on every surface', async ($, on) => {
  let periods = 0
  const lines: Array<string | undefined> = []
  on('command.register', () => ({ value: { command: 'numeric-doctor' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('ui.status', (_, e) => (lines.push(e.text), { value: undefined }))
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('session.id', () => ({ value: 'sid-1234' }))
  on('session.cwd', () => ({ value: '/repo' }))
  on('fs.exists', (_, e) => ({ value: e.path === '/tmp/numeric-doctor-sessions/sid-1234' || e.path === `${ART}/status.json` }))
  on('fs.read', (_, e) => ({ value: e.path.endsWith('status.json') ? JSON.stringify(STATUS) : `${ART}\n` }))
  // one clock.every dispatch per period: the mod's poll runs when it resolves; one period, then hold
  on('clock.every', () => (periods++ === 0 ? { value: undefined } : new Promise(() => {})))

  await $.session.start({ cwd: '/repo', surface: 'terminal', isInteractive: true })
  await new Promise(r => setTimeout(r, 100))
  expect(periods).toBeGreaterThan(0)
  expect(lines.at(-1)).toBe(STATUS.line)

  for (const surface of ['terminal', 'desktop', 'vscode', 'mobile'] as const) {
    const ui = await $.ui.mount({ ...PANE, surface })
    expect((await ui.find({ type: 'Text', text: /REPRODUCER/ }))?.text).toContain('REPRODUCER')
    expect((await ui.find({ type: 'Text', text: /repro-G1\.tar\.gz/ }))?.text).toContain(ART)
    expect((await ui.find({ type: 'Text', text: /100\/100/ }))?.text).toContain('c16-18')
    expect((await ui.find({ type: 'Text', text: /step7\[G1\]/ }))?.text).toContain('4242')
    await ui.unmount()
  }
})
