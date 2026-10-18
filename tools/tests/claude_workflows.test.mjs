// Unit tests for the Claude Code workflows in .claude/workflows/.
//
// A workflow is a script body that runs with injected globals (args, agent,
// parallel, phase, log). Each test runs the real script with a scripted mock
// agent(), so argument parsing, plan/execute behavior, context threading and
// the tune-search statistics are checked without Claude or a GPU.
//
//   node --test tools/tests/

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor

// Helpers each workflow defines before it validates its arguments, exposed to
// the tests through a hook injected at the top of the body (function
// declarations are hoisted, so they exist before the script throws).
const HELPERS = {
  'mad-benchmark-sweep.js': ['tokenize', 'parseCli', 'pyLiteralToJson', 'normalizeContext'],
  'mad-tune-search.js': ['tokenize', 'parseCli', 'pyLiteralToJson', 'normalizeContext', 'stats', 'gate'],
}

function load(name) {
  const source = readFileSync(path.join(ROOT, '.claude', 'workflows', name), 'utf8')
  assert.match(source, /^export const meta = /)
  const body = source.replace(/^export const meta = /, 'const meta = ')
  const expose = `__expose(() => ({ ${HELPERS[name].join(', ')} }));\n`
  return new AsyncFunction('args', 'agent', 'parallel', 'phase', 'log', '__expose', expose + body)
}

// Run a workflow. `respond(prompt, opts)` plays every agent() call.
async function run(name, args, respond = () => null) {
  const calls = []
  const logs = []
  let helpers = null
  let parallelUsed = false
  const agent = async (prompt, opts = {}) => {
    calls.push({ prompt, opts })
    return respond(prompt, opts)
  }
  const parallel = async thunks => {
    parallelUsed = true
    return Promise.all(thunks.map(t => t().catch(() => null)))
  }
  let result, error
  try {
    result = await load(name)(args, agent, parallel, () => {}, m => logs.push(m), f => { helpers = f })
  } catch (e) {
    error = e
  }
  return { result, error, calls, logs, parallelUsed, helpers: () => helpers() }
}

const helpersOf = async name => (await run(name, undefined)).helpers()

// ── shared parsing helpers ────────────────────────────────────────────────

for (const name of Object.keys(HELPERS)) {
  test(`${name}: tokenize keeps quoted text together`, async () => {
    const { tokenize } = await helpersOf(name)
    assert.deepEqual(tokenize(`--tag x -c '{"a": "b c"}' --plan`), ['--tag', 'x', '-c', '{"a": "b c"}', '--plan'])
    assert.deepEqual(tokenize(`a "b 'c'" d`), ['a', "b 'c'", 'd'])
  })

  test(`${name}: a Python dict context becomes JSON`, async () => {
    const { normalizeContext } = await helpersOf(name)
    assert.deepEqual(
      normalizeContext(`{'docker_env_vars': {'HF_TOKEN': 'it\\'s'}, 'debug': True, 'x': None, 'n': [1, 2,],}`),
      { docker_env_vars: { HF_TOKEN: "it's" }, debug: true, x: null, n: [1, 2] })
    assert.deepEqual(normalizeContext('{"a": 1}'), { a: 1 })
    assert.equal(normalizeContext(''), null)
  })

  test(`${name}: an unparseable context fails loudly`, async () => {
    const { normalizeContext } = await helpersOf(name)
    assert.throws(() => normalizeContext("{'a': "), /additional-context/)
    assert.throws(() => normalizeContext('[1, 2]'), /must be an object/)
  })
}

// ── mad-benchmark-sweep ───────────────────────────────────────────────────

const SWEEP = 'mad-benchmark-sweep.js'

function sweepAgent(models = { dummy_multi: ['dummy/dummy_multi'] }) {
  return (prompt, opts) => {
    if (opts.label === 'resolve') {
      const tags = JSON.parse(prompt.match(/Tags: (\[.*\])/)[1])
      return {
        preflightOk: true,
        tags: tags.map(tag => ({ tag, models: models[tag] || [], error: models[tag] ? null : 'no match' })),
      }
    }
    if (opts.phase === 'Run') {
      const command = prompt.match(/Command: (.*)/)[1]
      const planned = prompt.includes('Do NOT execute')
      return { cell: opts.label, command, status: planned ? 'planned' : 'ran', performance: planned ? null : 1, metric: null }
    }
    return 'report'
  }
}

test('sweep: a tag is required', async () => {
  const { error, calls } = await run(SWEEP, '--plan')
  assert.match(error.message, /at least one tag/)
  assert.equal(calls.length, 0)
})

test('sweep: parseCli reads tags, gpu counts and options', async () => {
  const { parseCli } = await helpersOf(SWEEP)
  const cfg = parseCli(`--tags a,b c --ngpus 1,8 --timeout 60 -o out.csv -c "{'k': 1}"`)
  assert.deepEqual(cfg.tags, ['a', 'b', 'c'])
  assert.deepEqual(cfg.nGpus, ['1', '8'])
  assert.equal(cfg.timeout, '60')
  assert.equal(cfg.output, 'out.csv')
  assert.equal(cfg.additionalContext, "{'k': 1}")
  assert.equal(cfg.execute, false)
  assert.equal(parseCli('--tags a --execute').execute, true)
  assert.equal(parseCli('--tags a --execute --plan').execute, false)
})

test('sweep: plans by default and runs nothing', async () => {
  const { result, calls, parallelUsed } = await run(SWEEP, '--tags dummy_multi', sweepAgent())
  assert.equal(result.execute, false)
  assert.equal(parallelUsed, true)
  const cells = calls.filter(c => c.opts.phase === 'Run')
  assert.equal(cells.length, 1)
  assert.match(cells[0].prompt, /Do NOT execute/)
  assert.doesNotMatch(cells[0].prompt, /mad-run\.sh start/)
  assert.equal(result.cells[0].status, 'planned')
})

test('sweep: structured args also plan unless execute is true', async () => {
  const planned = await run(SWEEP, { tags: 'dummy_multi' }, sweepAgent())
  assert.equal(planned.result.execute, false)
  const executed = await run(SWEEP, { tags: ['dummy_multi'], execute: true }, sweepAgent())
  assert.equal(executed.result.execute, true)
})

test('sweep: --execute runs each cell through mad-run.sh, one at a time', async () => {
  const { result, calls, parallelUsed } = await run(SWEEP, '--tags dummy_multi --ngpus 1,2 --execute', sweepAgent())
  assert.equal(result.execute, true)
  assert.equal(parallelUsed, false)
  const cells = calls.filter(c => c.opts.phase === 'Run')
  assert.equal(cells.length, 2)
  for (const c of cells) {
    assert.match(c.prompt, /bash \.claude\/skills\/mad-common\/mad-run\.sh start sweep_/)
    assert.equal(c.opts.agentType, 'mad-benchmark-runner')
  }
})

test('sweep: the context and n_gpus reach every cell command as JSON', async () => {
  const { result } = await run(SWEEP, `--tags dummy_multi --ngpus 8 -c "{'docker_env_vars': {'HF': 'x'}}"`, sweepAgent())
  assert.match(result.cells[0].command, /--additional-context '\{"docker_env_vars":\{"HF":"x"\},"n_gpus":"8"\}'/)
})

test('sweep: an unresolved tag becomes an error cell and is not run', async () => {
  const { result, calls } = await run(SWEEP, '--tags dummy_multi,nope --execute', sweepAgent())
  const cells = calls.filter(c => c.opts.phase === 'Run')
  assert.equal(cells.length, 1)
  const bad = result.cells.find(c => c.cell === 'nope')
  assert.equal(bad.status, 'error')
  assert.match(bad.notes, /did not resolve/)
})

test('sweep: a failed pre-flight stops before any cell', async () => {
  const { error, calls } = await run(SWEEP, '--tags dummy_multi --execute',
    (p, o) => o.label === 'resolve' ? { preflightOk: false, preflightOutput: 'no madengine', tags: [] } : null)
  assert.match(error.message, /Pre-flight failed/)
  assert.equal(calls.length, 1)
})

// ── mad-tune-search ───────────────────────────────────────────────────────

const TUNE = 'mad-tune-search.js'

test('tune-search: stats and the noise gate', async () => {
  const { stats, gate } = await helpersOf(TUNE)
  const base = stats([100, 102, 98])
  assert.equal(base.median, 100)
  assert.equal(base.spreadPct, 4)
  assert.equal(stats([1, 'x', NaN]).n, 1)

  assert.equal(gate(base, stats([110, 111, 109]), 'tok/s', 'tok/s', true).pass, true)
  assert.equal(gate(base, stats([103, 103, 103]), 'tok/s', 'tok/s', true).pass, false, 'inside the spread')
  assert.equal(gate(base, stats([110]), 'tok/s', 'tok/s', true).pass, false, 'one sample')
  assert.equal(gate(base, stats([110, 111]), 'tok/s', 'ms', true).pass, false, 'metric differs')
  assert.equal(gate(base, stats([90, 91, 89]), 'ms', 'ms', false).pass, true, 'latency: lower wins')
})

test('tune-search: parseCli defaults to plan mode', async () => {
  const { parseCli } = await helpersOf(TUNE)
  const cfg = parseCli('--tag a,b --repeats 2 --expect')
  assert.equal(cfg.tag, 'a')
  assert.deepEqual(cfg._extraTags, ['b'])
  assert.equal(cfg.repeats, 2)
  assert.equal(cfg.expect, true)
  assert.equal(cfg.execute, false)
  assert.equal(parseCli('--tag a --execute').execute, true)
})

test('tune-search: a tag is required', async () => {
  const { error, calls } = await run(TUNE, '--execute')
  assert.match(error.message, /needs one model tag/)
  assert.equal(calls.length, 0)
})

function tuneAgent({ models = ['dummy/dummy_multi'], tree = () => '', base = [100, 101, 99], cand = [110, 111, 109], trustworthy = true } = {}) {
  return (prompt, opts) => {
    const label = opts.label || ''
    if (label === 'resolve') return {
      resolvedModels: models, model: models[0], scriptPath: 'run.sh', baselineSummary: 'dummy',
      family: 'inference', pytorch: true, multipleResults: null, levers: ['max-num-seqs'],
    }
    if (label === 'baseline') return { status: 'ran', samples: base, metric: 'tok/s' }
    if (label.startsWith('tree:')) return { porcelain: tree(label) }
    if (label.startsWith('eval:')) {
      const planned = prompt.includes('Do NOT execute')
      return { status: planned ? 'planned' : 'ran', samples: planned ? [] : cand, metric: 'tok/s', commands: [] }
    }
    if (label.startsWith('verify:')) return { id: label.slice(7), trustworthy }
    if (opts.phase === 'Propose') return {
      candidates: [
        { id: 'c1', change: 'max-num-seqs=512', hypothesis: 'h', evidence: 'e', depth: 'dispatch', leverKind: 'arg' },
        { id: 'c2', change: 'HIP graphs', hypothesis: 'h', evidence: 'e', depth: 'dispatch', leverKind: 'env' },
      ],
    }
    if (['Operator', 'Dispatch', 'Kernel'].includes(opts.phase)) return { status: 'ran', evidence: 'ev', bottleneck: 'compute' }
    return 'report'
  }
}

test('tune-search: a tag must resolve to exactly one model', async () => {
  const { error } = await run(TUNE, '--tag vllm --execute', tuneAgent({ models: ['vllm/a', 'vllm/b'] }))
  assert.match(error.message, /exactly one model.*resolves to 2/)
})

test('tune-search: plan mode measures nothing and never reads git status', async () => {
  const { result, calls } = await run(TUNE, '--tag dummy_multi --candidates 2', tuneAgent())
  assert.equal(result.execute, false)
  assert.equal(calls.some(c => c.opts.label === 'baseline'), false)
  assert.equal(calls.some(c => (c.opts.label || '').startsWith('tree:')), false)
  assert.deepEqual(result.candidates.map(c => c.status), ['planned', 'planned'])
  assert.equal(result.candidates.some(c => c.verdict.trustworthy), false)
})

test('tune-search: a clean measured improvement is trustworthy', async () => {
  const { result, calls } = await run(TUNE, '--tag dummy_multi --candidates 2 --execute', tuneAgent())
  assert.equal(result.baseline.median, 100)
  assert.deepEqual(result.candidates.map(c => c.verdict.trustworthy), [true, true])
  const evals = calls.filter(c => (c.opts.label || '').startsWith('eval:'))
  assert.ok(evals.every(c => c.opts.agentType === 'mad-tuner'))
  assert.match(evals[0].prompt, /mad-run\.sh start/)
})

test('tune-search: the verifier can veto a passing gate', async () => {
  const { result } = await run(TUNE, '--tag dummy_multi --candidates 1 --execute', tuneAgent({ trustworthy: false }))
  assert.equal(result.candidates[0].gate.pass, true)
  assert.equal(result.candidates[0].verdict.trustworthy, false)
})

test('tune-search: a leaked edit discards the candidate and stops the search', async () => {
  const tree = label => (label === 'tree:c1' ? ' M scripts/vllm/configs/default.yaml' : '')
  const { result, calls, logs } = await run(TUNE, '--tag dummy_multi --candidates 2 --execute', tuneAgent({ tree }))
  assert.equal(result.candidates.length, 1)
  assert.equal(result.candidates[0].status, 'error')
  assert.deepEqual(result.candidates[0].samples, [])
  assert.equal(result.candidates[0].verdict.trustworthy, false)
  assert.match(result.candidates[0].notes, /default\.yaml/)
  assert.equal(calls.some(c => c.opts.label === 'eval:c2'), false)
  assert.ok(logs.some(m => /Not evaluated: c2/.test(m)))
})

test('tune-search: a Python-dict context reaches the clean baseline commands', async () => {
  const { calls } = await run(TUNE,
    `--tag dummy_multi --candidates 1 --execute -c "{'docker_env_vars': {'HF': 'x'}, 'tools': [{'name': 'rpd'}]}"`,
    tuneAgent())
  const baseline = calls.find(c => c.opts.label === 'baseline').prompt
  assert.match(baseline, /--additional-context '\{"docker_env_vars":\{"HF":"x"\}\}'/)
  assert.doesNotMatch(baseline, /"tools"/)
})
