// Unit tests for the Claude Code workflows in .claude/workflows/.
//
// A workflow is a script body that runs with injected globals (args, agent,
// parallel, phase, log). Each test runs the real script with a scripted mock
// agent(), so argument parsing, plan/execute behavior, context threading and
// the tune-search statistics are checked without Claude or a GPU.
//
//   node --test tools/tests/*.test.mjs

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
  'mad-tune-search.js': ['tokenize', 'parseCli', 'pyLiteralToJson', 'normalizeContext', 'stats', 'gate', 'leverFits', 'regexEscape'],
  'mad-optimize.js': ['tokenize', 'parseCli', 'pyLiteralToJson', 'normalizeContext', 'stats', 'gate', 'geakTaskFor', 'kernelGroupFor'],
  'mad-nan-hunt.js': ['tokenize', 'parseCli', 'pyLiteralToJson', 'normalizeContext', 'deployTargetOf', 'nanDetected', 'reproducingGroups', 'verifierAgrees'],
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

test('sweep: a group tag fans out per model and shared models run once', async () => {
  const models = { dummies: ['dummy/a', 'dummy/b'], dummy_a: ['dummy/a'] }
  const { result, calls } = await run(SWEEP, '--tags dummies,dummy_a --execute', sweepAgent(models))
  const cells = calls.filter(c => c.opts.phase === 'Run')
  assert.equal(cells.length, 2)
  assert.match(cells[0].prompt, /--tags 'dummy\/a'/)
  assert.match(cells[1].prompt, /--tags 'dummy\/b'/)
  assert.deepEqual(result.cells.map(c => c._viaTags), [['dummies', 'dummy_a'], ['dummies']])
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

function tuneAgent({ models = ['dummy/dummy_multi'], tree = () => '', base = [100, 101, 99], cand = [110, 111, 109], trustworthy = true, card = null, proposals = null } = {}) {
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
      candidates: proposals || [
        { id: 'c1', change: 'max-num-seqs=512', hypothesis: 'h', evidence: 'e', depth: 'dispatch', leverKind: 'arg' },
        { id: 'c2', change: 'HIP graphs', hypothesis: 'h', evidence: 'e', depth: 'dispatch', leverKind: 'env' },
      ],
    }
    if (opts.phase === 'Dispatch') return { status: 'ran', evidence: 'ev', bottleneck: 'compute', hotspots: [{ name: 'triton_mm', pct: 66 }] }
    if (opts.phase === 'Kernel') return { status: 'ran', evidence: 'ev', bottleneck: 'compute', card }
    if (['Operator', 'Assembly'].includes(opts.phase)) return { status: 'ran', evidence: 'ev', bottleneck: 'compute' }
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

test('tune-search: plan mode prints all four rungs and never stacks rocm_trace_lite with rocprofv3', async () => {
  const { result, calls } = await run(TUNE, '--tag dummy_multi --candidates 1', tuneAgent())
  const phases = calls.map(c => c.opts.phase)
  for (const rung of ['Operator', 'Dispatch', 'Kernel', 'Assembly']) assert.ok(phases.includes(rung), rung)
  const dispatchPrompt = calls.find(c => c.opts.phase === 'Dispatch').prompt
  assert.match(dispatchPrompt, /context --mode warm --run-dir \/var\/tmp\/mad-kernel\/tune_dummy_multi_kernel[\s\S]*context --mode dispatch --run-dir \/var\/tmp\/mad-kernel\/tune_dummy_multi_kernel[\s\S]*kernel-tools\.py rank --run-dir \/var\/tmp\/mad-kernel\/tune_dummy_multi_kernel/)
  assert.match(dispatchPrompt, /kernel_rank\.json/)
  const kernel = calls.find(c => c.opts.phase === 'Kernel').prompt
  assert.doesNotMatch(kernel, /--mode warm/)
  assert.match(kernel, /kernel-tools\.py context --mode counters --group compute --kernel 'triton_mm'/)
  assert.match(kernel, /--additional-context-file \/var\/tmp\/mad-kernel\/tune_dummy_multi_kernel\/context\.json/)
  assert.doesNotMatch(kernel, /\$\(/)
  assert.match(kernel, /kernel-tools\.py card --run-dir \/var\/tmp\/mad-kernel\/tune_dummy_multi_kernel/)
  assert.match(calls.find(c => c.opts.phase === 'Assembly').prompt, /kernel-tools\.py asm --run-dir \/var\/tmp\/mad-kernel\/tune_dummy_multi_kernel --kernel 'triton_mm'/)
  for (const c of calls) {
    const tools = c.prompt.match(/"tools":\[[^\]]*\]/g) || []
    for (const t of tools) assert.ok(!(t.includes('rocm_trace_lite') && t.includes('rocprof')), t)
  }
  assert.equal(result.assembly.status, 'ran')
})

test('tune-search: a launch-bound dispatch skips the kernel and assembly rungs', async () => {
  const respond = tuneAgent()
  const { result } = await run(TUNE, '--tag dummy_multi --candidates 1',
    (p, o) => o.phase === 'Dispatch' ? { status: 'ran', evidence: 'gaps', bottleneck: 'launch', hotspots: [] } : respond(p, o))
  assert.equal(result.kernel.status, 'skipped')
  assert.equal(result.assembly.status, 'skipped')
})

test('tune-search: a kernel lever that does not match the card bound is dropped before measuring', async () => {
  const { leverFits } = await helpersOf(TUNE)
  assert.equal(leverFits({ depth: 'kernel', leverBound: 'bandwidth' }, 'latency'), false)
  assert.equal(leverFits({ depth: 'kernel', leverBound: 'any' }, 'latency'), true)
  assert.equal(leverFits({ depth: 'assembly', leverBound: 'latency' }, 'latency'), true)
  assert.equal(leverFits({ depth: 'dispatch' }, 'latency'), true, 'only kernel and assembly levers are filtered')
  assert.equal(leverFits({ depth: 'kernel', leverBound: 'bandwidth' }, 'unknown'), true, 'an unknown bound filters nothing')

  const card = { kernel: 'triton_mm', bound: 'latency', line: 'triton_mm: latency' }
  const proposals = [
    { id: 'vec', change: 'vectorize loads', hypothesis: 'h', evidence: 'e', depth: 'kernel', leverKind: 'env', leverBound: 'bandwidth' },
    { id: 'occ', change: 'num_warps=4', hypothesis: 'h', evidence: 'e', depth: 'kernel', leverKind: 'env', leverBound: 'latency' },
  ]
  const { result, calls, logs } = await run(TUNE, '--tag dummy_multi --candidates 2 --execute', tuneAgent({ card, proposals }))
  assert.deepEqual(result.candidates.map(c => c.id), ['occ'])
  assert.deepEqual(result.dropped.map(c => c.id), ['vec'])
  assert.equal(calls.some(c => c.opts.label === 'eval:vec'), false)
  assert.ok(logs.some(m => /Dropped candidate vec/.test(m)))
  assert.match(calls.find(c => c.opts.phase === 'Propose').prompt, /bound is "latency"/)
})

test('tune-search: the kernel regex escapes the dispatch name', async () => {
  const { regexEscape } = await helpersOf(TUNE)
  assert.equal(regexEscape('void at::native::f<4>(int)'), 'void at::native::f<4>\\(int\\)')
})

// The rule a counter run is held to, up to where its retry commands start.
const STALL_RULE = /Stall rule\.[\s\S]*?using these commands/

test('tune-search: a stalled counter run is stopped and retried once with --serialize', async () => {
  const { calls } = await run(TUNE, '--tag dummy_multi --candidates 1 --execute', tuneAgent())
  const kernel = calls.find(c => c.opts.phase === 'Kernel').prompt
  assert.match(kernel, STALL_RULE)
  assert.match(kernel, /status line says running \(not queued\) and `rocm-smi --showuse` reports 0% GPU use at every check for 10 minutes/)
  assert.match(kernel, /mad-run\.sh stop <run-id>/)
  assert.match(kernel, /Do not run the card command on the stopped run's output/)
  assert.match(kernel, /retry ONCE/)
  assert.match(kernel, /Never retry more than once/)
  assert.match(kernel, /return status "error"/)
  assert.match(kernel, /context --mode counters --group compute --kernel 'triton_mm' --serialize --run-dir \/var\/tmp\/mad-kernel\/tune_dummy_multi_kernel/)
  assert.match(kernel, /-o perf_tune_kernel_serial\.csv --additional-context-file \/var\/tmp\/mad-kernel\/tune_dummy_multi_kernel\/context\.json/)
  assert.doesNotMatch(kernel, /\$\(/)
  const [firstAttempt] = kernel.split('Stall rule.')
  assert.doesNotMatch(firstAttempt, /--serialize/, 'the first attempt is not serialized')
})

test('tune-search: plan mode prints the retry commands and runs no stall check', async () => {
  const { calls } = await run(TUNE, '--tag dummy_multi --candidates 1', tuneAgent())
  const kernel = calls.find(c => c.opts.phase === 'Kernel').prompt
  assert.match(kernel, /retry commands for a stalled counter run \(.*--serialize/)
  assert.doesNotMatch(kernel, /Stall rule\./)
})

test('tune-search: a preset run without a kernel card has no stall retry', async () => {
  const respond = tuneAgent()
  const { calls } = await run(TUNE, '--tag dummy_multi --candidates 1 --execute',
    (p, o) => o.phase === 'Dispatch' ? { status: 'ran', evidence: 'rccl', bottleneck: 'communication', hotspots: [] } : respond(p, o))
  const kernel = calls.find(c => c.opts.phase === 'Kernel').prompt
  assert.doesNotMatch(kernel, /Stall rule\./)
  assert.doesNotMatch(kernel, /--serialize/)
})

test('tune-search: a card from the serialized retry says so in the evidence', async () => {
  const card = { kernel: 'triton_mm', bound: 'latency', line: 'triton_mm: latency' }
  const respond = tuneAgent({ card })
  const kernelResult = serialized => ({ status: 'ran', evidence: 'ev', bottleneck: 'compute', card, ...(serialized ? { serialized } : {}) })
  const serial = await run(TUNE, '--tag dummy_multi --candidates 1 --execute',
    (p, o) => o.phase === 'Kernel' ? kernelResult(true) : respond(p, o))
  assert.equal(serial.result.kernel.serialized, true)
  assert.match(serial.result.kernel.evidence, /AMD_SERIALIZE_KERNEL=3 after the first counter run stalled: read them as ratios, not as a latency score/)
  assert.ok(serial.logs.some(m => /Kernel: ran \(serialized retry\)/.test(m)))
  assert.match(serial.calls.find(c => c.opts.phase === 'Propose').prompt, /AMD_SERIALIZE_KERNEL=3/)

  const plain = await run(TUNE, '--tag dummy_multi --candidates 1 --execute',
    (p, o) => o.phase === 'Kernel' ? kernelResult(false) : respond(p, o))
  assert.doesNotMatch(plain.result.kernel.evidence, /SERIALIZE/)
  assert.ok(plain.logs.every(m => !/serialized retry/.test(m)))
})

test('tune-search: a counter run that stalls twice leaves the search going without a bound', async () => {
  const respond = tuneAgent()
  const { result, calls } = await run(TUNE, '--tag dummy_multi --candidates 1 --execute',
    (p, o) => o.phase === 'Kernel' ? { status: 'error', evidence: 'stalled twice, stopped both', reason: 'GPU idle for 10 minutes on the retry' } : respond(p, o))
  assert.equal(result.kernel.status, 'error')
  assert.ok(calls.some(c => c.opts.phase === 'Assembly'), 'the code objects from the warm-up are still summarized')
  assert.match(calls.find(c => c.opts.phase === 'Propose').prompt, /bound is "unknown"/)
  assert.ok(result.candidates.length > 0)
})

// ── mad-optimize ──────────────────────────────────────────────────────────

const OPT = 'mad-optimize.js'

function optAgent(over = {}) {
  return (prompt, opts) => {
    const label = opts.label || ''
    if (label === 'resolve') return {
      preflightOk: true,
      resolvedModels: ['pyt_fam/pyt_fam_fastwam_geak'],
      model: 'pyt_fam/pyt_fam_fastwam_geak',
      dockerfile: '../../docker/pyt_fam',
      url: '',
      args: '--mode arm --arm geak',
      ...(over.resolve || {}),
    }
    if (label === 'setup') return over.setup || { status: 'ok', kernelWorkflowPresent: true, sha: 'fa86fd1e9bec56dadf27d6575a0c8f8ab7875c57' }
    if (label === 'source') return over.source || {
      status: 'ok', hasSource: true, suffixes: ['.py'],
      hostPath: '/tmp/fastwam-geak', containerPath: '/opt/faster-action-models/src/fastwam-geak',
    }
    if (label === 'baseline') return { status: 'ran', samples: [24.56, 24.70], metric: 'ms' }
    if (label === 'profile') return over.profile || { status: 'ran', topKernel: 'triton_mm 65.9%', performance: 25.2, metric: 'ms' }
    if (label === 'geak') return over.geak || { status: 'ok', expDir: '/home/me/GEAK/kernel_workflow/exp/team_1', geomean: 0.8 }
    if (label === 'patched') return over.patched || { status: 'ran', samples: [20.1, 20.2], metric: 'ms' }
    if (label === 'smoke') return { status: 'ran', performance: 1, metric: 'success_rate', baselineRate: null }
    return 'report'
  }
}

test('optimize: the latency gate matches tune-search', async () => {
  const { stats, gate } = await helpersOf(OPT)
  const narrow = stats([99.15, 100.85])
  const better = stats([97.65, 97.75])
  assert.ok(Math.abs(narrow.spreadPct - 1.7) < 1e-9)
  assert.equal(gate(narrow, better, 'ms', 'ms', false).pass, true, '2.3% improvement beats a 1.7% spread')
  const wide = stats([98.85, 101.15])
  const mild = stats([98.25, 98.35])
  assert.ok(Math.abs(wide.spreadPct - 2.3) < 1e-9)
  assert.equal(gate(wide, mild, 'ms', 'ms', false).pass, false, '1.7% improvement loses to a 2.3% spread')
})

test('optimize: plan mode prints the GEAK handoff and does not clone or run', async () => {
  const { result, calls, logs } = await run(OPT, '--tag pyt_fam_fastwam_geak --smoke-tag pyt_fam_fastwam_geak2_smoke', optAgent())
  assert.equal(result.status, 'planned')
  assert.equal(result.execute, false)
  const printed = logs.join('\n') + JSON.stringify(result)
  assert.match(printed, /kernel_workflow\.js/)
  assert.match(printed, /workflow_dir/)
  assert.match(printed, /"budget":1/)
  assert.match(printed, /git clone https:\/\/github.com\/AMD-AGI\/GEAK\.git "\$HOME\/GEAK"/)
  assert.equal(result.containerPath, '/opt/faster-action-models/src/fastwam-geak')
  assert.match(result.hostPath, /\$HOME\/\.mad-optimize\/pyt_fam_pyt_fam_fastwam_geak\/src/)
  assert.equal(calls.length, 1)
  assert.equal(calls[0].opts.label, 'resolve')
  for (const c of calls) {
    assert.doesNotMatch(c.prompt, /git clone/)
    assert.doesNotMatch(c.prompt, /madengine run/)
  }
})

test('optimize: a missing kernel_workflow.js stops before profile and GEAK', async () => {
  const { result, calls } = await run(OPT, '--tag pyt_fam_fastwam_geak --execute', optAgent({
    setup: { status: 'error', kernelWorkflowPresent: false, reason: 'kernel_workflow.js missing after clone' },
  }))
  assert.equal(result.status, 'error')
  assert.match(result.reason, /kernel_workflow\.js missing/)
  assert.equal(calls.some(c => c.opts.label === 'profile'), false)
  assert.equal(calls.some(c => c.opts.label === 'geak'), false)
})

test('optimize: no editable source stops before GEAK', async () => {
  const { result, calls } = await run(OPT, '--tag pyt_fam_fastwam_geak --execute', optAgent({
    source: { status: 'error', hasSource: false, reason: 'this image has no kernel source; pass --kernel-path' },
  }))
  assert.equal(result.status, 'error')
  assert.match(result.reason, /no kernel source/)
  assert.equal(calls.some(c => c.opts.label === 'geak'), false)
  assert.equal(calls.some(c => c.opts.label === 'baseline'), false)
})

test('optimize: an empty GEAK exp dir skips the patched score', async () => {
  const { result, calls } = await run(OPT, '--tag pyt_fam_fastwam_geak --execute', optAgent({
    geak: { status: 'error', expDir: '', geomean: 0.5, reason: 'GEAK_RUN_STATUS=failed final_patch=missing' },
  }))
  assert.equal(result.status, 'error')
  assert.match(result.reason, /GEAK failed: GEAK_RUN_STATUS=failed/, 'the GEAK agent reason is passed through')
  assert.equal(calls.some(c => c.opts.label === 'patched'), false)
  assert.equal(calls.some(c => c.opts.label === 'baseline'), true)
})

test('optimize: GEAK is launched through geak-run.sh, not the Workflow tool', async () => {
  const { calls } = await run(OPT, '--tag pyt_fam_fastwam_geak --execute', optAgent({
    profile: { status: 'ran', topKernel: 'triton_mm - 64.8%', kernelName: 'triton_mm', inSourceTree: false, performance: 25, metric: 'ms' },
    geak: { status: 'error', expDir: null, reason: 'x' },
  }))
  const g = calls.find(c => c.opts.label === 'geak')
  assert.ok(g, 'the geak agent ran')
  assert.match(g.prompt, /geak-run\.sh start opt_geak/)
  assert.match(g.prompt, /must not try to use one/)
  assert.doesNotMatch(g.prompt, /Invoke the Workflow tool/)
  assert.match(g.prompt, /eval_dir: /)
  assert.match(g.prompt, /Optimize the hottest GPU kernel "triton_mm"/)
  assert.match(g.prompt, /not defined in that tree/)
})

test('optimize: plan mode prints the kernel-card commands', async () => {
  const { result } = await run(OPT, '--tag pyt_fam_fastwam_geak', optAgent())
  assert.equal(result.kernelCommands.length, 4)
  assert.match(result.profileCommand, /--mode warm[\s\S]*-o perf_opt_warm\.csv[\s\S]*--mode dispatch[\s\S]*-o perf_opt_profile\.csv[\s\S]*kernel-tools\.py rank/)
  assert.match(result.kernelCommands[0], /kernel-tools\.py context --mode counters --group '<compute\|memory>' --kernel '<top dispatch kernel>'/)
  assert.match(result.kernelCommands[1], /madengine run .*-o perf_opt_kernel\.csv --additional-context-file \/var\/tmp\/mad-kernel\/opt_\S+_kernel\/context\.json/)
  assert.doesNotMatch(result.profileCommand, /\$\(/)
  assert.doesNotMatch(result.kernelCommands[1], /rocm_trace_lite/)
  assert.match(result.kernelCommands[3], /kernel-tools\.py asm/)
})

test('optimize: the GEAK task names the bound from the kernel card', async () => {
  const { geakTaskFor, kernelGroupFor } = await helpersOf(OPT)
  const profile = { kernelName: 'triton_mm', topKernel: 'triton_mm 66%', inSourceTree: false }
  const task = geakTaskFor(profile, { bound: 'latency', line: 'triton_mm: latency, occupancy 2 (registers)' })
  assert.match(task, /Optimize the hottest GPU kernel "triton_mm"/)
  assert.match(task, /It is latency-bound, so raise occupancy/)
  assert.match(task, /occupancy 2 \(registers\)/)
  assert.doesNotMatch(geakTaskFor(profile, { bound: 'unknown' }), /-bound, so/)
  assert.doesNotMatch(geakTaskFor(profile, null), /Kernel card/)
  assert.equal(kernelGroupFor('triton_mm'), 'compute')
  assert.equal(kernelGroupFor('triton_red_fused_add_mean'), 'memory')

  const { calls } = await run(OPT, '--tag pyt_fam_fastwam_geak --execute', optAgent({
    profile: { status: 'ran', topKernel: 'triton_mm 66%', kernelName: 'triton_mm', inSourceTree: false, performance: 25, metric: 'ms' },
    geak: { status: 'error', expDir: null, reason: 'x' },
  }))
  const k = calls.find(c => c.opts.label === 'kernel')
  assert.ok(k, 'the kernel-card agent ran')
  assert.match(k.prompt, /--group compute --kernel 'triton_mm'/)
})

test('optimize: plan mode gives GEAK a fixed eval_dir', async () => {
  const { result } = await run(OPT, '--tag pyt_fam_fastwam_geak', optAgent())
  assert.equal(result.status, 'planned')
  assert.match(result.geakInvocation.args.eval_dir, /geak_eval$/)
})

const OPT_PROFILE = { status: 'ran', topKernel: 'triton_mm 66%', kernelName: 'triton_mm', inSourceTree: false, performance: 25, metric: 'ms' }
const OPT_CARD = { kernel: 'triton_mm', bound: 'latency', line: 'triton_mm: latency' }

function optKernelAgent(kernelResult, over = {}) {
  const base = optAgent({ profile: OPT_PROFILE, geak: { status: 'error', expDir: null, reason: 'x' }, ...over })
  return (prompt, opts) => opts.label === 'kernel' ? kernelResult : base(prompt, opts)
}

test('optimize: plan mode prints the serialize retry beside the counter commands', async () => {
  const { result, logs } = await run(OPT, '--tag pyt_fam_fastwam_geak', optAgent())
  assert.equal(result.kernelCommands.length, 4)
  assert.doesNotMatch(result.kernelCommands.join('\n'), /--serialize/)
  assert.equal(result.kernelRetryCommands.length, 4)
  assert.match(result.kernelRetryCommands[0], /kernel-tools\.py context --mode counters --group '<compute\|memory>' --kernel '<top dispatch kernel>' --serialize --run-dir/)
  assert.match(result.kernelRetryCommands[1], /-o perf_opt_kernel_serial\.csv --additional-context-file \/var\/tmp\/mad-kernel\/opt_\S+_kernel\/context\.json/)
  assert.match(result.kernelRetryCommands[2], /kernel-tools\.py card/)
  assert.ok(logs.some(m => /Kernel card retry, only if the counter run stalls: .*--serialize/.test(m)))
})

test('optimize: a stalled counter run is stopped and retried once with --serialize', async () => {
  const { calls } = await run(OPT, '--tag pyt_fam_fastwam_geak --execute', optKernelAgent('report'))
  const kernel = calls.find(c => c.opts.label === 'kernel').prompt
  assert.match(kernel, STALL_RULE)
  assert.match(kernel, /status line says running \(not queued\) and `rocm-smi --showuse` reports 0% GPU use at every check for 10 minutes/)
  assert.match(kernel, /mad-run\.sh stop <run-id>/)
  assert.match(kernel, /retry ONCE/)
  assert.match(kernel, /fresh run id such as opt_kernel_serial/)
  assert.match(kernel, /--group compute --kernel 'triton_mm' --serialize --run-dir/)
  assert.match(kernel, /-o perf_opt_kernel_serial\.csv/)
  assert.match(kernel, /Never take a card from a stalled run you stopped/)
  assert.doesNotMatch(kernel, /\$\(/)
  const [firstAttempt] = kernel.split('Stall rule.')
  assert.doesNotMatch(firstAttempt, /--serialize/, 'the first attempt is not serialized')
})

test('optimize: a card from the serialized retry carries the note into the GEAK task', async () => {
  const { calls, logs } = await run(OPT, '--tag pyt_fam_fastwam_geak --execute',
    optKernelAgent({ status: 'ran', serialized: true, card: OPT_CARD }))
  const geak = calls.find(c => c.opts.label === 'geak').prompt
  assert.match(geak, /Kernel card: triton_mm: latency \(counters from the retry with AMD_SERIALIZE_KERNEL=3 after a stall: ratios, not a latency score\)/)
  assert.ok(logs.some(m => /Kernel card: triton_mm: latency \(counters from the retry/.test(m)))

  const plain = await run(OPT, '--tag pyt_fam_fastwam_geak --execute',
    optKernelAgent({ status: 'ran', card: OPT_CARD }))
  assert.doesNotMatch(plain.calls.find(c => c.opts.label === 'geak').prompt, /SERIALIZE/)
})

test('optimize: a counter run that stalls twice leaves GEAK without a kernel card', async () => {
  const { calls, logs } = await run(OPT, '--tag pyt_fam_fastwam_geak --execute',
    optKernelAgent({ status: 'error', notes: 'stalled on both attempts' }))
  assert.ok(logs.some(m => /Kernel card: none \(stalled on both attempts\)/.test(m)))
  assert.doesNotMatch(calls.find(c => c.opts.label === 'geak').prompt, /Kernel card:/)
})

test('both workflows hold a counter run to the same stall rule', async () => {
  const [tune, opt] = await Promise.all([
    run(TUNE, '--tag dummy_multi --candidates 1 --execute', tuneAgent()),
    run(OPT, '--tag pyt_fam_fastwam_geak --execute', optKernelAgent('report')),
  ])
  const rule = r => r.calls.find(c => c.opts.phase === 'Kernel' || c.opts.label === 'kernel').prompt.match(STALL_RULE)[0]
  assert.equal(rule(tune).replace(/such as \S+,/, ''), rule(opt).replace(/such as \S+,/, ''))
})

// ── mad-nan-hunt ──────────────────────────────────────────────────────────

const NAN = 'mad-nan-hunt.js'

function nanAgent(over = {}) {
  return (prompt, opts) => {
    const label = opts.label || ''
    if (label === 'resolve') return over.resolve || { preflightOk: true, resolvedModels: ['dummy/dummy_nan'], model: 'dummy/dummy_nan', nGpus: '1', hostGpus: 8 }
    if (label === 'setup') return over.setup || { status: 'ok', head: 'x', intakeValid: true, intakeOutput: 'RUN_SCRIPT=...' }
    if (label.startsWith('detect-')) return over.detect || { runId: label, status: 'failed', signalLines: 2, firstMatch: 'step 20 loss: nan' }
    if (label.startsWith('capture-')) {
      const k = Number(label.split('-')[1])
      if (over.capture) return typeof over.capture === 'function' ? over.capture(k) : over.capture
      return { runId: `nan_capture_${k}`, status: 'failed', capsules: 1, manifest: false, capsuleDir: '/a/capsules/nan_capture_1', image: 'ci-dummy_nan' }
    }
    if (label === 'replay') return over.replay || { groups: [{ group: 'G1', capsules: 1, verdict: 'FP32_OVERFLOW?', hits: 10, runs: 10, concurrency: 2, logDir: '/a/5b/G1' }] }
    if (label === 'package') return over.package || { status: 'ok', tarball: '/a/repro-G1.tar.gz', sha256: 'abc', reproduced: '10/10' }
    if (label === 'rootcause') return { status: 'ok', rootCauseFile: '/a/rootcause/ROOT_CAUSE.md', summary: 's' }
    if (label === 'verify') return over.verify || { checks: [{ group: 'G1', hitsAgree: true, verdictAgree: true }] }
    if (label === 'report') return { reportPath: '/a/report.md', copied: [] }
    return null
  }
}

test('nan-hunt: decisions are made in code', async () => {
  const { nanDetected, reproducingGroups, verifierAgrees, deployTargetOf } = await helpersOf(NAN)
  assert.equal(nanDetected([{ signalLines: 0 }, { signalLines: 3 }, null]), 1)
  assert.deepEqual(reproducingGroups([{ group: 'G1', hits: 0 }, { group: 'G2', hits: 4 }]).map(g => g.group), ['G2'])
  const groups = [{ group: 'G1' }, { group: 'G2' }]
  assert.equal(verifierAgrees(groups, [{ group: 'G1', hitsAgree: true, verdictAgree: true }]), false, 'a group the verifier skipped is not verified')
  assert.equal(verifierAgrees(groups, groups.map(g => ({ ...g, hitsAgree: true, verdictAgree: true }))), true)
  assert.equal(deployTargetOf({ slurm: { nodes: 2 } }), 'slurm')
  assert.equal(deployTargetOf({ k8s: {} }), 'k8s')
  assert.equal(deployTargetOf(null), 'local')
})

test('nan-hunt: parseCli defaults to plan mode and reads its options', async () => {
  const { parseCli } = await helpersOf(NAN)
  const cfg = parseCli(`--tag dummy_nan --nan-doctor-home /x/nd --repeats 3 --capture-data values --groups all --rootcause --budget-minutes 30 -c '{"a": 1}'`)
  assert.equal(cfg.execute, false)
  assert.equal(cfg.tag, 'dummy_nan')
  assert.equal(cfg.nanDoctorHome, '/x/nd')
  assert.equal(cfg.repeats, 3)
  assert.equal(cfg.captureData, 'values')
  assert.equal(cfg.groups, 'all')
  assert.equal(cfg.rootcause, true)
  assert.equal(cfg.budgetMinutes, 30)
  assert.equal(parseCli('--tag x --execute').execute, true)
})

test('nan-hunt: a tag is required, one model only, no tools, no k8s', async () => {
  assert.match((await run(NAN, '--execute')).error.message, /needs one model tag/)
  assert.match((await run(NAN, '--tag a,b')).error.message, /exactly one tag/)
  assert.match((await run(NAN, `--tag x -c '{"tools": [{"name": "rpd"}]}'`)).error.message, /must not carry "tools"/)
  assert.match((await run(NAN, `--tag x -c '{"k8s": {}}'`)).error.message, /K8s is not supported/)
  const { error } = await run(NAN, '--tag dummies', nanAgent({ resolve: { preflightOk: true, resolvedModels: ['a', 'b'], model: '' } }))
  assert.match(error.message, /resolves to 2/)
})

test('nan-hunt: plan mode prints the clone, the intake and the runs, and runs nothing', async () => {
  const { result, calls, logs } = await run(NAN, '--tag dummy_nan --repeats 2', nanAgent())
  assert.equal(result.status, 'planned')
  assert.equal(calls.length, 1)
  assert.equal(calls[0].opts.label, 'resolve')
  assert.equal(calls[0].opts.agentType, 'mad-nan-doctor')
  const printed = logs.join('\n')
  assert.match(printed, /git clone https:\/\/github.com\/\S+\/nan-doctor\.git "\$HOME\/nan-doctor" && git -C "\$HOME\/nan-doctor" checkout [0-9a-f]{40}/)
  assert.equal(result.detectCommands.length, 2)
  assert.doesNotMatch(result.detectCommands[0], /nan_doctor/, 'clean runs carry no tool')
  assert.match(result.captureCommand, /"tools":\[\{"name":"nan_doctor"\}\]/)
  assert.match(result.artifactDir, /^\$HOME\/nan-doctor\/artifacts\/mad-\d{4}-\d{2}-\d{2}-dummy_dummy_nan$/)
  assert.equal(result.intake.env, 'madengine')
  assert.equal(result.intake.world_size, 1)
  assert.equal(result.intake.preferences.fix, false)
  assert.match(result.intake.run, /^madengine run --tags dummy\/dummy_nan --live-output$/)
})

test('nan-hunt: capture data mode and the user context reach the capture run', async () => {
  const { result } = await run(NAN, `--tag dummy_nan --capture-data values -c "{'docker_env_vars': {'A': '1'}}"`, nanAgent())
  const ctx = JSON.parse(result.captureCommand.match(/--additional-context '(.*)'$/)[1])
  assert.deepEqual(ctx.docker_env_vars, { A: '1', NAN_CAPTURE_DATA: '1' })
  assert.deepEqual(ctx.tools, [{ name: 'nan_doctor' }])
  assert.match(result.detectCommands[0], /"docker_env_vars":\{"A":"1"\}/)
})

test('nan-hunt: a full execute run ends with a verified reproducer', async () => {
  const { result, calls } = await run(NAN, '--tag dummy_nan --execute', nanAgent())
  assert.equal(result.status, 'reproducer')
  assert.equal(result.verified, true)
  assert.deepEqual(calls.map(c => c.opts.label), ['resolve', 'setup', 'detect-1', 'capture-1', 'replay', 'package', 'verify', 'report'])
  assert.equal(calls.find(c => c.opts.label === 'verify').opts.agentType, 'general-purpose')
  for (const c of calls.filter(c => /^(detect|capture)/.test(c.opts.label))) assert.match(c.prompt, /mad-run\.sh start/)
  assert.match(calls.find(c => c.opts.label === 'replay').prompt, /nan-doctor-run\.sh start <run-id> \S+ replay\|node5b/)
  assert.equal(calls.some(c => c.opts.label === 'rootcause'), false)
})

test('nan-hunt: no NaN in the clean runs stops before capture', async () => {
  const { result, calls } = await run(NAN, '--tag dummy_nan --execute --repeats 2', nanAgent({ detect: { runId: 'r', status: 'succeeded', signalLines: 0 } }))
  assert.equal(result.status, 'no-nan')
  assert.equal(calls.filter(c => c.opts.label.startsWith('detect-')).length, 2)
  assert.equal(calls.some(c => c.opts.label.startsWith('capture-')), false)
})

test('nan-hunt: --skip-detect goes straight to capture', async () => {
  const { calls } = await run(NAN, '--tag dummy_nan --execute --skip-detect', nanAgent())
  assert.equal(calls.some(c => c.opts.label.startsWith('detect-')), false)
  assert.equal(calls.some(c => c.opts.label === 'capture-1'), true)
})

test('nan-hunt: an empty capture is retried once, then stops', async () => {
  const empty = { runId: 'c', status: 'failed', capsules: 0, capsuleDir: '' }
  let r = await run(NAN, '--tag dummy_nan --execute', nanAgent({ capture: () => empty }))
  assert.equal(r.result.status, 'no-capsule')
  assert.equal(r.calls.filter(c => c.opts.label.startsWith('capture-')).length, 2)
  assert.equal(r.calls.some(c => c.opts.label === 'replay'), false)
  r = await run(NAN, '--tag dummy_nan --execute', nanAgent({
    capture: k => (k === 1 ? empty : { runId: 'c2', status: 'failed', capsules: 3, capsuleDir: '/a/c2', image: 'i' }),
  }))
  assert.equal(r.result.status, 'reproducer')
})

test('nan-hunt: zero hits means not reproduced, whatever the agent called it', async () => {
  const { result, calls } = await run(NAN, '--tag dummy_nan --execute', nanAgent({
    replay: { groups: [{ group: 'G1', verdict: 'KERNEL_BUG', hits: 0, runs: 5000 }] },
  }))
  assert.equal(result.status, 'not-reproduced')
  assert.equal(calls.some(c => c.opts.label === 'package'), false)
})

test('nan-hunt: a disagreeing verifier marks the reproducer unverified', async () => {
  const { result } = await run(NAN, '--tag dummy_nan --execute', nanAgent({
    verify: { checks: [{ group: 'G1', hitsAgree: false, verdictAgree: true, note: 'summary says 3/10' }] },
  }))
  assert.equal(result.status, 'reproducer-unverified')
  assert.equal(result.verified, false)
})

test('nan-hunt: --rootcause runs step 7 within the budget and never step 8', async () => {
  const { calls } = await run(NAN, '--tag dummy_nan --execute --rootcause --budget-minutes 45', nanAgent())
  const rc = calls.find(c => c.opts.label === 'rootcause')
  assert.ok(rc)
  assert.match(rc.prompt, /45 minutes/)
  assert.match(rc.prompt, /no step 8/)
})
