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
