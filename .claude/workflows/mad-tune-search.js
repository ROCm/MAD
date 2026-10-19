export const meta = {
  name: 'mad-tune-search',
  description: 'Descend operator, dispatch, and kernel profiling for a training or inference MAD model, then measure clean candidates and synthesize the winning config',
  phases: [
    { title: 'Baseline', detail: 'resolve exactly one model, classify it, and measure a clean baseline (N runs)' },
    { title: 'Operator', detail: 'TraceLens on a PyTorch Kineto trace, when the model is PyTorch' },
    { title: 'Dispatch', detail: 'rocm-trace-lite kernel-dispatch timeline' },
    { title: 'Kernel', detail: 'one rocprofv3 counter preset chosen from the dispatch class' },
    { title: 'Expect', detail: 'Primus projection numbers only when --expect is passed' },
    { title: 'Propose', detail: 'one-lever candidates citing the deepest evidence' },
    { title: 'Evaluate', detail: 'measure each candidate cleanly (sequential, N runs) and verify its delta' },
    { title: 'Synthesize', detail: 'pick the winning configuration' },
  ],
}

// ARGS — accepts EITHER:
//   (a) a structured object: { tag, target?, candidates?, execute?, repeats?,
//       additionalContext?, timeout?, depth?, profileTool?, expect? }
//   (b) a CLI-style string, e.g.
//       --tag pyt_vllm_qwen3-8b --target latency --depth auto --candidates 4 --execute
//       --tag primus_train/megatron_MI355X_llama3.1_8B-BF16-pretrain --expect
//
// The tag is required and must resolve (madengine discover) to exactly ONE model.
//
// THREE DEPTHS, ONE MODEL AT A TIME. madengine is the harness. Each depth is its
// own run. Never put rocm_trace_lite and rocprofv3_* in the same tools array.
//   - operator: torch_profiler_dynolog + tracelens (TRACELENS_MODE=pytorch).
//     Training counts optimizer.step(). Inference sets TORCH_PROFILE_ITERATIONS=0.
//     Skip when the model is not PyTorch.
//   - dispatch: rocm_trace_lite. Any HIP workload, training or inference.
//   - kernel: rocprofv3_compute | rocprofv3_memory | rocprofv3_communication,
//     chosen from the dispatch class. Skipped on auto when dispatch is launch-bound.
//
// CLEAN vs PROFILED: baseline and every candidate measurement strip `tools`.
// A profiled perf number is evidence, not the score.
//
// REPEATS: the baseline and each candidate are measured --repeats times
// (default 3, 1..5). Medians, spread, and deltas are computed here, not by an
// agent. A candidate is trustworthy only when both sides have >= 2 samples, the
// metric matches, the improvement beats the run-to-run spread (at least 1%), and
// the adversarial verifier agrees.
//
// EXPECT is OFF unless the user passes --expect / --expect auto. Projection only
// scores Primus training configs. Inference and every other trainer skip it even
// then. --expect off and --no-expect force it off.
//
// --depth operator|dispatch|kernel runs that rung only. --profile-tool NAME forces
// the matching rung and uses NAME as its tool. Default --depth auto descends.
//
// EXECUTE: plans by default — resolves the model, prints every command, and
// ranks candidates without touching GPUs. --execute (or execute:true) runs them;
// --plan / --no-execute / --dry-run force plan mode.
// Every run goes through .claude/skills/mad-common/mad-run.sh, which detaches it
// (a tool call ends after 10 minutes) and holds a host-wide GPU lock. Candidates
// are measured one at a time so GPU runs and file edits do not race.
//
// CLEAN TREE: candidates prefer levers that need no file edit (docker_env_vars,
// --tags <model>:<arg>=<value>). When one edits a tracked file it must revert
// it. After each executed candidate a separate agent reads `git status` and the
// workflow compares it with the snapshot taken before the first candidate; on
// any difference that candidate is an error and the search stops, so a leaked
// edit can never contaminate the next measurement.

function shellQuote(s) {
  return "'" + String(s).replace(/'/g, "'\\''") + "'"
}

function tokenize(s) {
  const out = []
  let cur = '', q = null
  for (let i = 0; i < s.length; i++) {
    const c = s[i]
    if (q) {
      if (c === q) q = null
      else cur += c
    } else if (c === '"' || c === "'") {
      q = c
    } else if (c === ' ' || c === '\t' || c === '\n') {
      if (cur) { out.push(cur); cur = '' }
    } else {
      cur += c
    }
  }
  if (cur) out.push(cur)
  return out
}

// Rewrite a Python dict literal (single quotes, True/False/None, trailing
// commas) as JSON. madengine accepts both forms; the workflow needs JSON so it
// can add a tools array or env vars without losing the user's keys.
function pyLiteralToJson(s) {
  let out = '', i = 0
  while (i < s.length) {
    const c = s[i]
    if (c === "'" || c === '"') {
      let str = ''
      i++
      while (i < s.length && s[i] !== c) {
        if (s[i] === '\\' && i + 1 < s.length) {
          const n = s[i + 1]
          str += (n === "'" ? "'" : n === '"' ? '"' : '\\' + n)
          i += 2
        } else {
          str += s[i++]
        }
      }
      if (i >= s.length) throw new Error('unterminated string')
      i++
      out += JSON.stringify(str.replace(/\\(.)/g, (m, n) => JSON.parse('"\\' + n + '"')))
    } else if (/[A-Za-z_]/.test(c)) {
      let w = ''
      while (i < s.length && /[A-Za-z0-9_]/.test(s[i])) w += s[i++]
      out += ({ True: 'true', False: 'false', None: 'null' })[w] || w
    } else {
      out += c
      i++
    }
  }
  return out.replace(/,(\s*[}\]])/g, '$1')
}

// Parse --additional-context into a plain object, or throw. Never silently drop
// it: it carries docker_env_vars such as the HF token.
function normalizeContext(raw) {
  if (raw == null || raw === '') return null
  if (typeof raw === 'object' && !Array.isArray(raw)) return raw
  let obj
  try { obj = JSON.parse(raw) } catch (e) {
    try { obj = JSON.parse(pyLiteralToJson(String(raw))) } catch (e2) {
      throw new Error(`--additional-context must be a JSON object or a Python dict literal (${e2.message}): ${raw}`)
    }
  }
  if (!obj || typeof obj !== 'object' || Array.isArray(obj))
    throw new Error(`--additional-context must be an object, got: ${raw}`)
  return obj
}

function parseExpect(v) {
  if (v == null || v === true || v === 'auto' || v === 'on' || v === '') return true
  if (v === false || v === 'off' || v === 'false' || v === '0') return false
  return true
}

function parseCli(s) {
  const toks = tokenize(s)
  const cfg = { execute: false, expect: false, depth: 'auto' }
  const tags = []
  for (let i = 0; i < toks.length; i++) {
    const t = toks[i]
    const next = () => toks[++i]
    if (t === '--tags' || t === '--tag' || t === '-t') {
      while (i + 1 < toks.length && !toks[i + 1].startsWith('-')) {
        next().split(',').forEach(x => { if (x) tags.push(x) })
      }
    } else if (t === '--additional-context' || t === '-c') {
      cfg.additionalContext = next()
    } else if (t === '--target') {
      cfg.target = next()
    } else if (t === '--candidates' || t === '-n') {
      cfg.candidates = parseInt(next(), 10)
    } else if (t === '--repeats') {
      cfg.repeats = parseInt(next(), 10)
    } else if (t === '--profile-tool') {
      cfg.profileTool = next()
    } else if (t === '--depth') {
      cfg.depth = next()
    } else if (t === '--timeout') {
      cfg.timeout = next()
    } else if (t === '--expect') {
      const v = (i + 1 < toks.length && !toks[i + 1].startsWith('-')) ? next() : 'auto'
      cfg.expect = parseExpect(v)
    } else if (t === '--no-expect') {
      cfg.expect = false
    } else if (t === '--no-execute' || t === '--plan' || t === '--dry-run') {
      cfg.execute = false
    } else if (t === '--execute') {
      cfg.execute = true
    }
  }
  if (tags.length) cfg.tag = tags[0]
  if (tags.length > 1) cfg._extraTags = tags.slice(1)
  return cfg
}

// --- deterministic statistics (never delegated to an agent) ---
function median(xs) {
  const s = [...xs].sort((a, b) => a - b)
  const m = Math.floor(s.length / 2)
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}
function stats(samples) {
  const xs = (samples || []).filter(x => typeof x === 'number' && isFinite(x))
  if (!xs.length) return { n: 0, median: null, spreadPct: null }
  const med = median(xs)
  const spreadPct = xs.length > 1 && med ? (Math.max(...xs) - Math.min(...xs)) / Math.abs(med) * 100 : null
  return { n: xs.length, median: med, spreadPct }
}
const MIN_NOISE_PCT = 1
function gate(base, cand, baseMetric, candMetric, higherIsBetter) {
  if (base.n < 2 || cand.n < 2)
    return { pass: false, deltaPct: null, noisePct: null, reason: `needs >= 2 samples on both sides (baseline ${base.n}, candidate ${cand.n})` }
  if (baseMetric && candMetric && baseMetric.trim().toLowerCase() !== candMetric.trim().toLowerCase())
    return { pass: false, deltaPct: null, noisePct: null, reason: `metric differs (baseline "${baseMetric}", candidate "${candMetric}")` }
  const deltaPct = (cand.median - base.median) / Math.abs(base.median) * 100
  const improvementPct = higherIsBetter ? deltaPct : -deltaPct
  const noisePct = Math.max(base.spreadPct || 0, cand.spreadPct || 0, MIN_NOISE_PCT)
  const pass = improvementPct > noisePct
  return {
    pass, deltaPct, noisePct,
    reason: pass
      ? `improves ${improvementPct.toFixed(2)}% > noise ${noisePct.toFixed(2)}%`
      : `improvement ${improvementPct.toFixed(2)}% does not beat noise ${noisePct.toFixed(2)}%`,
  }
}

let cfg
if (typeof args === 'string') {
  cfg = parseCli(args)
} else if (args && typeof args === 'object') {
  cfg = { execute: false, expect: false, depth: 'auto', ...args }
  if (Object.prototype.hasOwnProperty.call(args, 'expect')) cfg.expect = parseExpect(args.expect)
} else {
  cfg = { execute: false, expect: false, depth: 'auto' }
}

if (!cfg.tag)
  throw new Error('mad-tune-search needs one model tag, e.g. --tag pyt_vllm_qwen3-8b [--target latency] [--execute]')
const DEPTHS = ['auto', 'operator', 'dispatch', 'kernel']
const tag = cfg.tag
const target = cfg.target || 'throughput'
const nCandidates = Math.max(1, Math.min(cfg.candidates || 4, 8))
const repeats = Math.max(1, Math.min(cfg.repeats || 3, 5))
const execute = cfg.execute === true
const timeout = cfg.timeout || null
const expectOn = cfg.expect === true
let depth = DEPTHS.includes(cfg.depth) ? cfg.depth : 'auto'
const profileTool = cfg.profileTool || null

function depthForTool(name) {
  if (!name) return null
  if (name === 'rocm_trace_lite' || name === 'rocm_trace_lite_default' || name === 'rpd') return 'dispatch'
  if (String(name).startsWith('rocprof')) return 'kernel'
  if (name === 'torch_profiler_dynolog' || String(name).startsWith('tracelens')) return 'operator'
  return null
}
if (profileTool) {
  const forced = depthForTool(profileTool)
  if (depth === 'auto' && forced) depth = forced
  else if (forced && depth !== 'auto' && forced !== depth)
    log(`Note: --profile-tool ${profileTool} is a ${forced} tool; --depth ${depth} still selects the rung. A tool is applied only on its own rung.`)
}

const ctxObj = normalizeContext(cfg.additionalContext)
const baseObj = ctxObj ? { ...ctxObj } : {}
delete baseObj.tools
const cleanCtx = Object.keys(baseObj).length ? JSON.stringify(baseObj) : null
if (ctxObj && Array.isArray(ctxObj.tools) && ctxObj.tools.length)
  log('Note: tools in --additional-context are not used. Each depth builds its own tools array. Pass --depth or --profile-tool to force one rung.')

function toolsPayload(tools) {
  const names = tools.map(t => t.name)
  const hasRtl = names.some(n => String(n).startsWith('rocm_trace_lite'))
  const hasRocprof = names.some(n => String(n).startsWith('rocprof'))
  if (hasRtl && hasRocprof)
    throw new Error('refusing to stack rocm_trace_lite with rocprofv3 in one madengine run')
  return JSON.stringify({ ...baseObj, tools })
}
function ctxFlag(payload) {
  return payload ? ` --additional-context ${shellQuote(payload)}` : ''
}
function runCmd(outFile, payload) {
  const flags = ['--tags ' + shellQuote(tag), '--live-output', '-o ' + outFile]
  if (timeout) flags.push('--timeout ' + timeout)
  return `madengine run ${flags.join(' ')}${ctxFlag(payload)}`
}
function want(rung) {
  return depth === 'auto' || depth === rung
}
function skipped(reason) {
  return {
    status: 'skipped',
    reason,
    bottleneck: 'unknown',
    bound: 'unknown',
    evidence: reason,
    hotspots: [],
    topOperators: [],
    tracePath: null,
    preset: null,
  }
}

const KERNEL_PRESET = {
  compute: 'rocprofv3_compute',
  memory: 'rocprofv3_memory',
  communication: 'rocprofv3_communication',
  mixed: 'rocprofv3_compute',
}

if (cfg._extraTags && cfg._extraTags.length)
  log(`Note: mad-tune-search tunes ONE model. Using "${tag}"; ignoring extra tags [${cfg._extraTags.join(', ')}]. Use mad-benchmark-sweep to compare multiple models.`)
if (execute && repeats < 2)
  log('Note: --repeats 1 takes one sample per configuration, so no candidate can be marked trustworthy. Use --repeats 2 or more for a recommendation.')
log(`Tuning "${tag}" for ${target}. depth=${depth}. repeats=${repeats}. expect=${expectOn}. execute=${execute}. ${cleanCtx ? 'clean context ready' : 'no additional-context'}.`)
if (!execute) log('Plan mode: commands are printed and candidates ranked, nothing runs. Pass --execute to measure on this host.')

const PREFLIGHT = 'Pre-flight: run `bash .claude/skills/mad-common/preflight.sh` and stop if it exits non-zero.'
const MAD_RUN = 'bash .claude/skills/mad-common/mad-run.sh'
const RUN_HOW = `Run every madengine command through the run helper, never in the foreground (a tool call ends after 10 minutes):
  ${MAD_RUN} start <run-id> -- <madengine command>
then repeat \`${MAD_RUN} wait <run-id>\` until its first line says MAD_RUN_STATUS=succeeded or failed. Use a fresh run-id per run (append -2, -3, ... if start says it exists).`

// ── Phase 1: Baseline (resolve + classify, then CLEAN measurement) ─────────
phase('Baseline')
const BASELINE_SCHEMA = {
  type: 'object',
  required: ['resolvedModels', 'model', 'scriptPath', 'baselineSummary', 'family', 'pytorch'],
  properties: {
    resolvedModels: { type: 'array', items: { type: 'string' } },
    model: { type: 'string' },
    scriptPath: { type: 'string' },
    baselineSummary: { type: 'string' },
    multipleResults: { type: ['string', 'null'] },
    family: { type: 'string', enum: ['training', 'inference', 'other'] },
    pytorch: { type: 'boolean' },
    primusConfig: { type: ['string', 'null'] },
    framework: { type: ['string', 'null'] },
    levers: { type: 'array', items: { type: 'string' } },
  },
}
const baseline = await agent(
  `In the MAD repo, resolve and classify tag "${tag}" for tuning (target: ${target}). Do not run any benchmark.
${PREFLIGHT}
Run \`madengine discover --tags ${shellQuote(tag)} --json\` and put the "name" of every model it lists in resolvedModels.
If it lists exactly one model, read that model's registry entry — scripts/<dir>/models.json (paths in it are
relative to that directory) or, for a generated model, the directory's get_models_json.py — plus its run script,
dockerfile, tags, and args. Set model to the discovered name.
Classify:
- family "training" when tags/scripts/dockerfile indicate pretrain, finetune, primus, megatron, torchtitan, maxtext, or torchtune.
- family "inference" when they indicate vllm, sglang, inference, or serving.
- family "other" otherwise. A Primus training config is training even if the model name is familiar from inference.
- pytorch true for PyTorch, vLLM, SGLang, Torchtune, and Primus Megatron/TorchTitan. pytorch false for JAX/MaxText and other non-PyTorch stacks.
- primusConfig: the --config_path value when discover shows one under scripts/Primus, else null.
- framework: modules.pre_trainer.framework or modules.post_trainer.framework from that YAML when you can read it, else null.
List levers that actually exist for this family. Training: batch, precision, TP/PP/EP, RCCL/NCCL, recompute, PYTORCH_TUNABLEOP_ENABLED. Inference: gpu-memory-utilization, max-num-seqs, max-num-batched-tokens, tensor-parallel, precision, HIP graphs, attention backend.
Report whether the entry sets "multiple_results" (its CSV filename, or null).`,
  { label: 'resolve', phase: 'Baseline', schema: BASELINE_SCHEMA, agentType: 'mad-benchmark-runner' }
)
if (!baseline) throw new Error(`Baseline step returned no result for "${tag}"; nothing was measured.`)
const resolvedModels = baseline.resolvedModels || []
if (resolvedModels.length !== 1)
  throw new Error(`mad-tune-search tunes exactly one model, but "${tag}" resolves to ${resolvedModels.length}: [${resolvedModels.join(', ')}]. Pass one model name (e.g. ${resolvedModels[0] || 'dir/model'}).`)

const multiCsv = baseline.multipleResults || null
const parseHint = multiCsv
  ? `This is a multiple_results model — performance is NOT a stdout line; after each run read the per-metric CSV "${multiCsv}" (and the -o file) and take the primary ${target} metric.`
  : 'Parse the "performance: <value> <unit>" line from each run\'s output. Ignore perf numbers taken under a profiler.'
const higherIsBetter = target !== 'latency'

const SAMPLES_SCHEMA = {
  type: 'object',
  required: ['status', 'samples'],
  properties: {
    status: { type: 'string', enum: ['planned', 'ran', 'skipped', 'error'] },
    samples: { type: 'array', items: { type: 'number' } },
    metric: { type: ['string', 'null'] },
    commands: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
}
function sampleOutFiles(prefix) {
  return Array.from({ length: repeats }, (_, k) => `${prefix}_r${k + 1}.csv`)
}
const baseOuts = sampleOutFiles('perf_tune_baseline')
let baseRuns
if (execute) {
  baseRuns = await agent(
    `Measure a CLEAN baseline for "${baseline.model}" (profiler OFF, no tools key): ${repeats} sequential run(s), one per command below. Do not reuse an old perf.csv row.
${baseOuts.map(o => '  ' + runCmd(o, cleanCtx)).join('\n')}
${RUN_HOW}
If no AMD GPUs are present (rocm-smi/amd-smi), set status "skipped" with no samples. ${parseHint}
Return status, one number per successful run in samples (same order), the metric unit, and the commands you ran.`,
    { label: 'baseline', phase: 'Baseline', schema: SAMPLES_SCHEMA, agentType: 'mad-benchmark-runner' }
  )
}
if (!baseRuns) baseRuns = { status: execute ? 'error' : 'planned', samples: [], metric: null, commands: baseOuts.map(o => runCmd(o, cleanCtx)), notes: execute ? 'baseline agent returned no result' : 'plan mode' }
const baseStats = stats(baseRuns.samples)
baseline.baselinePerf = baseStats.median
baseline.baselineMetric = baseRuns.metric || null
log(`Baseline: ${baseRuns.status}. ${baseStats.n ? `median ${baseStats.median} ${baseRuns.metric || ''} over ${baseStats.n} run(s)${baseStats.spreadPct != null ? `, spread ${baseStats.spreadPct.toFixed(2)}%` : ''}` : 'no samples'}`)

function operatorTools() {
  const env = baseline.family === 'inference'
    ? { TORCH_PROFILE_ITERATIONS: '0', TORCH_PROFILE_DURATION_MS: '500', TORCH_PROFILE_WARMUP_S: '20' }
    : { TORCH_PROFILE_ITERATIONS: '5', TORCH_PROFILE_WARMUP_S: '60' }
  return [
    { name: 'torch_profiler_dynolog', env_vars: env },
    { name: 'tracelens', env_vars: { TRACELENS_MODE: 'pytorch' } },
  ]
}
function dispatchTools() {
  const rtl = profileTool === 'rocm_trace_lite' || profileTool === 'rocm_trace_lite_default' || profileTool === 'rpd'
  return [{ name: rtl ? profileTool : 'rocm_trace_lite' }]
}

const RUNG_SCHEMA = {
  type: 'object',
  required: ['status', 'evidence'],
  properties: {
    status: { type: 'string', enum: ['ran', 'planned', 'skipped'] },
    bottleneck: { type: 'string', enum: ['compute', 'memory', 'communication', 'launch', 'mixed', 'unknown'] },
    bound: { type: 'string' },
    evidence: { type: 'string' },
    reason: { type: 'string' },
    preset: { type: ['string', 'null'] },
    tracePath: { type: ['string', 'null'] },
    hotspots: { type: 'array', items: { type: 'object' } },
    topOperators: { type: 'array', items: { type: 'object' } },
  },
}

// ── Phase 2: Operator (TraceLens) ──────────────────────────────────────────
phase('Operator')
let operator
if (!want('operator')) {
  operator = skipped(`depth=${depth}; operator rung not selected`)
} else if (baseline.pytorch === false) {
  operator = skipped('non-PyTorch model; TraceLens operator capture needs a Kineto trace from torch.profiler')
} else {
  const opPayload = toolsPayload(operatorTools())
  const opCmd = runCmd('perf_tune_operator.csv', opPayload)
  const opAction = execute
    ? `If AMD GPUs are present, run this command once. ${RUN_HOW}
Then read tracelens_output/tracelens_summary.json (or the per-trace CSV) written by this run. The perf number on this run is profiled — do not treat it as the baseline. If no GPUs, or no Kineto trace appears, set status "skipped".`
    : 'Do NOT run anything. If an existing tracelens_output/tracelens_summary.json is on disk, read it and set status "planned" only when you did not run; otherwise status "skipped". Print the command either way.'
  operator = await agent(
    `Operator-level profile of "${baseline.model}" (family: ${baseline.family}, target: ${target}).
${PREFLIGHT}
This rung is TraceLens on a PyTorch Kineto trace. Command:
${opCmd}
Tools are torch_profiler_dynolog plus tracelens. Do not add rocm_trace_lite or rocprofv3 to this command.
${baseline.family === 'inference' ? 'Inference has no optimizer.step(); TORCH_PROFILE_ITERATIONS is already 0 (duration capture).' : 'Training uses iteration capture (TORCH_PROFILE_ITERATIONS=5).'}
${opAction}
Return status, bound (compute|memory|communication|launch|mixed|unknown), evidence, topOperators [{name, pct, module}], tracePath.`,
    { phase: 'Operator', schema: RUNG_SCHEMA, agentType: 'mad-benchmark-runner' }
  ) || skipped('operator agent returned no result')
}
log(`Operator: ${operator.status}. ${operator.evidence || operator.reason || ''}`)

// ── Phase 3: Dispatch (rocm-trace-lite) ────────────────────────────────────
phase('Dispatch')
let dispatch
if (!want('dispatch')) {
  dispatch = skipped(`depth=${depth}; dispatch rung not selected`)
} else {
  const dPayload = toolsPayload(dispatchTools())
  const dCmd = runCmd('perf_tune_dispatch.csv', dPayload)
  const dAction = execute
    ? `If AMD GPUs are present, run this command once. ${RUN_HOW}
Then read rocm_trace_lite_output/ (trace.db or trace_summary.txt) from this run for kernel dispatch names and their share of GPU time. If no GPUs, set status "skipped". The perf number is profiled — ignore it as a score.`
    : 'Do NOT run anything. Read an existing rocm_trace_lite_output/ if present. Print the command. Set status "planned" when you only printed it, "skipped" when there is nothing to read.'
  dispatch = await agent(
    `Dispatch-level profile of "${baseline.model}" (family: ${baseline.family}).
${PREFLIGHT}
This rung is rocm-trace-lite only. Command:
${dCmd}
Do not add rocprof, rocprofv3_*, or tracelens to this command. rocm-trace-lite and rocprofv3 cannot share a run.
Operator evidence so far: ${operator.evidence || operator.reason || 'none'}
${dAction}
Classify bottleneck as compute, memory, communication, launch, mixed, or unknown:
- compute: GEMM/conv kernels dominate (Cijk*, *gemm*, wvSplitK)
- memory: attention, elementwise, norm, or copy kernels dominate
- communication: RCCL/NCCL collectives are a large share
- launch: many tiny kernels and timeline gaps
Return status, bottleneck, evidence citing hotspot percents, hotspots [{name, pct}], tracePath.`,
    { phase: 'Dispatch', schema: RUNG_SCHEMA, agentType: 'mad-benchmark-runner' }
  ) || skipped('dispatch agent returned no result')
}
log(`Dispatch: ${dispatch.status} ${dispatch.bottleneck || ''}. ${(dispatch.hotspots || []).slice(0, 3).map(h => `${h.name} ${h.pct != null ? h.pct + '%' : ''}`).join(', ')}`)

// ── Phase 4: Kernel (one rocprofv3 preset) ─────────────────────────────────
phase('Kernel')
let kernel
let kernelPreset = null
if (depth === 'kernel' && profileTool && String(profileTool).startsWith('rocprof'))
  kernelPreset = profileTool
else if (dispatch && KERNEL_PRESET[dispatch.bottleneck])
  kernelPreset = KERNEL_PRESET[dispatch.bottleneck]
else if (depth === 'kernel')
  kernelPreset = 'rocprofv3_compute'

const skipKernel = !want('kernel')
  || (depth === 'auto' && dispatch.bottleneck === 'launch')
  || (depth === 'auto' && !kernelPreset)
if (skipKernel) {
  const why = !want('kernel')
    ? `depth=${depth}; kernel rung not selected`
    : (dispatch.bottleneck === 'launch'
      ? 'dispatch is launch-bound; counter collection would not explain the gaps'
      : 'dispatch did not name a compute, memory, or communication hotspot')
  kernel = skipped(why)
} else {
  const kPayload = toolsPayload([{ name: kernelPreset }])
  const kCmd = runCmd('perf_tune_kernel.csv', kPayload)
  const kAction = execute
    ? `If AMD GPUs are present, run this command once. ${RUN_HOW}
Then read rocprof_output/ from this run for the counter or kernel summary. If no GPUs, set status "skipped". Ignore the profiled perf number as a score.`
    : 'Do NOT run anything. Read an existing rocprof_output/ if present. Print the command and set status "planned" or "skipped".'
  kernel = await agent(
    `Kernel-level profile of "${baseline.model}" using preset ${kernelPreset}.
${PREFLIGHT}
Dispatch said ${dispatch.bottleneck || 'unknown'}-bound. Evidence: ${dispatch.evidence || dispatch.reason || 'none'}
Command (this preset only — do not add rocm_trace_lite):
${kCmd}
${kAction}
Return status, preset "${kernelPreset}", bottleneck, evidence from the counters, hotspots [{name, pct}], tracePath.`,
    { phase: 'Kernel', schema: RUNG_SCHEMA, agentType: 'mad-benchmark-runner' }
  ) || skipped('kernel agent returned no result')
  kernel.preset = kernel.preset || kernelPreset
}
log(`Kernel: ${kernel.status} preset=${kernel.preset || kernelPreset || 'none'}. ${kernel.evidence || kernel.reason || ''}`)

// ── Phase 5: Expect (opt-in Primus projection) ─────────────────────────────
phase('Expect')
const EXPECT_SCHEMA = {
  type: 'object',
  required: ['source'],
  properties: {
    source: { type: 'string', enum: ['projection', 'planned', 'skipped'] },
    fits: { type: ['boolean', 'null'] },
    memoryPerGpuGb: { type: ['number', 'null'] },
    expectedTokensPerSecPerGpu: { type: ['number', 'null'] },
    expectedTflops: { type: ['number', 'null'] },
    recommended: { type: ['object', 'null'] },
    reason: { type: 'string' },
  },
}
function expectSkipped(reason) {
  return {
    source: 'skipped',
    fits: null,
    memoryPerGpuGb: null,
    expectedTokensPerSecPerGpu: null,
    expectedTflops: null,
    recommended: null,
    reason,
  }
}
let expectResult
if (!expectOn) {
  expectResult = expectSkipped('Expect is off. Pass --expect to attach Primus projection numbers for a Primus training model.')
} else {
  const expectAction = execute
    ? 'Run the projection commands if the model qualifies and scripts/Primus is present. If primus-cli or the YAML is missing, set source "skipped" and say why. Do not write these numbers into perf.csv.'
    : 'Do NOT run primus-cli. Print the projection commands and set source "planned", or "skipped" when the model does not qualify.'
  expectResult = await agent(
    `Supply expected tokens/s and memory fit for "${baseline.model}" (tag ${tag}).
${PREFLIGHT}
Read and follow .claude/skills/primus-projection/SKILL.md. That skill is the step that supplies the expected numbers. Also read scripts/Primus/skills/primus-projection/SKILL.md when that file exists, and use its command shapes.
Discovered primusConfig: ${baseline.primusConfig || 'null'}
Discovered framework: ${baseline.framework || 'null'}
Family: ${baseline.family}
${expectAction}
Supported frameworks: megatron, torchtitan, maxtext (jax), torchrec_dlrm. Inference and every other trainer return source "skipped".
Return source, fits, memoryPerGpuGb, expectedTokensPerSecPerGpu, expectedTflops, recommended {tp, pp, ep, cp, mbs, gbs, schedule, recompute} or null, and reason.`,
    { phase: 'Expect', schema: EXPECT_SCHEMA, agentType: 'mad-benchmark-runner' }
  ) || expectSkipped('expect agent returned no result')
}
log(`Expect: ${expectResult.source}. ${expectResult.reason || ''}`)

// ── Phase 6: Propose ───────────────────────────────────────────────────────
phase('Propose')
const CANDS_SCHEMA = {
  type: 'object',
  required: ['candidates'],
  properties: {
    candidates: { type: 'array', items: {
      type: 'object',
      required: ['id', 'change', 'hypothesis', 'evidence'],
      properties: {
        id: { type: 'string' },
        change: { type: 'string' },
        hypothesis: { type: 'string' },
        evidence: { type: 'string' },
        depth: { type: 'string', enum: ['operator', 'dispatch', 'kernel', 'expect'] },
        leverKind: { type: 'string', enum: ['env', 'arg', 'config', 'other'] },
      },
    } },
  },
}
const proposed = await agent(
  `Propose ${nCandidates} DISTINCT tuning candidates to improve ${target} for "${baseline.model}" (family: ${baseline.family}).
Baseline: ${baseline.baselineSummary}${baseStats.n ? ` (clean median ${baseStats.median} ${baseline.baselineMetric || ''} over ${baseStats.n} run(s))` : ''}
Operator: ${operator.status} — ${operator.evidence || operator.reason || ''}
Top operators: ${JSON.stringify((operator.topOperators || []).slice(0, 5))}
Dispatch: ${dispatch.status} ${dispatch.bottleneck || ''} — ${dispatch.evidence || dispatch.reason || ''}
Hotspots: ${JSON.stringify((dispatch.hotspots || []).slice(0, 5))}
Kernel: ${kernel.status} preset=${kernel.preset || 'none'} — ${kernel.evidence || kernel.reason || ''}
Expect: ${JSON.stringify(expectResult)}
Levers available on this model: ${(baseline.levers || []).join(', ')}

Use the deepest rung that actually returned evidence. Cite that evidence on every candidate.
- Operator evidence: precision, batch or sequence, attention backend, the module TraceLens named. Training may also change microbatch or recompute. Inference may change gpu-memory-utilization, max-num-seqs, max-num-batched-tokens, prefix cache, or HIP graphs.
- Dispatch launch-bound: HIP graphs, fewer syncs, larger batch.
- Kernel evidence: PYTORCH_TUNABLEOP_ENABLED, hipBLASLt, or the kernel flag the counters implicate.
- Training parallelism (TP/PP/EP, RCCL_*) only when dispatch or the kernel preset shows collectives.
- When Expect source is "projection" and recommended is set, prefer candidates from that config, best projected tokens/s first, and drop a config Projection says will not fit. Read the optimization tables in scripts/Primus/skills/primus-projection/SKILL.md and propose only a lever that file lists and this config supports. Still change ONE lever per candidate.
- Prefer levers that need no file edit: leverKind "env" (docker_env_vars) or "arg" (madengine passes --tags <model>:<arg>=<value> to the script as --<arg> <value>).
Each candidate returns id, change, hypothesis, evidence, depth (operator|dispatch|kernel|expect), leverKind (env|arg|config|other).`,
  { phase: 'Propose', schema: CANDS_SCHEMA, agentType: 'mad-tuner' }
)
if (!proposed) throw new Error('Propose step returned no result; no candidates to evaluate.')
const candidates = (proposed.candidates || []).slice(0, nCandidates)
log(`Evaluating ${candidates.length} candidate(s) sequentially, CLEAN (profiler off), ${repeats} run(s) each. execute=${execute}`)

// ── Phase 7: Evaluate (CLEAN, sequential, N runs) + adversarial verify ─────
phase('Evaluate')
const VERDICT_SCHEMA = {
  type: 'object',
  required: ['id', 'trustworthy'],
  properties: {
    id: { type: 'string' },
    trustworthy: { type: 'boolean' },
    reason: { type: 'string' },
  },
}
const ctxHint = cleanCtx
  ? 'A CLEAN base --additional-context (profiler OFF) is in the commands. If this candidate\'s leverKind is "env", apply the change by MERGING the env var into that context\'s "docker_env_vars" object (show the merged commands you actually run); do not export it separately.'
  : 'If leverKind is "env", apply it by adding --additional-context \'{"docker_env_vars": {"VAR": "value"}}\' to every command.'
const direction = higherIsBetter
  ? 'Higher is better (throughput).'
  : 'Lower is better (latency).'

// Tracked-file state, read by its own agent so a candidate cannot vouch for itself.
const TREE_SCHEMA = {
  type: 'object',
  required: ['porcelain'],
  properties: { porcelain: { type: 'string' } },
}
async function trackedChanges(label) {
  const r = await agent(
    'From the MAD repo root run `git status --porcelain --untracked-files=no` and return its exact output in porcelain ("" when it prints nothing). Change nothing.',
    { label, phase: 'Evaluate', schema: TREE_SCHEMA, effort: 'low' }
  )
  return r ? r.porcelain.trim() : null
}
const treeBefore = execute && candidates.length ? await trackedChanges('tree:before') : null
if (execute && candidates.length && treeBefore === null)
  throw new Error('Could not read git status before evaluating candidates; nothing was measured.')

const evaluated = []
for (const cand of candidates) {
  const safeId = String(cand.id).replace(/[^A-Za-z0-9._-]/g, '_')
  const outs = sampleOutFiles(`perf_tune_${safeId}`)
  const cmds = outs.map(o => runCmd(o, cleanCtx))
  const action = execute
    ? `If AMD GPUs are present (check rocm-smi/amd-smi), apply the change ONCE, run the ${repeats} commands CLEAN (no tools key) one after another, then REVERT any file/config edit. The workflow compares \`git status\` with its pre-candidate snapshot afterwards and stops the search on any leftover change. ${RUN_HOW}
${parseHint} If no GPUs, set status "skipped" with no samples.`
    : 'Do NOT execute. Produce the exact commands (plus precisely how to apply the change) and set status "planned" with no samples.'

  const runs = await agent(
    `Evaluate tuning candidate ${cand.id} for "${baseline.model}" (family: ${baseline.family}, leverKind: ${cand.leverKind || 'other'}, depth: ${cand.depth || 'dispatch'}).
${PREFLIGHT}
Change: ${cand.change}
Hypothesis: ${cand.hypothesis}
Motivating evidence: ${cand.evidence}
Base commands (CLEAN, one per sample):
${cmds.map(c => '  ' + c).join('\n')}
${ctxHint}
${action}
Return status, one number per successful run in samples (same order), the metric unit, the exact commands you ran, and notes.`,
    { label: `eval:${cand.id}`, phase: 'Evaluate', schema: SAMPLES_SCHEMA, agentType: 'mad-tuner' }
  ) || { status: 'error', samples: [], metric: null, commands: cmds, notes: 'evaluate agent returned no result' }

  let leaked = null
  if (execute) {
    const treeAfter = await trackedChanges(`tree:${cand.id}`)
    if (treeAfter !== treeBefore)
      leaked = treeAfter === null
        ? 'could not read git status after this candidate'
        : `tracked files differ from the pre-candidate snapshot:\n${treeAfter || '(clean)'}\nexpected:\n${treeBefore || '(clean)'}`
  }
  if (leaked) {
    runs.status = 'error'
    runs.samples = []
    runs.notes = `${runs.notes ? runs.notes + ' ' : ''}Discarded: ${leaked}`
  }

  const candStats = stats(runs.samples)
  const g = gate(baseStats, candStats, baseline.baselineMetric, runs.metric, higherIsBetter)
  const measured = {
    id: cand.id,
    status: runs.status,
    commands: runs.commands || cmds,
    metric: runs.metric || null,
    samples: runs.samples || [],
    median: candStats.median,
    spreadPct: candStats.spreadPct,
    deltaPct: g.deltaPct,
    gate: g,
    notes: runs.notes || '',
  }

  let verdict = { id: cand.id, trustworthy: false, reason: `gate: ${g.reason}` }
  if (g.pass) {
    verdict = await agent(
      `Adversarially review tuning candidate ${cand.id}.
Target: ${target}. ${direction}
Baseline (computed): median ${baseStats.median} ${baseline.baselineMetric || ''}, samples ${JSON.stringify(baseRuns.samples)}, spread ${baseStats.spreadPct != null ? baseStats.spreadPct.toFixed(2) + '%' : 'n/a'}
Candidate (computed): ${JSON.stringify(measured)}
Hypothesis was: ${cand.hypothesis}
The numeric gate passed: ${g.reason}. Look for what the numbers cannot show: the change was not actually applied as described, a run used a profiler, a run failed or was retried, the edit leaked into other runs, or the result does not follow from the hypothesis. Return id, trustworthy, reason.`,
      { label: `verify:${cand.id}`, phase: 'Evaluate', schema: VERDICT_SCHEMA, agentType: 'mad-perf-analyst' }
    ) || { id: cand.id, trustworthy: false, reason: 'verify agent returned no result' }
  }
  evaluated.push({ ...measured, _change: cand.change, _evidence: cand.evidence, _depth: cand.depth,
    verdict: { ...verdict, trustworthy: g.pass && verdict.trustworthy === true } })
  if (leaked) {
    const rest = candidates.slice(candidates.indexOf(cand) + 1).map(c => c.id)
    log(`Stopped: candidate ${cand.id} left the working tree changed. Restore it (git diff / git checkout) before re-running.${rest.length ? ` Not evaluated: ${rest.join(', ')}.` : ''}`)
    break
  }
}

// ── Phase 8: Synthesize ────────────────────────────────────────────────────
phase('Synthesize')
const clean = evaluated.filter(Boolean)
const report = await agent(
  `Synthesize a MAD tuning search for "${baseline.model}" (family: ${baseline.family}, target: ${target}).
${direction}
Baseline: ${baseline.baselineSummary}${baseStats.n ? ` (clean median ${baseStats.median} ${baseline.baselineMetric || ''} over ${baseStats.n} run(s), spread ${baseStats.spreadPct != null ? baseStats.spreadPct.toFixed(2) + '%' : 'n/a'})` : ' (not measured)'}
Operator: ${operator.status} — ${operator.evidence || operator.reason || ''}
Dispatch: ${dispatch.status} ${dispatch.bottleneck || ''} — ${dispatch.evidence || dispatch.reason || ''}
Kernel: ${kernel.status} preset=${kernel.preset || 'none'} — ${kernel.evidence || kernel.reason || ''}
Expect: ${JSON.stringify(expectResult)}
Evaluated candidates (JSON; median, spreadPct, deltaPct and gate are computed, verdict.trustworthy is final): ${JSON.stringify(clean)}

Produce:
(1) a one-line headline: the depth that explained the bottleneck, and the best trustworthy improvement (or that none beat baseline);
(2) one paragraph each for operator, dispatch, and kernel, including a skipped reason when that rung did not run;
(3) a table: Candidate | Depth | Change | Evidence | Status | Median (n) | vs Baseline | Trustworthy;
(4) when Expect source is "projection", the projected tokens/s/GPU beside the measured number;
(5) the recommended configuration — ONLY from candidates whose verdict.trustworthy is true; if none ran, present the ranked plan to test on a GPU host;
(6) the exact next clean madengine command.
Each candidate wrote its own perf_tune_<id>_r<k>.csv files. Do not invent measurements or recompute the given statistics. Do not treat a profiled run as the score.`,
  { phase: 'Synthesize', agentType: 'mad-perf-analyst' }
)

return {
  model: baseline.model,
  target,
  family: baseline.family,
  execute,
  depth,
  repeats,
  expect: expectOn,
  baseline: { ...baseRuns, ...baseStats },
  operator,
  dispatch,
  kernel,
  expectResult,
  candidates: clean,
  report,
}
