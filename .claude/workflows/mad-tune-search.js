export const meta = {
  name: 'mad-tune-search',
  description: 'Descend operator, dispatch, and kernel profiling for a training or inference MAD model, then measure clean candidates and synthesize the winning config',
  phases: [
    { title: 'Baseline', detail: 'classify the model and establish a clean baseline number' },
    { title: 'Operator', detail: 'TraceLens on a PyTorch Kineto trace, when the model is PyTorch' },
    { title: 'Dispatch', detail: 'rocm-trace-lite kernel-dispatch timeline' },
    { title: 'Kernel', detail: 'one rocprofv3 counter preset chosen from the dispatch class' },
    { title: 'Expect', detail: 'Primus projection numbers only when --expect is passed' },
    { title: 'Propose', detail: 'one-lever candidates citing the deepest evidence' },
    { title: 'Evaluate', detail: 'measure each candidate cleanly (sequential) and verify its delta' },
    { title: 'Synthesize', detail: 'pick the winning configuration' },
  ],
}

// ARGS — accepts EITHER:
//   (a) a structured object: { tag, target?, candidates?, execute?,
//       additionalContext?, timeout?, depth?, profileTool?, expect? }
//   (b) a CLI-style string, e.g.
//       --tag pyt_vllm_qwen3-8b --target latency --depth auto --candidates 4
//       --tag primus_train/megatron_MI355X_llama3.1_8B-BF16-pretrain --expect
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
// EXPECT is OFF unless the user passes --expect / --expect auto. Projection only
// scores Primus training configs. Inference and every other trainer skip it even
// then. --expect off and --no-expect force it off.
//
// --depth operator|dispatch|kernel runs that rung only. --profile-tool NAME forces
// the matching rung and uses NAME as its tool. Default --depth auto descends.
//
// EXECUTE: runs for real by default. --no-execute / --plan prints commands only.
// Candidates are measured one at a time so GPU runs and file edits do not race.

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

function parseExpect(v) {
  if (v == null || v === true || v === 'auto' || v === 'on' || v === '') return true
  if (v === false || v === 'off' || v === 'false' || v === '0') return false
  return true
}

function parseCli(s) {
  const toks = tokenize(s)
  const cfg = { execute: true, expect: false, depth: 'auto' }
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

let cfg
if (typeof args === 'string') {
  cfg = parseCli(args)
} else if (args && typeof args === 'object') {
  cfg = { execute: true, expect: false, depth: 'auto', ...args }
  if (Object.prototype.hasOwnProperty.call(args, 'expect')) cfg.expect = parseExpect(args.expect)
} else {
  cfg = { execute: true, expect: false, depth: 'auto' }
}

const DEPTHS = ['auto', 'operator', 'dispatch', 'kernel']
const tag = cfg.tag || 'bert'
const target = cfg.target || 'throughput'
const nCandidates = Math.max(1, Math.min(cfg.candidates || 4, 8))
const execute = cfg.execute !== false
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

let ctxObj = null
if (cfg.additionalContext) {
  try { ctxObj = JSON.parse(cfg.additionalContext) }
  catch (e) { log('Warning: --additional-context is not valid JSON; clean runs pass it through verbatim, and profiled runs use a tools-only context.') }
}
const baseObj = (ctxObj && typeof ctxObj === 'object') ? { ...ctxObj } : null
if (baseObj) delete baseObj.tools
const cleanCtx = baseObj
  ? JSON.stringify(baseObj)
  : (cfg.additionalContext || null)
if (ctxObj && Array.isArray(ctxObj.tools) && ctxObj.tools.length)
  log('Note: tools in --additional-context are not used. Each depth builds its own tools array. Pass --depth or --profile-tool to force one rung.')

function toolsPayload(tools) {
  const names = tools.map(t => t.name)
  const hasRtl = names.some(n => String(n).startsWith('rocm_trace_lite'))
  const hasRocprof = names.some(n => String(n).startsWith('rocprof'))
  if (hasRtl && hasRocprof)
    throw new Error('refusing to stack rocm_trace_lite with rocprofv3 in one madengine run')
  const o = { ...(baseObj || {}), tools }
  return JSON.stringify(o)
}
function ctxFlag(payload) {
  return payload ? ` --additional-context ${shellQuote(payload)}` : ''
}
function runCmd(outFile, payload) {
  const flags = ['--tags ' + tag, '--live-output', '-o ' + outFile]
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
log(`Tuning "${tag}" for ${target}. depth=${depth}. expect=${expectOn}. execute=${execute}. ${cleanCtx ? 'clean context ready' : 'no additional-context'}.`)

const PREFLIGHT = 'Pre-flight: run `bash .claude/skills/mad-common/preflight.sh` and stop if it exits non-zero.'

// ── Phase 1: Baseline (CLEAN) ──────────────────────────────────────────────
phase('Baseline')
const BASELINE_SCHEMA = {
  type: 'object',
  required: ['model', 'scriptPath', 'baselineSummary', 'family', 'pytorch'],
  properties: {
    model: { type: 'string' },
    scriptPath: { type: 'string' },
    baselineSummary: { type: 'string' },
    multipleResults: { type: ['string', 'null'] },
    baselinePerf: { type: ['number', 'null'] },
    baselineMetric: { type: ['string', 'null'] },
    family: { type: 'string', enum: ['training', 'inference', 'other'] },
    pytorch: { type: 'boolean' },
    primusConfig: { type: ['string', 'null'] },
    framework: { type: ['string', 'null'] },
    levers: { type: 'array', items: { type: 'string' } },
  },
}
const baseAction = execute
  ? `If AMD GPUs are present, establish a CLEAN baseline (profiler OFF, no tools key): either reuse the most recent clean perf row for this model in perf.csv or its results CSV, or run "${runCmd('perf_tune_baseline.csv', cleanCtx)}" once and parse it. Report baselinePerf + baselineMetric.`
  : 'Do NOT run anything. Read the static config only; leave baselinePerf null.'
const baseline = await agent(
  `In the MAD repo, establish the tuning baseline for tag "${tag}" (target: ${target}).
${PREFLIGHT}
Run \`madengine discover --tags ${tag}\` first. Find the models.json entry (or the dynamic
entry), its scripts/.../run.sh, dockerfile, tags, and args.
Classify:
- family "training" when tags/scripts/dockerfile indicate pretrain, finetune, primus, megatron, torchtitan, maxtext, or torchtune.
- family "inference" when they indicate vllm, sglang, inference, or serving.
- family "other" otherwise. A Primus training config is training even if the model name is familiar from inference.
- pytorch true for PyTorch, vLLM, SGLang, Torchtune, and Primus Megatron/TorchTitan. pytorch false for JAX/MaxText and other non-PyTorch stacks.
- primusConfig: the --config_path value when discover shows one under scripts/Primus, else null.
- framework: modules.pre_trainer.framework or modules.post_trainer.framework from that YAML when you can read it, else null.
List levers that actually exist for this family. Training: batch, precision, TP/PP/EP, RCCL/NCCL, recompute, PYTORCH_TUNABLEOP_ENABLED. Inference: gpu-memory-utilization, max-num-seqs, max-num-batched-tokens, tensor-parallel, precision, HIP graphs, attention backend.
Report whether the entry sets "multiple_results" (its CSV filename, or null).
${baseAction}`,
  { phase: 'Baseline', schema: BASELINE_SCHEMA }
)
const multiCsv = baseline.multipleResults || null
const parseHint = multiCsv
  ? `This is a multiple_results model — performance is NOT a stdout line; after the run read the per-metric CSV "${multiCsv}" (and the -o file) and report the primary ${target} metric.`
  : 'Parse the "performance: <value> <unit>" line from stdout. Ignore perf numbers taken under a profiler.'
const higherIsBetter = target !== 'latency'

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
    ? `If AMD GPUs are present, run this command once. Then read tracelens_output/tracelens_summary.json (or the per-trace CSV). The perf number on this run is profiled — do not treat it as the baseline. If no GPUs, or no Kineto trace appears, set status "skipped".`
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
    { phase: 'Operator', schema: RUNG_SCHEMA }
  )
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
    ? 'If AMD GPUs are present, run this command once and read rocm_trace_lite_output/ (trace.db or trace_summary.txt) for kernel dispatch names and their share of GPU time. If no GPUs, set status "skipped". The perf number is profiled — ignore it as a score.'
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
    { phase: 'Dispatch', schema: RUNG_SCHEMA }
  )
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
    ? 'If AMD GPUs are present, run this command once and read rocprof_output/ for the counter or kernel summary. If no GPUs, set status "skipped". Ignore the profiled perf number as a score.'
    : 'Do NOT run anything. Read an existing rocprof_output/ if present. Print the command and set status "planned" or "skipped".'
  kernel = await agent(
    `Kernel-level profile of "${baseline.model}" using preset ${kernelPreset}.
${PREFLIGHT}
Dispatch said ${dispatch.bottleneck || 'unknown'}-bound. Evidence: ${dispatch.evidence || dispatch.reason || 'none'}
Command (this preset only — do not add rocm_trace_lite):
${kCmd}
${kAction}
Return status, preset "${kernelPreset}", bottleneck, evidence from the counters, hotspots [{name, pct}], tracePath.`,
    { phase: 'Kernel', schema: RUNG_SCHEMA }
  )
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
let expectResult
if (!expectOn) {
  expectResult = {
    source: 'skipped',
    fits: null,
    memoryPerGpuGb: null,
    expectedTokensPerSecPerGpu: null,
    expectedTflops: null,
    recommended: null,
    reason: 'Expect is off. Pass --expect to attach Primus projection numbers for a Primus training model.',
  }
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
    { phase: 'Expect', schema: EXPECT_SCHEMA }
  )
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
Baseline: ${baseline.baselineSummary}
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
Each candidate returns id, change, hypothesis, evidence, depth (operator|dispatch|kernel|expect), leverKind (env|arg|config|other).`,
  { phase: 'Propose', schema: CANDS_SCHEMA }
)
const candidates = (proposed.candidates || []).slice(0, nCandidates)
log(`Evaluating ${candidates.length} candidate(s) sequentially, CLEAN (profiler off). execute=${execute}`)

// ── Phase 7: Evaluate (CLEAN, sequential) + adversarial verify ─────────────
phase('Evaluate')
const EVAL_SCHEMA = {
  type: 'object',
  required: ['id', 'status'],
  properties: {
    id: { type: 'string' },
    status: { type: 'string', enum: ['planned', 'ran', 'skipped', 'error'] },
    command: { type: 'string' },
    performance: { type: ['number', 'null'] },
    metric: { type: ['string', 'null'] },
    notes: { type: 'string' },
  },
}
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
  ? 'A CLEAN base --additional-context (profiler OFF) is in the command. If this candidate\'s leverKind is "env", apply the change by MERGING the env var into that context\'s "docker_env_vars" object (show the merged command you actually run); do not export it separately.'
  : 'If leverKind is "env", apply by adding --additional-context \'{"docker_env_vars": {"VAR": "value"}}\' (or exporting the var).'
const direction = higherIsBetter
  ? 'Higher is better (throughput).'
  : 'Lower is better (latency).'

const evaluated = []
for (const cand of candidates) {
  const safeId = String(cand.id).replace(/[^A-Za-z0-9._-]/g, '_')
  const outFile = `perf_tune_${safeId}.csv`
  const cmd = runCmd(outFile, cleanCtx)
  const action = execute
    ? `If AMD GPUs are present (check rocm-smi/amd-smi), apply the change, RUN the command CLEAN (no tools key), parse the result, then REVERT any file/config edit so the next candidate starts clean. ${parseHint} Results land in ${outFile}. If no GPUs, set status "skipped".`
    : 'Do NOT execute. Produce the exact command (plus precisely how to apply the change) and set status "planned".'

  const evalResult = await agent(
    `Evaluate tuning candidate ${cand.id} for "${baseline.model}" (family: ${baseline.family}, leverKind: ${cand.leverKind || 'other'}, depth: ${cand.depth || 'dispatch'}).
${PREFLIGHT}
Change: ${cand.change}
Hypothesis: ${cand.hypothesis}
Motivating evidence: ${cand.evidence}
Base command (CLEAN): ${cmd}
${ctxHint}
${action}
Return id, status, the exact command you ran, and performance/metric if measured.`,
    { label: `eval:${cand.id}`, phase: 'Evaluate', schema: EVAL_SCHEMA }
  )

  const verdict = await agent(
    `Adversarially review tuning candidate ${cand.id}.
Target: ${target}. ${direction}
Baseline: ${baseline.baselinePerf != null ? baseline.baselinePerf + ' ' + (baseline.baselineMetric || '') : baseline.baselineSummary}
Claimed result: ${JSON.stringify(evalResult)}
Hypothesis was: ${cand.hypothesis}
Is the conclusion trustworthy? Default to trustworthy=false if the change was not actually
measured (planned/skipped), if only one sample was taken, if it ran under a profiler, if the
delta is in the wrong direction for ${target}, or if the delta vs baseline is within
run-to-run noise (~1-2%). Return id, trustworthy, reason.`,
    { label: `verify:${cand.id}`, phase: 'Evaluate', schema: VERDICT_SCHEMA }
  )
  evaluated.push({ ...evalResult, _change: cand.change, _evidence: cand.evidence, _depth: cand.depth, verdict })
}

// ── Phase 8: Synthesize ────────────────────────────────────────────────────
phase('Synthesize')
const clean = evaluated.filter(Boolean)
const report = await agent(
  `Synthesize a MAD tuning search for "${baseline.model}" (family: ${baseline.family}, target: ${target}).
${direction}
Baseline: ${baseline.baselineSummary}${baseline.baselinePerf != null ? ` (clean baseline: ${baseline.baselinePerf} ${baseline.baselineMetric || ''})` : ''}
Operator: ${operator.status} — ${operator.evidence || operator.reason || ''}
Dispatch: ${dispatch.status} ${dispatch.bottleneck || ''} — ${dispatch.evidence || dispatch.reason || ''}
Kernel: ${kernel.status} preset=${kernel.preset || 'none'} — ${kernel.evidence || kernel.reason || ''}
Expect: ${JSON.stringify(expectResult)}
Evaluated candidates (JSON, each with a verdict): ${JSON.stringify(clean)}

Produce:
(1) a one-line headline: the depth that explained the bottleneck, and the best trustworthy improvement (or that none beat baseline);
(2) one paragraph each for operator, dispatch, and kernel, including a skipped reason when that rung did not run;
(3) a table: Candidate | Depth | Change | Evidence | Status | Performance | vs Baseline | Trustworthy;
(4) when Expect source is "projection", the projected tokens/s/GPU beside the measured number;
(5) the recommended configuration — ONLY from candidates whose verdict is trustworthy; if none ran, present the ranked plan to test on a GPU host;
(6) the exact next clean madengine command.
Each candidate wrote its own perf_tune_<id>.csv. Do not invent measurements. Do not treat a profiled run as the score.`,
  { phase: 'Synthesize' }
)

return {
  model: baseline.model,
  target,
  family: baseline.family,
  execute,
  depth,
  expect: expectOn,
  operator,
  dispatch,
  kernel,
  expectResult,
  candidates: clean,
  report,
}
