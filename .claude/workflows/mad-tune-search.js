export const meta = {
  name: 'mad-tune-search',
  description: 'Profile a MAD model to diagnose its bottleneck, generate profiling-informed tuning candidates, measure each cleanly + adversarially verify, and synthesize the winning config',
  phases: [
    { title: 'Baseline', detail: 'read the model + establish a clean baseline number' },
    { title: 'Diagnose', detail: 'profile once to find hotspots and classify the bottleneck' },
    { title: 'Propose', detail: 'generate distinct candidates, each citing profiling evidence' },
    { title: 'Evaluate', detail: 'measure each candidate cleanly (sequential) and verify its delta' },
    { title: 'Synthesize', detail: 'pick the winning configuration' },
  ],
}

// ARGS — accepts EITHER:
//   (a) a structured object: { tag, target?, candidates?, execute?,
//       additionalContext?, timeout?, profileTool? }
//   (b) a CLI-style string mirroring `/mad-benchmark` / `/mad-profile`, e.g.
//       --tag pyt_vllm_qwen3-8b --candidates 6 \
//         --additional-context '{"gpu_vendor":"AMD","tools":[{"name":"rocm_trace_lite"}],...}'
//
// WHY profiling is split from measurement (the core design):
//   - DIAGNOSE uses a profiler (tools[]) to find WHAT to tune: kernel hotspots,
//     and whether the workload is compute- / memory- / communication- / launch-bound.
//   - EVALUATE measures WHETHER a change helped using CLEAN runs (profiler OFF),
//     because tracing overhead (e.g. rocm_trace_lite intercepts ~all GPU ops) buries
//     the small perf deltas tuning produces and would make verify reject real wins.
//   So we run the profiler exactly ONCE (Diagnose), then strip the `tools` key from
//   the context for the baseline and every candidate measurement.
//
// EXECUTE semantics: runs for real by default. Pass --no-execute / --plan to only
// diagnose-from-static + produce the ranked candidate plan without GPU measurement.
//
// SEQUENTIAL evaluation: candidates are measured one at a time — parallel GPU runs
// contend (corrupting deltas) and parallel run.sh/config edits race on the same files.

// --- tokenizer: split a CLI string respecting single/double quotes ---
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

// --- parse a CLI string into the structured cfg the search uses ---
function parseCli(s) {
  const toks = tokenize(s)
  const cfg = { execute: true }
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
    } else if (t === '--timeout') {
      cfg.timeout = next()
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

// --- normalize args (string | object | undefined) into one cfg ---
let cfg
if (typeof args === 'string') {
  cfg = parseCli(args)
} else if (args && typeof args === 'object') {
  cfg = { execute: true, ...args }
} else {
  cfg = { execute: true }
}

const tag = cfg.tag || 'bert'
const target = cfg.target || 'throughput'
const nCandidates = Math.max(1, Math.min(cfg.candidates || 4, 8))
const execute = cfg.execute !== false
const timeout = cfg.timeout || null

// --- Split the user's --additional-context into a PROFILED and a CLEAN variant. ---
// profiledCtx: keeps (or adds) a `tools` entry → used once in Diagnose.
// cleanCtx:    `tools` key removed → used for baseline + every candidate measurement.
let ctxObj = null
if (cfg.additionalContext) {
  try { ctxObj = JSON.parse(cfg.additionalContext) }
  catch (e) { log(`Warning: --additional-context is not valid JSON; passing it through verbatim and skipping clean/profiled split.`) }
}

let cleanCtx = null      // JSON string, tools stripped
let profiledCtx = null   // JSON string, tools present
let profileToolName = cfg.profileTool || null

if (ctxObj && typeof ctxObj === 'object') {
  // derive the profiling tool name from the context if not given explicitly
  if (!profileToolName && Array.isArray(ctxObj.tools) && ctxObj.tools[0] && ctxObj.tools[0].name)
    profileToolName = ctxObj.tools[0].name
  if (!profileToolName) profileToolName = 'rocm_trace_lite'

  const cleanObj = { ...ctxObj }
  delete cleanObj.tools
  cleanCtx = JSON.stringify(cleanObj)

  const profObj = { ...ctxObj, tools: [{ name: profileToolName }] }
  profiledCtx = JSON.stringify(profObj)
} else if (cfg.additionalContext) {
  // not parseable as object — use verbatim for clean; still attempt a profiled variant
  cleanCtx = cfg.additionalContext
  if (!profileToolName) profileToolName = 'rocm_trace_lite'
}

const cleanCtxFlag = cleanCtx ? ` --additional-context '${cleanCtx}'` : ''
const profCtxFlag = profiledCtx ? ` --additional-context '${profiledCtx}'`
  : ` --additional-context '{"tools": [{"name": "${profileToolName || 'rocm_trace_lite'}"}]}'`

if (cfg._extraTags && cfg._extraTags.length)
  log(`Note: mad-tune-search tunes ONE model. Using "${tag}"; ignoring extra tags [${cfg._extraTags.join(', ')}]. Use mad-benchmark-sweep to compare multiple models.`)
log(`Tuning "${tag}" for ${target}. execute=${execute}. profile tool=${profileToolName || 'rocm_trace_lite'}. ${cleanCtx ? 'context split into clean+profiled' : 'no additional-context'}.`)

// ── Phase 1: Baseline (CLEAN) ──────────────────────────────────────────────
phase('Baseline')
const BASELINE_SCHEMA = {
  type: 'object',
  required: ['model', 'scriptPath', 'baselineSummary'],
  properties: {
    model: { type: 'string' },
    scriptPath: { type: 'string' },
    baselineSummary: { type: 'string' },
    multipleResults: { type: ['string', 'null'] },
    baselinePerf: { type: ['number', 'null'] },
    baselineMetric: { type: ['string', 'null'] },
    levers: { type: 'array', items: { type: 'string' } },
  },
}
const baseAction = execute
  ? `If AMD GPUs are present, establish a CLEAN baseline number (profiler OFF): either reuse the most recent clean perf row for this model if one exists in perf.csv / the model's results CSV, or run "madengine run --tags ${tag} --live-output -o perf_tune_baseline.csv${cleanCtxFlag}" once and parse it. Report baselinePerf + baselineMetric.`
  : `Do NOT run anything. Read the static config only; leave baselinePerf null.`
const baseline = await agent(
  `In the MAD repo, establish the tuning baseline for tag "${tag}" (target: ${target}).
Pre-flight: before running madengine, verify it is installed:
  if ! command -v madengine &>/dev/null; then
    if [ -f requirements.txt ] && grep -q madengine requirements.txt; then
      pip install -r requirements.txt
    else
      echo "[pre-flight] madengine not found. Install: pip install git+https://github.com/ROCm/madengine.git@main"; exit 1
    fi
  fi
  [ -f models.json ] || echo "[pre-flight] Warning: not in MAD repo root."
Find its models.json entry, its scripts/.../run.sh, and any config it references.
Summarize the current configuration and list the tuning levers available for this
stack (env vars like MAD_MODEL_BATCH_SIZE / PYTORCH_TUNABLEOP_ENABLED / NCCL_*/RCCL_*,
or args like tensor-parallel size, precision, gpu-memory-utilization, max-num-seqs).
Report whether the entry sets "multiple_results" (its CSV filename, or null).
${baseAction}`,
  { phase: 'Baseline', schema: BASELINE_SCHEMA }
)
const multiCsv = baseline.multipleResults || null
const parseHint = multiCsv
  ? `This is a multiple_results model — performance is NOT a stdout line; after the run read the per-metric CSV "${multiCsv}" (and the -o file) and report the primary ${target} metric.`
  : `Parse the "performance: <value> <unit>" line from stdout.`

// ── Phase 2: Diagnose (PROFILED, once) ─────────────────────────────────────
phase('Diagnose')
const DIAG_SCHEMA = {
  type: 'object',
  required: ['bottleneck', 'evidence', 'hotspots'],
  properties: {
    bottleneck: { type: 'string', enum: ['compute', 'memory', 'communication', 'launch', 'mixed', 'unknown'] },
    evidence: { type: 'string' },
    hotspots: { type: 'array', items: {
      type: 'object',
      required: ['name', 'pct'],
      properties: { name: { type: 'string' }, pct: { type: ['number', 'null'] }, kind: { type: 'string' } },
    } },
    recommendedLevers: { type: 'array', items: { type: 'string' } },
    tracePath: { type: ['string', 'null'] },
  },
}
const diagAction = execute
  ? `If AMD GPUs are present, profile the model ONCE: prefer reusing an existing recent trace dir (e.g. rocm_trace_lite_output/) if present and current; otherwise run "madengine run --tags ${tag} --live-output -o perf_tune_profiled.csv${profCtxFlag}". Then read the trace summary (e.g. trace_summary.txt / the trace db) to extract the top GPU-kernel hotspots with their % of GPU time. If no GPUs, classify from static config + known model characteristics and set tracePath null.`
  : `Do NOT run anything. Classify the likely bottleneck from the static config, model architecture, and any EXISTING trace output already on disk (e.g. rocm_trace_lite_output/trace_summary.txt). Set tracePath to that file if found, else null.`
const diag = await agent(
  `Diagnose the performance bottleneck of "${baseline.model}" to guide ${target} tuning.
Baseline config: ${baseline.baselineSummary}
${diagAction}

Classify the bottleneck as exactly one of: compute, memory, communication, launch, mixed, unknown — using these best-practice signals:
- compute: GEMM/conv kernels (e.g. Cijk*, *gemm*, wvSplitK) dominate GPU time; high occupancy.
  Levers: PYTORCH_TUNABLEOP_ENABLED, hipBLASLt tuning, fp8/lower precision, larger batch.
- memory: attention/elementwise/norm/copy kernels dominate; low arithmetic intensity / bandwidth-bound.
  Levers: KV-cache dtype, kernel fusion, gpu-memory-utilization, batch size.
- communication: RCCL/NCCL collectives (AllReduce/AllGather) are a large share (multi-GPU).
  Levers: NCCL_*/RCCL_* tuning, tensor-parallel vs pipeline-parallel topology.
- launch: many tiny kernels with timeline gaps / low GPU utilization.
  Levers: HIP/CUDA graphs, larger batch, fewer sync points.
Return bottleneck, evidence (cite the hotspot %s), hotspots[], recommendedLevers[], tracePath.`,
  { phase: 'Diagnose', schema: DIAG_SCHEMA }
)
log(`Diagnosis: ${diag.bottleneck}-bound. Top hotspots: ${(diag.hotspots || []).slice(0, 3).map(h => `${h.name} ${h.pct != null ? h.pct + '%' : ''}`).join(', ')}`)

// ── Phase 3: Propose (profiling-informed) ──────────────────────────────────
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
        leverKind: { type: 'string', enum: ['env', 'arg', 'config', 'other'] },
      },
    } },
  },
}
const proposed = await agent(
  `Propose ${nCandidates} DISTINCT tuning candidates to improve ${target} for "${baseline.model}".
Baseline: ${baseline.baselineSummary}
Diagnosis: ${diag.bottleneck}-bound. Evidence: ${diag.evidence}
Top hotspots: ${JSON.stringify((diag.hotspots || []).slice(0, 5))}
Recommended levers from diagnosis: ${(diag.recommendedLevers || []).join(', ')}
Generic available levers: ${(baseline.levers || []).join(', ')}

Each candidate changes ONE lever (so its effect is attributable) and should TARGET the
diagnosed bottleneck. Return id, change (concrete value + how to apply it), hypothesis,
evidence (which hotspot/diagnosis finding motivates it), and leverKind:
"env" (environment variable), "arg" (run-script/CLI arg), "config" (config-file value), or "other".`,
  { phase: 'Propose', schema: CANDS_SCHEMA }
)
const candidates = (proposed.candidates || []).slice(0, nCandidates)
log(`Evaluating ${candidates.length} candidate(s) sequentially, CLEAN (profiler off). execute=${execute}`)

// ── Phase 4: Evaluate (CLEAN, sequential) + adversarial verify ─────────────
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
  ? `A CLEAN base --additional-context (profiler OFF) is in the command. If this candidate's leverKind is "env", apply the change by MERGING the env var into that context's "docker_env_vars" object (show the merged command you actually run); do not export it separately.`
  : `If leverKind is "env", apply by adding --additional-context '{"docker_env_vars": {"VAR": "value"}}' (or exporting the var).`

const evaluated = []
for (const cand of candidates) {
  const safeId = String(cand.id).replace(/[^A-Za-z0-9._-]/g, '_')
  const outFile = `perf_tune_${safeId}.csv`
  const flags = ['--tags ' + tag, '--live-output', '-o ' + outFile]
  if (timeout) flags.push('--timeout ' + timeout)
  const cmd = `madengine run ${flags.join(' ')}${cleanCtxFlag}`
  const action = execute
    ? `If AMD GPUs are present (check rocm-smi/amd-smi), apply the change, RUN the command CLEAN (profiler OFF), parse the result, then REVERT any file/config edit so the next candidate starts clean. ${parseHint} Results land in ${outFile}. If no GPUs, set status "skipped".`
    : `Do NOT execute. Produce the exact command (plus precisely how to apply the change) and set status "planned".`

  const evalResult = await agent(
    `Evaluate tuning candidate ${cand.id} for "${baseline.model}" (leverKind: ${cand.leverKind || 'other'}).
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
Baseline: ${baseline.baselinePerf != null ? baseline.baselinePerf + ' ' + (baseline.baselineMetric || '') : baseline.baselineSummary}
Claimed result: ${JSON.stringify(evalResult)}
Hypothesis was: ${cand.hypothesis}
Is the conclusion trustworthy? Default to trustworthy=false if the change was not actually
measured (planned/skipped), if only one sample was taken, if it ran under a profiler, or if
the delta vs baseline is within run-to-run noise (~1-2%). Return id, trustworthy, reason.`,
    { label: `verify:${cand.id}`, phase: 'Evaluate', schema: VERDICT_SCHEMA }
  )
  evaluated.push({ ...evalResult, _change: cand.change, _evidence: cand.evidence, verdict })
}

// ── Phase 5: Synthesize ────────────────────────────────────────────────────
phase('Synthesize')
const clean = evaluated.filter(Boolean)
const report = await agent(
  `Synthesize a MAD tuning search for "${baseline.model}" (target: ${target}).
Baseline: ${baseline.baselineSummary}${baseline.baselinePerf != null ? ` (clean baseline: ${baseline.baselinePerf} ${baseline.baselineMetric || ''})` : ''}
Diagnosis: ${diag.bottleneck}-bound — ${diag.evidence}
Top hotspots: ${JSON.stringify((diag.hotspots || []).slice(0, 5))}
Evaluated candidates (JSON, each with a verdict): ${JSON.stringify(clean)}

Produce:
(1) a one-line headline: the bottleneck and the best trustworthy improvement (or that none beat baseline);
(2) the diagnosis summary (what's the bottleneck and why, citing hotspot %s);
(3) a table: Candidate | Change | Evidence | Status | Performance | vs Baseline | Trustworthy;
(4) the recommended configuration — ONLY from candidates whose verdict is trustworthy; if none ran/were trustworthy, present the ranked plan to test on a GPU host;
(5) the exact next command(s) to run.
Each candidate wrote its own perf_tune_<id>.csv (no shared-file clobbering). Do not invent measurements.`,
  { phase: 'Synthesize' }
)

return { model: baseline.model, target, execute, bottleneck: diag.bottleneck, diagnosis: diag, candidates: clean, report }
