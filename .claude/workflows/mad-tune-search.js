export const meta = {
  name: 'mad-tune-search',
  description: 'Generate N candidate tuning configs for one MAD model, run (or plan) each, adversarially verify perf deltas, and synthesize the winning config',
  phases: [
    { title: 'Baseline', detail: 'read the model + establish the baseline config' },
    { title: 'Propose', detail: 'generate distinct tuning candidates' },
    { title: 'Evaluate', detail: 'run or plan each candidate and verify its delta' },
    { title: 'Synthesize', detail: 'pick the winning configuration' },
  ],
}

// args: { tag: string, target?: 'throughput'|'latency', candidates?: number, execute?: boolean }
const cfg = args || {}
const tag = cfg.tag || 'bert'
const target = cfg.target || 'throughput'
const nCandidates = Math.max(1, Math.min(cfg.candidates || 4, 8))
const execute = cfg.execute === true

phase('Baseline')
const BASELINE_SCHEMA = {
  type: 'object',
  required: ['model', 'scriptPath', 'baselineSummary'],
  properties: {
    model: { type: 'string' },
    scriptPath: { type: 'string' },
    baselineSummary: { type: 'string' },
    levers: { type: 'array', items: { type: 'string' } },
  },
}
const baseline = await agent(
  `In the MAD repo, establish the tuning baseline for tag "${tag}".
Find its models.json entry, its scripts/.../run.sh, and any config it references.
Summarize the current configuration and list the tuning levers available for this
stack (env vars like MAD_MODEL_BATCH_SIZE / PYTORCH_TUNABLEOP_ENABLED / NCCL_*,
or args like tensor-parallel size, precision, gpu-memory-utilization).
Optimization target: ${target}.`,
  { phase: 'Baseline', schema: BASELINE_SCHEMA }
)

phase('Propose')
const CANDS_SCHEMA = {
  type: 'object',
  required: ['candidates'],
  properties: {
    candidates: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'change', 'hypothesis'],
        properties: {
          id: { type: 'string' },
          change: { type: 'string' },
          hypothesis: { type: 'string' },
        },
      },
    },
  },
}
const proposed = await agent(
  `Propose ${nCandidates} DISTINCT tuning candidates to improve ${target} for model
"${baseline.model}". Baseline: ${baseline.baselineSummary}
Available levers: ${(baseline.levers || []).join(', ')}
Each candidate changes ONE lever (so its effect is attributable). Return id,
change (concrete value + how to apply it), and hypothesis.`,
  { phase: 'Propose', schema: CANDS_SCHEMA }
)
const candidates = (proposed.candidates || []).slice(0, nCandidates)
log(`Evaluating ${candidates.length} candidate(s). execute=${execute}`)

// Pipeline: evaluate each candidate, then adversarially verify its claimed delta.
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

const evaluated = await pipeline(
  candidates,
  (cand) => {
    const action = execute
      ? `If AMD GPUs are present, apply the change, run "madengine run --tags ${tag} --live-output", parse the performance line, then REVERT the change. If no GPUs, status "skipped".`
      : `Do NOT execute. Produce the exact "madengine run --tags ${tag} ..." command (plus how to apply the change) and set status "planned".`
    return agent(
      `Evaluate tuning candidate ${cand.id} for "${baseline.model}".
Change: ${cand.change}
Hypothesis: ${cand.hypothesis}
${action}
Return id, status, command, and performance/metric if measured.`,
      { label: `eval:${cand.id}`, phase: 'Evaluate', schema: EVAL_SCHEMA }
    )
  },
  (evalResult, cand) =>
    agent(
      `Adversarially review tuning candidate ${cand.id}.
Claimed result: ${JSON.stringify(evalResult)}
Hypothesis was: ${cand.hypothesis}
Is the conclusion trustworthy? Default to trustworthy=false if the change was
not actually measured (planned/skipped), if only one sample was taken, or if the
delta is within noise. Return id, trustworthy, reason.`,
      { label: `verify:${cand.id}`, phase: 'Evaluate', schema: VERDICT_SCHEMA }
    ).then(v => ({ ...evalResult, verdict: v }))
)

phase('Synthesize')
const clean = evaluated.filter(Boolean)
const report = await agent(
  `Synthesize a MAD tuning search for "${baseline.model}" (target: ${target}).
Baseline: ${baseline.baselineSummary}
Evaluated candidates (JSON, each with a verdict): ${JSON.stringify(clean)}
Produce: (1) a table Candidate | Change | Status | Performance | Trustworthy;
(2) the recommended configuration — only from candidates whose verdict is
trustworthy; if none ran, present the ranked plan to test on a GPU host;
(3) the exact next command(s) to run.
Do not invent measurements.`,
  { phase: 'Synthesize' }
)

return { model: baseline.model, target, execute, candidates: clean, report }
