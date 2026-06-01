export const meta = {
  name: 'mad-benchmark-sweep',
  description: 'Fan out a benchmark matrix across MAD models/tags, run (or plan) each with madengine, parse results, and synthesize a comparison table',
  phases: [
    { title: 'Resolve', detail: 'expand the sweep matrix into concrete madengine invocations' },
    { title: 'Run', detail: 'execute or plan each benchmark in parallel' },
    { title: 'Synthesize', detail: 'parse results into a comparison table' },
  ],
}

// ARGS — accepts EITHER:
//   (a) a structured object: { tags: string[], nGpus?: string[], execute?: boolean,
//       additionalContext?: string, timeout?: number }
//   (b) a CLI-style string mirroring `/mad-benchmark`, e.g.
//       --tags pyt_vllm_qwen3-8b,bert --additional-context '{"gpu_vendor":"AMD",...}' --live-output
//
// Multiple tags: pass several after --tags and/or comma-separate them. Each tag
// becomes one sweep cell. --additional-context is threaded verbatim into EVERY
// cell command (so HF tokens / docker_env_vars reach each run).
//
// EXECUTE semantics: runs for real by default. Pass --no-execute (or --plan, or
// execute:false) to only validate commands + resolve tags without touching GPUs.
// Even when executing, each cell first checks for AMD GPUs and self-skips if none.
//
// NOTE: there is intentionally no precision axis. In MAD, precision is NOT a
// runtime `madengine run` flag: for training it is fixed per-model via
// `training_precision` in models.json, and for inference it is baked into the
// pre-trained model/image. To compare precisions, list the distinct model tags.

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

// --- parse a CLI string into the structured cfg the sweep uses ---
function parseCli(s) {
  const toks = tokenize(s)
  const cfg = { tags: [], nGpus: [], execute: true }
  for (let i = 0; i < toks.length; i++) {
    const t = toks[i]
    const next = () => toks[++i]
    if (t === '--tags' || t === '-t') {
      // collect following non-flag tokens, comma-splitting each
      while (i + 1 < toks.length && !toks[i + 1].startsWith('-')) {
        next().split(',').forEach(x => { if (x) cfg.tags.push(x) })
      }
    } else if (t === '--additional-context' || t === '-c') {
      cfg.additionalContext = next()
    } else if (t === '--ngpus' || t === '--n-gpus') {
      next().split(',').forEach(x => { if (x) cfg.nGpus.push(x) })
    } else if (t === '--timeout') {
      cfg.timeout = next()
    } else if (t === '-o' || t === '--output') {
      cfg.output = next()
    } else if (t === '--no-execute' || t === '--plan' || t === '--dry-run') {
      cfg.execute = false
    } else if (t === '--execute') {
      cfg.execute = true
    }
    // --live-output and unknown flags are ignored (live-output is always added below)
  }
  return cfg
}

// --- normalize args (string | object | undefined) into one cfg ---
let cfg
if (typeof args === 'string') {
  cfg = parseCli(args)
} else if (args && typeof args === 'object') {
  cfg = { execute: true, ...args }
  if (typeof cfg.tags === 'string') cfg.tags = [cfg.tags]
} else {
  cfg = { tags: [], nGpus: [], execute: true }
}

const tags = (cfg.tags && cfg.tags.length) ? cfg.tags : ['bert']
const nGpus = (cfg.nGpus && cfg.nGpus.length) ? cfg.nGpus : [null]
const execute = cfg.execute !== false
const addlCtx = cfg.additionalContext || null
const timeout = cfg.timeout || null

// Build the matrix of {tag, nGpus} cells.
const matrix = []
for (const tag of tags)
  for (const g of nGpus)
    matrix.push({ tag, nGpus: g })

log(`Sweep matrix: ${matrix.length} cell(s) across ${tags.length} tag(s) [${tags.join(', ')}]. execute=${execute}${addlCtx ? ', additional-context threaded' : ''}`)

const CELL_SCHEMA = {
  type: 'object',
  required: ['cell', 'command', 'status'],
  properties: {
    cell: { type: 'string' },
    command: { type: 'string' },
    status: { type: 'string', enum: ['planned', 'ran', 'skipped', 'error'] },
    performance: { type: ['number', 'null'] },
    metric: { type: ['string', 'null'] },
    notes: { type: 'string' },
  },
}

const results = await parallel(matrix.map((cell, i) => () => {
  const label = `${cell.tag}${cell.nGpus ? '/' + cell.nGpus + 'gpu' : ''}`
  // Each cell writes its OWN perf file so parallel runs never clobber a shared
  // perf.csv. A safe filename is derived from the label.
  const safe = label.replace(/[^A-Za-z0-9._-]/g, '_')
  const outFile = `perf_${safe}.csv`
  const flags = [
    '--tags ' + cell.tag,
    '--live-output',
    '-o ' + outFile,
  ]
  if (timeout) flags.push('--timeout ' + timeout)
  // Thread the user-supplied --additional-context verbatim into every cell. If a
  // per-cell n_gpus axis is set, try to merge it into the JSON; fall back to the
  // raw string + a separate context if merge isn't possible.
  let ctx = ''
  if (addlCtx) {
    let merged = addlCtx
    if (cell.nGpus) {
      try {
        const obj = JSON.parse(addlCtx)
        obj.n_gpus = cell.nGpus
        merged = JSON.stringify(obj)
      } catch (e) { /* leave raw; n_gpus axis ignored for this cell */ }
    }
    ctx = ` --additional-context '${merged}'`
  } else if (cell.nGpus) {
    ctx = ` --additional-context '{"n_gpus": "${cell.nGpus}"}'`
  }
  const cmd = `madengine run ${flags.join(' ')}${ctx}`

  const action = execute
    ? `If AMD GPUs are present (check rocm-smi/amd-smi), RUN this command from the /home/ysha/MAD repo root and parse results. This model may emit a "performance: <value> <unit>" stdout line OR (if it is a multiple_results model) write per-metric rows to its own CSV — in that case report the primary throughput metric. Results land in ${outFile}. If no GPUs, set status "skipped".`
    : `Do NOT execute. Validate the command is well-formed and the tag resolves via "madengine discover --tags ${cell.tag}". Set status "planned".`

  return agent(
    `MAD benchmark sweep cell "${label}".
Command: ${cmd}
${action}
Return the cell label, the exact command, status, and performance/metric if available.`,
    { label: `bench:${label}`, phase: 'Run', schema: CELL_SCHEMA }
  ).then(r => ({ ...r, _label: label, _output: outFile }))
}))

phase('Synthesize')
const clean = results.filter(Boolean)
const summary = await agent(
  `Synthesize a MAD benchmark sweep into a markdown comparison table.
Cells (JSON, each with its own _output perf file): ${JSON.stringify(clean)}
Produce: (1) a table with columns Cell | Command | Status | Performance | Metric;
(2) a short headline (how many planned/ran/skipped/errored);
(3) explicitly flag any cell whose status is "error" or whose tag did not resolve —
do not let a failed/unresolved cell read as success;
(4) if any ran, the fastest and slowest cells (respecting that metric units differ);
(5) note that each cell wrote a separate perf_<cell>.csv (to avoid clobbering) and
that they can be concatenated for a combined report.
Do not invent numbers — only use provided data.`,
  { phase: 'Synthesize' }
)

return { execute, cells: clean, report: summary }
