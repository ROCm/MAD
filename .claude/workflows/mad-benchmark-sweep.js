export const meta = {
  name: 'mad-benchmark-sweep',
  description: 'Fan out a benchmark matrix across MAD models/tags, run (or plan) each with madengine, parse results, and synthesize a comparison table',
  phases: [
    { title: 'Resolve', detail: 'pre-flight once and resolve every tag with madengine discover' },
    { title: 'Run', detail: 'plan cells in parallel, or execute them one at a time' },
    { title: 'Synthesize', detail: 'parse results into a comparison table' },
  ],
}

// ARGS — accepts EITHER:
//   (a) a structured object: { tags: string[], nGpus?: string[], execute?: boolean,
//       additionalContext?: string, timeout?: number }
//   (b) a CLI-style string mirroring `/mad-benchmark`, e.g.
//       --tags pyt_vllm_qwen3-8b,dummy_multi --additional-context '{"gpu_vendor":"AMD",...}' --execute
//
// At least one tag is required. Pass several after --tags and/or comma-separate
// them; each tag becomes one sweep cell. --additional-context (JSON or a Python
// dict literal) is normalized to JSON and threaded into EVERY cell command, so
// HF tokens / docker_env_vars reach each run.
//
// EXECUTE semantics: plans by default — resolves tags and prints each cell's
// command without touching GPUs. Pass --execute (or execute:true) to run them.
// --plan / --no-execute / --dry-run force plan mode. Even when executing, each
// cell first checks for AMD GPUs and self-skips if none.
// Executed cells run ONE AT A TIME: madengine gives every run the GPUs from
// index 0 up and shares working files (perf_entry*, build_manifest.json), so
// parallel runs on one host would collide. Each run also goes through
// .claude/skills/mad-common/mad-run.sh, which detaches it (a tool call ends after
// 10 minutes) and holds a host-wide GPU lock. Planned cells run in parallel.
//
// NOTE: there is intentionally no precision axis. In MAD, precision is NOT a
// runtime `madengine run` flag: for training it is fixed per-model via
// `training_precision` in the model's registry entry (scripts/<dir>/models.json),
// and for inference it is baked into the pre-trained model/image. To compare
// precisions, list the distinct model tags.

// Bash-single-quote an arbitrary string so it survives embedding in a shell command.
function shellQuote(s) {
  return "'" + String(s).replace(/'/g, "'\\''") + "'"
}

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

// Rewrite a Python dict literal (single quotes, True/False/None, trailing
// commas) as JSON. madengine accepts both forms; the workflow needs JSON so it
// can merge keys into the context.
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

// --- parse a CLI string into the structured cfg the sweep uses ---
function parseCli(s) {
  const toks = tokenize(s)
  const cfg = { tags: [], nGpus: [], execute: false }
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
  cfg = { execute: false, ...args }
  if (typeof cfg.tags === 'string') cfg.tags = cfg.tags.split(',').filter(Boolean)
} else {
  cfg = { tags: [], nGpus: [], execute: false }
}

if (!cfg.tags || !cfg.tags.length)
  throw new Error('mad-benchmark-sweep needs at least one tag, e.g. --tags pyt_vllm_qwen3-8b,dummy_multi [--execute]')
const tags = cfg.tags
const nGpus = (cfg.nGpus && cfg.nGpus.length) ? cfg.nGpus : [null]
const execute = cfg.execute === true
const ctxObj = normalizeContext(cfg.additionalContext)
const timeout = cfg.timeout || null
const PREFLIGHT = 'bash .claude/skills/mad-common/preflight.sh'
const MAD_RUN = 'bash .claude/skills/mad-common/mad-run.sh'

// Build the matrix of {tag, nGpus} cells.
const matrix = []
for (const tag of tags)
  for (const g of nGpus)
    matrix.push({ tag, nGpus: g })

log(`Sweep matrix: ${matrix.length} cell(s) across ${tags.length} tag(s) [${tags.join(', ')}]. execute=${execute}${ctxObj ? ', additional-context threaded' : ''}`)
if (!execute) log('Plan mode: commands are resolved and printed, nothing runs. Pass --execute to run them on this host.')

// ── Resolve: one pre-flight, then madengine discover per tag ───────────────
phase('Resolve')
const RESOLVE_SCHEMA = {
  type: 'object',
  required: ['preflightOk', 'tags'],
  properties: {
    preflightOk: { type: 'boolean' },
    preflightOutput: { type: 'string' },
    tags: { type: 'array', items: {
      type: 'object',
      required: ['tag', 'models'],
      properties: {
        tag: { type: 'string' },
        models: { type: 'array', items: { type: 'string' } },
        error: { type: ['string', 'null'] },
      },
    } },
  },
}
const resolved = await agent(
  `Resolve MAD tags for a benchmark sweep. Run from the MAD repo root.
1. Run \`${PREFLIGHT}\`. If it exits non-zero, return preflightOk false with its output and stop.
2. For each tag below, run \`madengine discover --tags <tag> --json\` (read-only, no GPU) and return the "name" of every model it lists. A tag that matches nothing gets an empty list and the error text.
Tags: ${JSON.stringify(tags)}`,
  { label: 'resolve', phase: 'Resolve', schema: RESOLVE_SCHEMA, agentType: 'mad-benchmark-runner' }
)
if (!resolved) throw new Error('Resolve step returned no result; nothing was run.')
if (!resolved.preflightOk) throw new Error(`Pre-flight failed; nothing was run.\n${resolved.preflightOutput || ''}`)
const modelsByTag = {}
for (const t of resolved.tags || []) modelsByTag[t.tag] = t
for (const tag of tags) {
  const r = modelsByTag[tag]
  log(`Tag "${tag}": ${r && r.models.length ? `${r.models.length} model(s) [${r.models.slice(0, 5).join(', ')}${r.models.length > 5 ? ', …' : ''}]` : 'did not resolve'}`)
}

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

function runCell(cell, i) {
  const label = `${cell.tag}${cell.nGpus ? '/' + cell.nGpus + 'gpu' : ''}`
  // Each cell writes its OWN perf file so results never mix in a shared
  // perf.csv. A safe filename is derived from the label.
  const safe = label.replace(/[^A-Za-z0-9._-]/g, '_')
  const outFile = `perf_${safe}.csv`
  const r = modelsByTag[cell.tag]
  if (!r || !r.models.length) {
    return Promise.resolve({
      cell: label, command: '', status: 'error', performance: null, metric: null,
      notes: `tag did not resolve via madengine discover${r && r.error ? ': ' + r.error : ''}`,
      _label: label, _output: outFile,
    })
  }
  const flags = [
    '--tags ' + shellQuote(cell.tag),
    '--live-output',
    '-o ' + outFile,
  ]
  if (timeout) flags.push('--timeout ' + timeout)
  // Thread the user-supplied context into every cell, plus the per-cell
  // n_gpus axis when one is set.
  let ctx = ''
  if (ctxObj || cell.nGpus) {
    const merged = { ...(ctxObj || {}) }
    if (cell.nGpus) merged.n_gpus = String(cell.nGpus)
    ctx = ` --additional-context ${shellQuote(JSON.stringify(merged))}`
  }
  const cmd = `madengine run ${flags.join(' ')}${ctx}`
  const runId = `sweep_${safe}_${i}`

  const action = execute
    ? `If AMD GPUs are present (check rocm-smi/amd-smi), run this command from the MAD repo root through the run helper — never in the foreground, a tool call ends after 10 minutes:
  ${MAD_RUN} start ${runId} -- ${cmd}
(if that run-id already exists, append -2, -3, ... to it), then repeat \`${MAD_RUN} wait <run-id>\` until its first line says MAD_RUN_STATUS=succeeded or failed. This tag resolves to ${r.models.length} model(s). A model may emit a "performance: <value> <unit>" stdout line OR (if it is a multiple_results model) write per-metric rows to its own CSV — in that case report the primary throughput metric. Results land in ${outFile}. If no GPUs, set status "skipped".`
    : `Do NOT execute. Check the command is well-formed (the tag already resolved to ${r.models.length} model(s)). Set status "planned".`

  return agent(
    `MAD benchmark sweep cell "${label}".
Command: ${cmd}
${action}
Return the cell label, the exact command, status, and performance/metric if available.`,
    { label: `bench:${label}`, phase: 'Run', schema: CELL_SCHEMA, agentType: 'mad-benchmark-runner' }
  ).then(res => res
    ? { ...res, _label: label, _output: outFile }
    : { cell: label, command: cmd, status: 'error', performance: null, metric: null,
        notes: 'cell agent returned no result', _label: label, _output: outFile })
}

phase('Run')
let results
if (execute) {
  results = []
  for (let i = 0; i < matrix.length; i++) results.push(await runCell(matrix[i], i))
} else {
  results = await parallel(matrix.map((cell, i) => () => runCell(cell, i)))
}

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
(5) note that each cell wrote a separate perf_<cell>.csv and that they can be
concatenated for a combined report.
Do not invent numbers — only use provided data.`,
  { phase: 'Synthesize', agentType: 'mad-perf-analyst' }
)

return { execute, cells: clean, report: summary }
