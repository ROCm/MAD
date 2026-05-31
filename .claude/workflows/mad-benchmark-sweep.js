export const meta = {
  name: 'mad-benchmark-sweep',
  description: 'Fan out a benchmark matrix across MAD models/tags, run (or plan) each with madengine, parse results, and synthesize a comparison table',
  phases: [
    { title: 'Resolve', detail: 'expand the sweep matrix into concrete madengine invocations' },
    { title: 'Run', detail: 'execute or plan each benchmark in parallel' },
    { title: 'Synthesize', detail: 'parse results into a comparison table' },
  ],
}

// args: { tags: string[], nGpus?: string[], execute?: boolean, output?: string }
// Without execute:true (default) agents PLAN and validate commands only — safe on GPU-less hosts.
//
// NOTE: there is intentionally no precision axis. In MAD, precision is NOT a
// runtime `madengine run` flag: for training it is fixed per-model via
// `training_precision` in models.json, and for inference it is baked into the
// pre-trained model/image. To compare precisions, list the distinct model tags
// in `tags` instead.
const cfg = args || {}
const tags = Array.isArray(cfg.tags) ? cfg.tags : (cfg.tags ? [cfg.tags] : ['bert'])
const nGpus = cfg.nGpus && cfg.nGpus.length ? cfg.nGpus : [null]
const execute = cfg.execute === true

// Build the matrix of {tag, nGpus} cells.
const matrix = []
for (const tag of tags)
  for (const g of nGpus)
    matrix.push({ tag, nGpus: g })

log(`Sweep matrix: ${matrix.length} cell(s) across ${tags.length} tag(s). execute=${execute}`)

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
  ].join(' ')
  const ctxParts = []
  if (cell.nGpus) ctxParts.push(`"n_gpus": "${cell.nGpus}"`)
  const ctx = ctxParts.length ? ` --additional-context '{${ctxParts.join(', ')}}'` : ''
  const cmd = `madengine run ${flags}${ctx}`

  const action = execute
    ? `If AMD GPUs are present (check rocm-smi/amd-smi), RUN this command and parse the "performance: <value> <unit>" line from output. Results land in ${outFile}. If no GPUs, set status "skipped".`
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
