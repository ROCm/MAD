export const meta = {
  name: 'mad-benchmark-sweep',
  description: 'Fan out a benchmark matrix across MAD models/tags, run (or plan) each with madengine, parse results, and synthesize a comparison table',
  phases: [
    { title: 'Resolve', detail: 'expand the sweep matrix into concrete madengine invocations' },
    { title: 'Run', detail: 'execute or plan each benchmark in parallel' },
    { title: 'Synthesize', detail: 'parse results into a comparison table' },
  ],
}

// args: { tags: string[], precisions?: string[], nGpus?: string[], execute?: boolean, output?: string }
// Without execute:true (default) agents PLAN and validate commands only — safe on GPU-less hosts.
const cfg = args || {}
const tags = Array.isArray(cfg.tags) ? cfg.tags : (cfg.tags ? [cfg.tags] : ['bert'])
const precisions = cfg.precisions && cfg.precisions.length ? cfg.precisions : [null]
const nGpus = cfg.nGpus && cfg.nGpus.length ? cfg.nGpus : [null]
const execute = cfg.execute === true

// Build the matrix of {tag, precision, nGpus} cells.
const matrix = []
for (const tag of tags)
  for (const precision of precisions)
    for (const g of nGpus)
      matrix.push({ tag, precision, nGpus: g })

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
  const label = `${cell.tag}${cell.precision ? '/' + cell.precision : ''}${cell.nGpus ? '/' + cell.nGpus + 'gpu' : ''}`
  const flags = [
    '--tags ' + cell.tag,
    '--live-output',
    cfg.output ? '-o ' + cfg.output : '',
  ].filter(Boolean).join(' ')
  const ctxParts = []
  if (cell.nGpus) ctxParts.push(`"n_gpus": "${cell.nGpus}"`)
  const ctx = ctxParts.length ? ` --additional-context '{${ctxParts.join(', ')}}'` : ''
  const cmd = `madengine run ${flags}${ctx}`

  const action = execute
    ? `If AMD GPUs are present (check rocm-smi/amd-smi), RUN this command and parse the "performance: <value> <unit>" line from output. If no GPUs, set status "skipped".`
    : `Do NOT execute. Validate the command is well-formed and the tag resolves via "madengine discover --tags ${cell.tag}". Set status "planned".`

  return agent(
    `MAD benchmark sweep cell "${label}".
Command: ${cmd}
${action}
${cell.precision ? 'Intended precision: ' + cell.precision + ' (confirm the tag/model supports it).' : ''}
Return the cell label, the exact command, status, and performance/metric if available.`,
    { label: `bench:${label}`, phase: 'Run', schema: CELL_SCHEMA }
  ).then(r => ({ ...r, _label: label }))
}))

phase('Synthesize')
const clean = results.filter(Boolean)
const summary = await agent(
  `Synthesize a MAD benchmark sweep into a markdown comparison table.
Cells (JSON): ${JSON.stringify(clean)}
Produce: (1) a table with columns Cell | Command | Status | Performance | Metric;
(2) a short headline (how many planned/ran/skipped/errored);
(3) if any ran, the fastest and slowest cells (respecting that metric units differ).
Do not invent numbers — only use provided data.`,
  { phase: 'Synthesize' }
)

return { execute, cells: clean, report: summary }
