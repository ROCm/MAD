export const meta = {
  name: 'mad-optimize',
  description: 'Profile one MAD model, run GEAK kernel_workflow on its kernel source, and keep the patch only when a clean latency gate passes',
  phases: [
    { title: 'Resolve', detail: 'pre-flight and resolve exactly one model' },
    { title: 'Setup', detail: 'pinned GEAK checkout outside the repo, only on execute' },
    { title: 'Source', detail: 'kernel source from --kernel-path or a copy out of the image' },
    { title: 'Score', detail: 'clean baseline, then the patched tree, two runs each' },
    { title: 'Profile', detail: 'a warm-up, then a rocprofv3 kernel trace for the top kernel, then one counter run for its kernel card and a code-object summary; the card shapes the GEAK task' },
    { title: 'GEAK', detail: 'kernel_workflow writes the host source tree' },
    { title: 'Smoke', detail: 'optional smoke tag only after the gate passes' },
  ],
}

// ARGS — a structured object or a CLI string:
//   --tag pyt_fam_fastwam_geak [--smoke-tag pyt_fam_fastwam_geak2_smoke]
//   [--budget 1] [--repeats 2] [--geak-home DIR] [--kernel-path DIR]
//   [--mount /container=/host] [-c '<ctx>'] [--execute]
//
// Plans by default. --execute clones GEAK when the pinned checkout is missing,
// copies kernel source, measures a clean baseline, profiles, runs
// kernel_workflow, then measures the patched tree. GEAK's geomean does not
// decide. The copy of gate() below must stay the same as mad-tune-search.js.
//
// The Profile phase follows .claude/skills/mad-common/kernel-analysis.md:
// dispatch names the top kernel, one rocprofv3 counter run of that kernel gives
// its kernel card (bound: compute | bandwidth | latency | unknown), and
// kernel-tools.py asm summarizes its code object. The GEAK task names the bound.
// A failed counter run is recorded and the task falls back to the dispatch line.

const GEAK_REPO = 'https://github.com/AMD-AGI/GEAK.git'
// fa86fd1. kernel_workflow/kernel_workflow.js exists on this commit.
const GEAK_SHA = 'fa86fd1e9bec56dadf27d6575a0c8f8ab7875c57'
const MIN_CLAUDE = '2.1.177'
const FAM_ROOT = '/opt/faster-action-models'

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

function parseCli(s) {
  const toks = tokenize(s)
  const cfg = { execute: false }
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
    } else if (t === '--geak-home') {
      cfg.geakHome = next()
    } else if (t === '--kernel-path') {
      cfg.kernelPath = next()
    } else if (t === '--budget') {
      cfg.budget = parseInt(next(), 10)
    } else if (t === '--repeats') {
      cfg.repeats = parseInt(next(), 10)
    } else if (t === '--smoke-tag') {
      cfg.smokeTag = next()
    } else if (t === '--mount') {
      cfg.mount = next()
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

// Same median, spread, and keep rule as mad-tune-search.js. Keep the two copies
// the same: improvement must beat max(baseline spread, candidate spread, 1%).
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

function regexEscape(s) {
  return String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

// Counter group from the kernel name: matrix kernels get the compute group.
function kernelGroupFor(name) {
  return /mm|gemm|Cijk|conv|attn|attention|flash|mfma|bmm|matmul/i.test(String(name || '')) ? 'compute' : 'memory'
}

const BOUND_HINT = {
  compute: 'raise MFMA utilization (tile and MFMA instruction shape, fewer non-matrix instructions in the inner loop)',
  bandwidth: 'cut the bytes it moves (16-byte vector loads, fusing a producer or consumer, LDS reuse)',
  latency: 'raise occupancy or cut stalls and launches (fewer registers or LDS per workgroup, prefetch, fusing small launches)',
}

function geakTaskFor(profile, card) {
  if (!profile.kernelName) return profile.topKernel
  let task = `Optimize the hottest GPU kernel "${profile.kernelName}" (${profile.topKernel}) under the kernel_path tree.`
  if (profile.inSourceTree === false)
    task += ' It is not defined in that tree (likely torch.compile/Inductor generated): optimize the code in kernel_path that produces it, and say so if nothing there can change it.'
  if (card && BOUND_HINT[card.bound])
    task += ` Kernel card: ${card.line || card.bound + '-bound'}. It is ${card.bound}-bound, so ${BOUND_HINT[card.bound]}.`
  return task
}

function parseMount(raw) {
  if (!raw) return null
  const eq = String(raw).indexOf('=')
  if (eq <= 0 || eq === String(raw).length - 1)
    throw new Error('--mount must be container=host, e.g. --mount /opt/src=/home/me/src')
  return { container: raw.slice(0, eq), host: raw.slice(eq + 1) }
}

function containerSource(model, mountContainer) {
  if (mountContainer) return mountContainer
  const blob = [model && model.dockerfile, model && model.name, model && model.args].filter(Boolean).join(' ')
  if (!/pyt_fam|faster-action-models/.test(blob)) return ''
  if (/cae-fastwam|--mode cae|\bcae\b/.test(blob)) return `${FAM_ROOT}/src/cae-fastwam`
  if (/geak/.test(blob)) return `${FAM_ROOT}/src/fastwam-geak`
  return `${FAM_ROOT}/src/fastwam-baseline`
}

function safeName(name) {
  return String(name || 'model').replace(/[^A-Za-z0-9._-]/g, '_')
}

let cfg
if (typeof args === 'string') {
  cfg = parseCli(args)
} else if (args && typeof args === 'object') {
  cfg = { execute: false, ...args }
} else {
  cfg = { execute: false }
}

if (!cfg.tag)
  throw new Error('mad-optimize needs one model tag, e.g. --tag pyt_fam_fastwam_geak [--budget 1] [--execute]')
if (cfg._extraTags && cfg._extraTags.length)
  throw new Error(`mad-optimize takes exactly one tag; got extra [${cfg._extraTags.join(', ')}].`)

const tag = cfg.tag
const execute = cfg.execute === true
const timeout = cfg.timeout || null
const kernelPathArg = cfg.kernelPath || null
const smokeTag = cfg.smokeTag || null
const mount = parseMount(cfg.mount)
let budget = Number(cfg.budget == null ? 1 : cfg.budget)
if (!Number.isFinite(budget) || budget < 1) budget = 1
budget = Math.trunc(budget)
let repeats = Number(cfg.repeats == null ? 2 : cfg.repeats)
if (!Number.isFinite(repeats)) repeats = 2
repeats = Math.max(2, Math.min(Math.trunc(repeats), 5))
const geakHome = cfg.geakHome || '$HOME/GEAK'
const kernelWorkflow = `${geakHome}/kernel_workflow/kernel_workflow.js`
const workflowDir = `${geakHome}/kernel_workflow`
const cloneCommand = `git clone ${GEAK_REPO} "${geakHome}" && git -C "${geakHome}" checkout ${GEAK_SHA}`
const ctxObj = normalizeContext(cfg.additionalContext)

const PREFLIGHT = 'bash .claude/skills/mad-common/preflight.sh'
const MAD_RUN = 'bash .claude/skills/mad-common/mad-run.sh'
const GEAK_RUN = 'bash .claude/skills/mad-common/geak-run.sh'
const RUN_HOW = `Each Bash tool call runs exactly one command. A call containing &&, ;, or a newline that joins two commands is denied, and nothing in this session can approve it, so the phase stops. Never join commands. Never prepend a rocm-smi or GPU check.
Run every madengine command through the run helper, never in the foreground (a tool call ends after 10 minutes). One call:
  ${MAD_RUN} start <run-id> -- <madengine command>
Later calls, one command each: \`${MAD_RUN} wait <run-id>\` until its first line says MAD_RUN_STATUS=succeeded or failed. Use a fresh run-id per run (append -2, -3, ... if start says it exists). GEAK's gpu_lock.sh and MAD's /tmp/mad-gpu.lock do not share a lock, so run one phase at a time.`

function withoutTools(extraMount) {
  const obj = { ...(ctxObj || {}) }
  delete obj.tools
  if (extraMount) {
    obj.docker_mounts = { ...(obj.docker_mounts || {}), [extraMount.container]: extraMount.host }
  }
  return Object.keys(obj).length ? JSON.stringify(obj) : null
}
function ctxFlag(payload) {
  return payload ? ` --additional-context ${shellQuote(payload)}` : ''
}
function runCmd(modelTag, outFile, payload) {
  const flags = ['--tags ' + shellQuote(modelTag), '--live-output', '-o ' + outFile]
  if (timeout) flags.push('--timeout ' + timeout)
  return `madengine run ${flags.join(' ')}${ctxFlag(payload)}`
}

if (!execute) log('Plan mode: the clone command and the GEAK handoff are printed. Nothing is cloned and no GPU run starts. Pass --execute to run them.')
log(`mad-optimize "${tag}". budget=${budget}. repeats=${repeats}. execute=${execute}. geak=${geakHome}.`)

phase('Resolve')
const RESOLVE_SCHEMA = {
  type: 'object',
  required: ['preflightOk', 'resolvedModels', 'model'],
  properties: {
    preflightOk: { type: 'boolean' },
    preflightOutput: { type: 'string' },
    resolvedModels: { type: 'array', items: { type: 'string' } },
    model: { type: 'string' },
    dockerfile: { type: 'string' },
    url: { type: 'string' },
    args: { type: 'string' },
  },
}
const resolved = await agent(
  `Resolve one MAD model for mad-optimize. Run from the MAD repo root.
1. Run \`${PREFLIGHT}\` exactly as written, as its own command. Append nothing (no \`;\`, \`&&\`, \`|\` or \`echo $?\`): the permission rules allow that exact command only, and the Bash result already reports a non-zero exit. If it exits non-zero, return preflightOk false with its output and stop.
2. Run \`madengine discover --tags ${shellQuote(tag)} --json\` (read-only). Put every model "name" in resolvedModels.
3. If there is exactly one model, also return its dockerfile, url, and args from that JSON object.
Do not build an image and do not start a benchmark.`,
  { label: 'resolve', phase: 'Resolve', schema: RESOLVE_SCHEMA, agentType: 'mad-benchmark-runner' }
)
if (!resolved) throw new Error('Resolve step returned no result; nothing was run.')
if (!resolved.preflightOk) throw new Error(`Pre-flight failed; nothing was run.\n${resolved.preflightOutput || ''}`)
const resolvedModels = resolved.resolvedModels || []
if (resolvedModels.length !== 1)
  throw new Error(`mad-optimize takes exactly one model, but "${tag}" resolves to ${resolvedModels.length}: [${resolvedModels.join(', ')}]. Pass one model name.`)

const model = {
  name: resolved.model || resolvedModels[0],
  dockerfile: resolved.dockerfile || '',
  url: resolved.url || '',
  args: resolved.args || '',
}
const containerPath = kernelPathArg ? (mount ? mount.container : containerSource(model, null)) : containerSource(model, mount && mount.container)
const hostPath = kernelPathArg || (mount && mount.host) || `$HOME/.mad-optimize/${safeName(model.name)}/src`
const sourceMount = (containerPath && hostPath) ? { container: containerPath, host: hostPath } : null
const cleanPayload = withoutTools(null)
const patchedPayload = sourceMount ? withoutTools(sourceMount) : cleanPayload
const baseOuts = Array.from({ length: repeats }, (_, k) => `perf_opt_baseline_r${k + 1}.csv`)
const patchedOuts = Array.from({ length: repeats }, (_, k) => `perf_opt_patched_r${k + 1}.csv`)
const baseCommands = baseOuts.map(o => runCmd(model.name, o, cleanPayload))
const patchedCommands = patchedOuts.map(o => runCmd(model.name, o, patchedPayload))
const KERNEL_TOOLS = 'python3 .claude/skills/mad-common/kernel-tools.py'
// The code-object cache runs to hundreds of MB; keep it out of the repo.
const kernelRunDir = `/var/tmp/mad-kernel/opt_${safeName(model.name)}_kernel`
const kernelCtxArg = cleanPayload ? ` --context ${shellQuote(cleanPayload)}` : ''
// Two plain commands, not ctx=$(...): a headless session cannot approve a
// command substitution.
function ctxFile(mode) {
  return `${kernelRunDir}/${mode === 'counters' ? 'context.json' : `context_${mode}.json`}`
}
function ctxRun(outFile, mode) {
  const flags = ['--tags ' + shellQuote(model.name), '--live-output', '-o ' + outFile]
  if (timeout) flags.push('--timeout ' + timeout)
  return `madengine run ${flags.join(' ')} --additional-context-file ${ctxFile(mode)}`
}
// A cold torch.compile traces its own autotuning, so the trace runs on a warmed
// cache; rocprofv3 also records kernels replayed from HIP graphs.
const profileCommand = [
  `${KERNEL_TOOLS} context --mode warm --run-dir ${kernelRunDir}${kernelCtxArg}`,
  ctxRun('perf_opt_warm.csv', 'warm'),
  `${KERNEL_TOOLS} context --mode dispatch --run-dir ${kernelRunDir}${kernelCtxArg}`,
  ctxRun('perf_opt_profile.csv', 'dispatch'),
  `${KERNEL_TOOLS} rank --run-dir ${kernelRunDir}`,
].join('\n')
function kernelCommands(kernelName) {
  const kernelArg = kernelName ? shellQuote(regexEscape(kernelName)) : "'<top dispatch kernel>'"
  const group = kernelName ? kernelGroupFor(kernelName) : "'<compute|memory>'"
  return [
    `${KERNEL_TOOLS} context --mode counters --group ${group} --kernel ${kernelArg} --run-dir ${kernelRunDir}${kernelCtxArg}`,
    ctxRun('perf_opt_kernel.csv', 'counters'),
    `${KERNEL_TOOLS} card --run-dir ${kernelRunDir}`,
    `${KERNEL_TOOLS} asm --run-dir ${kernelRunDir}${kernelName ? ' --kernel ' + shellQuote(kernelName) : ''}`,
  ]
}

const evalDir = `$HOME/.mad-optimize/${safeName(model.name)}/geak_eval`
function geakInvocation(task) {
  return {
    scriptPath: kernelWorkflow,
    args: {
      kernel_path: hostPath,
      workflow_dir: workflowDir,
      budget,
      gpu_ids: '0',
      task,
      apply_to_original: 'true',
      eval_dir: evalDir,
    },
  }
}
// kernel_workflow.js is a Workflow script. Workflows do not nest and subagents have no Workflow
// tool, so the GEAK phase starts it in its own headless `claude -p` session via geak-run.sh.
const geakLaunch = `${GEAK_RUN} start opt_geak ${geakHome} <args.json>   # then: ${GEAK_RUN} wait opt_geak`

const plannedInvocation = geakInvocation('(top kernel from the dispatch profile)')
log(`Clone, only when ${kernelWorkflow} is missing: ${cloneCommand}`)
log(`Kernel source: container ${containerPath || '(none)'} -> host ${hostPath}`)
log(`GEAK script ${kernelWorkflow} workflow_dir ${workflowDir} budget ${budget}`)
log(`GEAK invocation ${JSON.stringify(plannedInvocation)}`)

function stopped(reason, extra) {
  log(reason)
  return {
    execute, status: 'error', reason, model: model.name,
    cloneCommand, geakHome, kernelWorkflow, workflowDir, budget,
    geakInvocation: plannedInvocation,
    containerPath, hostPath,
    ...extra,
  }
}

if (!execute) {
  for (const c of baseCommands) log(`Baseline (image tree, no source mount): ${c}`)
  log(`Profile: ${profileCommand}`)
  for (const c of kernelCommands(null)) log(`Kernel card (top kernel from the profile): ${c}`)
  for (const c of patchedCommands) log(`Patched (source mounted): ${c}`)
  if (smokeTag) log(`Smoke, only if the gate passes: ${runCmd(smokeTag, 'perf_opt_smoke.csv', patchedPayload)}`)
  return {
    execute: false, status: 'planned', model: model.name,
    cloneCommand, geakHome, kernelWorkflow, workflowDir, budget,
    geakInvocation: plannedInvocation,
    geakLaunch,
    containerPath, hostPath,
    baselineCommands: baseCommands,
    patchedCommands,
    profileCommand,
    kernelCommands: kernelCommands(null),
  }
}

phase('Setup')
const SETUP_SCHEMA = {
  type: 'object',
  required: ['status'],
  properties: {
    status: { type: 'string', enum: ['ok', 'error'] },
    kernelWorkflowPresent: { type: 'boolean' },
    sha: { type: 'string' },
    claudeVersion: { type: 'string' },
    reason: { type: 'string' },
  },
}
const setup = await agent(
  `Set up the GEAK checkout for mad-optimize. Do not run a benchmark. Do not pip install. Do not install or replace Claude Code. Do not invoke e2e_workflow.
GEAK_HOME is ${geakHome}. The pinned commit is ${GEAK_SHA}. The file that must exist is ${kernelWorkflow}.
Run each check below as its own command, exactly as written. Do not chain commands with \`;\`, \`&&\`, \`|\` or a subshell: the permission rules refuse a compound command, and each Bash result already reports its own exit.
1. \`claude --version\` must be at least ${MIN_CLAUDE}. If it is older or missing, return status "error" and name the required version. Do not upgrade it.
2. \`rocminfo\` must mention gfx950, and one of rocprofv3, rocprof-compute, or rocprof must be on PATH. If either is missing, return status "error" and say which check failed.
3. If ${kernelWorkflow} exists and \`git -C ${shellQuote(geakHome)} rev-parse HEAD\` is ${GEAK_SHA}, return status "ok" and kernelWorkflowPresent true. Do not fetch and do not move the checkout.
4. If ${geakHome} does not exist, run: ${cloneCommand}
   Then check that ${kernelWorkflow} exists. If the clone worked but the file is still absent, return status "error", kernelWorkflowPresent false, reason "kernel_workflow.js missing after clone".
5. If ${geakHome} exists but HEAD is not ${GEAK_SHA}, or the file is absent, return status "error". Do not delete or overwrite that directory.
Return status, kernelWorkflowPresent, sha, claudeVersion, and reason.`,
  { label: 'setup', phase: 'Setup', schema: SETUP_SCHEMA, agentType: 'mad-benchmark-runner' }
)
if (!setup || setup.status === 'error' || setup.kernelWorkflowPresent === false)
  return stopped(setup && setup.reason ? setup.reason : 'kernel_workflow.js missing after clone')

phase('Source')
const SOURCE_SCHEMA = {
  type: 'object',
  required: ['status', 'hasSource'],
  properties: {
    status: { type: 'string', enum: ['ok', 'error'] },
    hasSource: { type: 'boolean' },
    hostPath: { type: 'string' },
    containerPath: { type: 'string' },
    suffixes: { type: 'array', items: { type: 'string' } },
    reason: { type: 'string' },
  },
}
const sourcePrompt = kernelPathArg
  ? `Use the host directory ${kernelPathArg} as the kernel source. Do not copy from an image and do not clone. It must contain a .py, .triton, or .hip file. If it contains only binaries or wheels, return status "error", hasSource false, reason "this image has no kernel source; pass --kernel-path".`
  : containerPath
    ? `Copy the in-image tree to the host. Do not clone GEAK. Do not start a benchmark.
Container path: ${containerPath}
Host path: ${hostPath} (expand $HOME).
Find an existing image whose name contains pyt_fam (docker images). If none exists, build with \`madengine build --tags ${shellQuote(tag)}\` and do not run the model.
Then run three separate commands, without a command substitution: docker create <image> (it prints a container id), docker cp "<id>:${containerPath}" "${hostPath}", docker rm <id>.
The copy must contain a .py, .triton, or .hip file. If it does not, return status "error", hasSource false, reason "this image has no kernel source; pass --kernel-path".`
    : model.url
      ? `The model url is ${model.url}. The image cloned that repo when it was built. Find the checkout inside the image and docker cp it to ${hostPath}. Do not clone GEAK and do not start a benchmark. Editable source means a .py, .triton, or .hip file. If you cannot find one, return status "error", hasSource false, reason "this image has no kernel source; pass --kernel-path".`
      : `This model has no --kernel-path, no known in-image tree, and an empty url. Do not copy and do not clone. Return status "error", hasSource false, reason "this image has no kernel source; pass --kernel-path".`
const source = await agent(
  `${sourcePrompt}
Return status, hasSource, hostPath, containerPath, suffixes, and reason.`,
  { label: 'source', phase: 'Source', schema: SOURCE_SCHEMA, agentType: 'mad-benchmark-runner' }
)
if (!source || source.status === 'error' || source.hasSource === false)
  return stopped(source && source.reason ? source.reason : 'this image has no kernel source; pass --kernel-path')

const SAMPLES_SCHEMA = {
  type: 'object',
  required: ['status', 'samples'],
  properties: {
    status: { type: 'string', enum: ['planned', 'ran', 'skipped', 'error'] },
    samples: { type: 'array', items: { type: 'number' } },
    metric: { type: ['string', 'null'] },
    notes: { type: 'string' },
  },
}

phase('Score')
const baselineRun = await agent(
  `Measure a CLEAN baseline of "${model.name}" on the image tree. ${repeats} sequential runs. No tools key and no source mount, so the container imports the files baked into the image.
${baseCommands.map(c => '  ' + c).join('\n')}
${RUN_HOW}
Run ids opt_base_r1, opt_base_r2, ... If no AMD GPUs, status "skipped" and samples [].
Parse the "performance: <value> <unit>" line from each run. A profiled number is not a sample.
Return status, one number per successful run, and the metric unit.`,
  { label: 'baseline', phase: 'Score', schema: SAMPLES_SCHEMA, agentType: 'mad-benchmark-runner' }
)
const baseStats = stats(baselineRun && baselineRun.samples)
if (!baselineRun || baselineRun.status === 'error' || baselineRun.status === 'skipped' || baseStats.n < 2)
  return stopped(`baseline did not produce ${repeats} clean samples`, { baseline: baseStats })

phase('Profile')
const PROFILE_SCHEMA = {
  type: 'object',
  required: ['status', 'topKernel'],
  properties: {
    status: { type: 'string', enum: ['ran', 'skipped', 'error'] },
    topKernel: { type: 'string' },
    kernelName: { type: 'string' },
    inSourceTree: { type: ['boolean', 'null'] },
    performance: { type: ['number', 'null'] },
    metric: { type: ['string', 'null'] },
    notes: { type: 'string' },
  },
}
const profile = await agent(
  `Dispatch profile of "${model.name}". A warm-up run with no profiler, then one rocprofv3 kernel-trace run on the same code-object cache. No source mount.
The first Bash call is the first line below and nothing else. Each later line is its own later call.
${profileCommand}
${RUN_HOW}
Run id opt_warm, then opt_profile. The context line writes the file the madengine line reads. Then run the rank command (no GPU). The warm-up perf number is not a score. Read ${kernelRunDir}/kernel_rank.json and set topKernel to the single hottest kernel line (name and share) and kernelName to just that kernel's name. Then grep -rl kernelName (or its distinctive stem) under ${hostPath}; set inSourceTree true if an editable source file defines or generates it, false if not (for example an Inductor-generated triton_* kernel). The performance value on this run is evidence. Do not treat it as the score.
If no GPU or no trace, status "error" and topKernel "".`,
  { label: 'profile', phase: 'Profile', schema: PROFILE_SCHEMA, agentType: 'mad-benchmark-runner' }
)
if (!profile || profile.status === 'error' || !profile.topKernel)
  return stopped(profile && profile.notes ? profile.notes : 'dispatch profile produced no kernel line', { baseline: baseStats })

const KERNEL_SCHEMA = {
  type: 'object',
  required: ['status'],
  properties: {
    status: { type: 'string', enum: ['ran', 'skipped', 'error'] },
    card: { type: ['object', 'null'], properties: {
      kernel: { type: 'string' },
      bound: { type: 'string', enum: ['compute', 'bandwidth', 'latency', 'unknown'] },
      pctPeakBandwidth: { type: ['number', 'null'] },
      pctPeakMfma: { type: ['number', 'null'] },
      occupancy: { type: ['number', 'null'] },
      occupancyLimiter: { type: ['string', 'null'] },
      asmStatus: { type: ['string', 'null'] },
      line: { type: 'string' },
    } },
    notes: { type: 'string' },
  },
}
let kernelProfile = null
if (profile.kernelName) {
  kernelProfile = await agent(
    `Kernel card for "${profile.kernelName}" in "${model.name}". Read .claude/skills/mad-common/kernel-analysis.md first. One rocprofv3 counter run, no rocm_trace_lite, no source mount (the image tree, same as the baseline).
${kernelCommands(profile.kernelName).join('\n')}
Run the context line, then the madengine line through the run helper (run id opt_kernel), then the card and asm commands (they read files, no GPU). If GPU use stays at 0% for more than a few minutes, the counter run has hung: stop it with the run helper and return status "error".
${RUN_HOW}
rocprofv3 runs the workload once per counter pass, so this takes several times longer than a clean run. Its performance number is not a sample.
If the run fails but ${kernelRunDir}/kernel_card.json already names this kernel, return status "ran" and that card, and put the failure in notes. A killed counter run must not throw away a card from this run directory.
Return status and card {kernel, bound exactly as ${kernelRunDir}/kernel_card.json states it, pctPeakBandwidth, pctPeakMfma, occupancy and occupancyLimiter from kernel_asm_summary.json, asmStatus (its "status"), line: the kernel card line from kernel-analysis.md, numbers from the JSON files only}.`,
    { label: 'kernel', phase: 'Profile', schema: KERNEL_SCHEMA, agentType: 'mad-benchmark-runner' }
  )
}
const card = kernelProfile && kernelProfile.status === 'ran' ? kernelProfile.card || null : null
log(`Kernel card: ${card ? card.line || card.bound : `none (${(kernelProfile && kernelProfile.notes) || 'no kernel name from the profile'})`}`)

phase('GEAK')
const geakTask = geakTaskFor(profile, card)
if (profile.inSourceTree === false)
  log(`Warning: "${profile.kernelName}" was not found in ${hostPath}; GEAK may have nothing editable to patch.`)
const liveInvocation = geakInvocation(geakTask)
const GEAK_SCHEMA = {
  type: 'object',
  required: ['status'],
  properties: {
    status: { type: 'string', enum: ['ok', 'error'] },
    expDir: { type: ['string', 'null'] },
    geomean: { type: ['number', 'null'] },
    reason: { type: 'string' },
  },
}
const geak = await agent(
  `Run GEAK kernel_workflow for "${model.name}". You have no Workflow tool and must not try to use one. kernel_workflow.js is started in its own headless Claude session by the helper ${GEAK_RUN}, which holds the same GPU lock as ${MAD_RUN}.
1. Write /tmp/opt_geak_args.json with one Bash call in exactly this shape. Do not use a heredoc, a brace, or a JSON literal in the command: those are denied and the phase stops. Expand $HOME to an absolute path in kernel_path and eval_dir. The task value is the task field below, as one argument:
python3 -I -c 'import json,sys; json.dump(dict(kernel_path=sys.argv[1], workflow_dir=sys.argv[2], budget=sys.argv[3], gpu_ids=sys.argv[4], task=sys.argv[5], apply_to_original=sys.argv[6], eval_dir=sys.argv[7]), open("/tmp/opt_geak_args.json","w"))' '<kernel_path>' '<workflow_dir>' '<budget>' '<gpu_ids>' '<task>' 'true' '<eval_dir>'
Fields:
${Object.entries(liveInvocation.args).map(([k, v]) => k + ': ' + v).join('\n')}
2. If eval_dir already exists and is not empty, return status "error" with that reason. Do not delete it.
3. Run: ${GEAK_RUN} start opt_geak ${geakHome} /tmp/opt_geak_args.json  (append -2, -3, ... to the run-id if start says it exists). The diff belongs at the eval_dir as final_patch.diff. If the session wrote it one directory above eval_dir, status copies it into eval_dir.
4. Repeat \`${GEAK_RUN} wait <run-id>\` until its first line says GEAK_RUN_STATUS=succeeded or failed. A tool call ends after 10 minutes, so keep calling wait. This can take hours.
5. On succeeded, return status "ok" and expDir = the eval_dir from that line. Read geomean from the log or ${evalDir}/ if GEAK printed one (null if none); it does not decide whether the patch is kept.
6. On failed, return status "error", expDir null, and reason = the status line plus the most telling lines of the log tail (the real cause, for example no Workflow tool or the session returned before the workflow finished).
Do not start e2e_workflow. Do not run madengine in this phase.`,
  { label: 'geak', phase: 'GEAK', schema: GEAK_SCHEMA, agentType: 'mad-benchmark-runner' }
)
const expDir = geak && geak.expDir
if (!geak || geak.status === 'error' || !expDir)
  return stopped(`GEAK failed: ${(geak && geak.reason) || 'no result from the GEAK agent'}`, { baseline: baseStats, geomean: geak && geak.geomean, profile })

phase('Score')
const patchedRun = await agent(
  `Measure the patched tree of "${model.name}". ${repeats} sequential clean runs. The source mount makes ${containerPath || hostPath} inside the container read ${hostPath} on the host. No tools key.
${patchedCommands.map(c => '  ' + c).join('\n')}
${RUN_HOW}
Run ids opt_patched_r1, opt_patched_r2, ... Parse "performance: <value> <unit>". Return status, samples, and metric.`,
  { label: 'patched', phase: 'Score', schema: SAMPLES_SCHEMA, agentType: 'mad-benchmark-runner' }
)
const candStats = stats(patchedRun && patchedRun.samples)
const verdict = gate(baseStats, candStats, baselineRun.metric, patchedRun && patchedRun.metric, false)

let smoke = null
if (verdict.pass && smokeTag) {
  phase('Smoke')
  const smokeCmd = runCmd(smokeTag, 'perf_opt_smoke.csv', patchedPayload)
  smoke = await agent(
    `The latency gate passed. Run the smoke tag once, with the same source mount as the patched score, so the smoke imports the patched tree.
Command: ${smokeCmd}
${RUN_HOW}
Run id opt_smoke. Parse "performance: <value> <unit>". If you already have an unpatched smoke rate for this tag, return it as baselineRate; otherwise null.
A success_rate below that unpatched rate rejects the patch.`,
    { label: 'smoke', phase: 'Smoke', schema: {
      type: 'object',
      required: ['status'],
      properties: {
        status: { type: 'string' },
        performance: { type: ['number', 'null'] },
        metric: { type: ['string', 'null'] },
        baselineRate: { type: ['number', 'null'] },
      },
    }, agentType: 'mad-benchmark-runner' }
  )
}
let kept = verdict.pass
let reason = verdict.reason
if (smoke && typeof smoke.baselineRate === 'number' && typeof smoke.performance === 'number' && smoke.performance < smoke.baselineRate) {
  kept = false
  reason = `${reason}; smoke ${smoke.performance} is below the unpatched rate ${smoke.baselineRate}`
}
if (!verdict.pass) log(`Gate rejected the patch. The smoke tag was not run. ${reason}`)

const reportBody = {
  model: model.name,
  expDir,
  geomean: geak.geomean,
  baseline: baseStats,
  baselineMetric: baselineRun.metric,
  patched: candStats,
  patchedMetric: patchedRun && patchedRun.metric,
  gate: verdict,
  kept,
  reason,
  hostPath,
  containerPath,
  profiledPerformance: profile.performance,
  topKernel: profile.topKernel,
  kernelCard: card,
  geakTask,
  smoke,
}
phase('Smoke')
const report = await agent(
  `Write the mad-optimize report for this one model. Use only the JSON below. Do not invent numbers.
The profiled performance is evidence, not the score. GEAK's geomean does not decide. The gate does.
Say whether the patch is kept, both medians, the spread, the gate reason, the exp dir, the kernel card line as given (or that there was none), and the smoke rate when one was measured.
${JSON.stringify(reportBody)}`,
  { phase: 'Smoke', agentType: 'mad-perf-analyst' }
)

return {
  execute: true,
  status: kept ? 'kept' : 'rejected',
  model: model.name,
  cloneCommand, geakHome, kernelWorkflow, workflowDir, budget,
  geakInvocation: liveInvocation,
  containerPath, hostPath, expDir,
  kernelCard: card,
  baseline: baseStats,
  patched: candStats,
  gate: verdict,
  kept,
  reason,
  smoke,
  report,
}
