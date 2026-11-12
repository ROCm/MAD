export const meta = {
  name: 'mad-nan-hunt',
  description: 'Hunt a NaN in one MAD PyTorch model with nan-doctor: detect it in clean runs, capture the first NaN-producing op with the nan_doctor tool, replay it without the model, and package a standalone reproducer',
  phases: [
    { title: 'Resolve', detail: 'pre-flight, resolve exactly one model, pick the deploy target' },
    { title: 'Setup', detail: 'pinned nan-doctor checkout outside the repo, and the intake; only on execute' },
    { title: 'Detect', detail: 'clean runs; the NaN signal is matched by grep and decided in code' },
    { title: 'Capture', detail: 'one run with the nan_doctor tool; capsules counted, retried once' },
    { title: 'Replay', detail: 'group the capsules and replay the chosen groups under load (debug-nan step 5)' },
    { title: 'Package', detail: 'standalone reproducer for the first reproducing group (step 6)' },
    { title: 'Root-cause', detail: 'debug-nan step 7 within --budget-minutes, only with --rootcause' },
    { title: 'Verify', detail: 'a separate agent re-reads the replay logs and must agree with every verdict' },
    { title: 'Report', detail: 'debug-nan step 9 report, copied to nan_hunts/' },
  ],
}

// ARGS — a structured object or a CLI string:
//   --tag dummy_nan [--nan-doctor-home DIR] [-c '<ctx>'] [--repeats 1]
//   [--skip-detect] [--nan-signal '<ERE>'] [--capture-data shapes|values]
//   [--groups largest|all] [--rootcause --budget-minutes 60] [--timeout S] [--execute]
//
// PLANS BY DEFAULT: resolves the model and prints every command and the intake
// without cloning or touching a GPU. --execute runs them; --plan / --no-execute /
// --dry-run force plan mode.
//
// nan-doctor is the method, madengine the harness (docs: nan-doctor's
// docs/proposals/mad-madengine-integration.md, ADR 0001). The model runs only
// through mad-run.sh; replays only through nan-doctor-run.sh, which holds the
// same host GPU lock. Every phase agent follows nan-doctor's debug-nan skill
// with the intake this script writes (step1-intake.md, env-madengine.md).
//
// DECIDED IN CODE, NOT BY AN AGENT: whether a run NaN'd (grep count of the
// NaN signal in its log), whether a capture produced capsules (file count),
// whether a group reproduced (hits > 0 in its replay), and whether the
// verifier agreed. Agents report counts; this script draws the conclusions.
//
// NOT AUTOMATED: raising the NaN rate by oversubscribing GPUs (debug-nan step 3;
// it fights the GPU lock), the step-5N standalone fallback when no capsule
// replays, and fixes (step 8). Those stay in the interactive /mad-debug-nan.
// K8s is not supported yet.

const NAN_DOCTOR_REPO = 'https://github.com/AMD-BRAIN-Internal/nan-doctor.git'
// 626e1d6: intake schema 1, env-madengine.md with nan-doctor-run.sh.
const NAN_DOCTOR_SHA = '626e1d67ba9ca77848f8fded30342f5bf875e236'
const DEFAULT_SIGNAL = '\\b(loss|grad_norm)\\b.*\\b(nan|inf)\\b'   // ERE, matched case-insensitively
const VERDICTS = ['KERNEL_BUG', 'KERNEL_BUG_UNVERIFIED', 'PROPAGATED', 'FP32_OVERFLOW?', 'NOT_REPRODUCED', 'NOT_SET_UP']

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
    } else if (t === '--nan-doctor-home') {
      cfg.nanDoctorHome = next()
    } else if (t === '--repeats') {
      cfg.repeats = parseInt(next(), 10)
    } else if (t === '--skip-detect') {
      cfg.skipDetect = true
    } else if (t === '--nan-signal') {
      cfg.nanSignal = next()
    } else if (t === '--capture-data') {
      cfg.captureData = next()
    } else if (t === '--groups') {
      cfg.groups = next()
    } else if (t === '--rootcause') {
      cfg.rootcause = true
    } else if (t === '--budget-minutes') {
      cfg.budgetMinutes = parseInt(next(), 10)
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

function safeName(name) {
  return String(name || 'model').replace(/[^A-Za-z0-9._-]/g, '_')
}

function deployTargetOf(ctx) {
  const c = ctx || {}
  if (c.llm_d || c.k8s || c.kubernetes) return 'k8s'
  return c.slurm ? 'slurm' : 'local'
}

// Decisions. Pure functions of what the agents counted.
function nanDetected(runs) {
  return (runs || []).filter(r => r && Number(r.signalLines) > 0).length
}
function reproducingGroups(groups) {
  return (groups || []).filter(g => g && Number(g.hits) > 0)
}
function verifierAgrees(groups, checks) {
  const byGroup = new Map((checks || []).map(c => [String(c.group), c]))
  return (groups || []).every(g => {
    const c = byGroup.get(String(g.group))
    return !!c && c.hitsAgree === true && c.verdictAgree === true
  })
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
  throw new Error('mad-nan-hunt needs one model tag, e.g. --tag dummy_nan [--execute]')
if (cfg._extraTags && cfg._extraTags.length)
  throw new Error(`mad-nan-hunt takes exactly one tag; got extra [${cfg._extraTags.join(', ')}].`)

const tag = cfg.tag
const execute = cfg.execute === true
const timeout = cfg.timeout || null
const ndHome = cfg.nanDoctorHome || '$HOME/nan-doctor'
let repeats = Number(cfg.repeats == null ? 1 : cfg.repeats)
if (!Number.isFinite(repeats)) repeats = 1
repeats = Math.max(1, Math.min(Math.trunc(repeats), 5))
const skipDetect = cfg.skipDetect === true
const signal = cfg.nanSignal || DEFAULT_SIGNAL
const captureData = cfg.captureData === 'values' ? 'values' : 'shapes'
const groupsPref = cfg.groups === 'all' ? 'all' : 'largest'
const rootcause = cfg.rootcause === true
let budgetMinutes = Number(cfg.budgetMinutes == null ? 60 : cfg.budgetMinutes)
if (!Number.isFinite(budgetMinutes) || budgetMinutes < 1) budgetMinutes = 60
budgetMinutes = Math.trunc(budgetMinutes)
const CAPTURE_ATTEMPTS = 2

const ctxObj = normalizeContext(cfg.additionalContext)
if (ctxObj && ctxObj.tools)
  throw new Error('--additional-context must not carry "tools": the capture run adds nan_doctor itself, and clean runs carry none.')
const deployTarget = deployTargetOf(ctxObj)
if (deployTarget === 'k8s')
  throw new Error('mad-nan-hunt supports local and SLURM runs; K8s is not supported yet.')

const PREFLIGHT = 'bash .claude/skills/mad-common/preflight.sh'
const MAD_RUN = 'bash .claude/skills/mad-common/mad-run.sh'
const ND_RUN = 'bash .claude/skills/mad-common/nan-doctor-run.sh'
const SKILL_MAP = `${ndHome}/.claude/skills/user/debug-nan/SKILL.md`
const cloneCommand = `git clone ${NAN_DOCTOR_REPO} "${ndHome}" && git -C "${ndHome}" checkout ${NAN_DOCTOR_SHA}`
const RUN_HOW = `Each Bash tool call runs exactly one command. A call containing &&, ;, or a newline that joins two commands is denied, and nothing in this session can approve it, so the phase stops. Never join commands.
Run every madengine command through the run helper, never in the foreground (a tool call ends after 10 minutes): \`${MAD_RUN} start <run-id> -- <madengine command>\`, then \`${MAD_RUN} wait <run-id>\` until its first line says MAD_RUN_STATUS=succeeded or failed.
Run every replay through the replay helper, never bare docker: \`${ND_RUN} start <run-id> ${ndHome} replay|node5b KEY=VALUE ...\`, then \`${ND_RUN} wait <run-id>\` until NAN_DOCTOR_RUN_STATUS=succeeded or failed.
Use a fresh run-id per run (append -2, -3, ... if start says it exists). Never edit a tracked MAD file.`

function ctxWith(extra) {
  const obj = { ...(ctxObj || {}), ...(extra || {}) }
  return Object.keys(obj).length ? JSON.stringify(obj) : null
}
function runCmd(payload) {
  const flags = ['--tags ' + shellQuote(tag), '--live-output', '-o perf_nan_hunt.csv']
  if (timeout) flags.push('--timeout ' + timeout)
  return `madengine run ${flags.join(' ')}${payload ? ' --additional-context ' + shellQuote(payload) : ''}`
}
const cleanCommand = runCmd(ctxWith(null))
const captureEnv = { ...((ctxObj && ctxObj.docker_env_vars) || {}) }
if (captureData === 'values') captureEnv.NAN_CAPTURE_DATA = '1'
const captureCommand = runCmd(ctxWith({
  tools: [{ name: 'nan_doctor' }],
  ...(Object.keys(captureEnv).length ? { docker_env_vars: captureEnv } : {}),
}))
const grepCommand = `grep -ciE ${shellQuote(signal)} .mad-runs/<run-id>/log`

if (!execute) log('Plan mode: the clone command, the intake and every run are printed. Nothing is cloned and no GPU run starts. Pass --execute to run them.')
log(`mad-nan-hunt "${tag}". target=${deployTarget}. repeats=${repeats}. capture-data=${captureData}. groups=${groupsPref}. rootcause=${rootcause}. execute=${execute}. nan-doctor=${ndHome}@${NAN_DOCTOR_SHA.slice(0, 7)}.`)

phase('Resolve')
const RESOLVE_SCHEMA = {
  type: 'object',
  required: ['preflightOk', 'resolvedModels', 'model'],
  properties: {
    preflightOk: { type: 'boolean' },
    preflightOutput: { type: 'string' },
    resolvedModels: { type: 'array', items: { type: 'string' } },
    model: { type: 'string' },
    nGpus: { type: 'string' },
    hostGpus: { type: 'integer' },
  },
}
const resolved = await agent(
  `Resolve one MAD model for mad-nan-hunt. Run from the MAD repo root.
1. Run \`${PREFLIGHT}\` exactly as written, as its own command. If it exits non-zero, return preflightOk false with its output and stop.
2. Run \`madengine discover --tags ${shellQuote(tag)} --json\` (read-only). Put every model "name" in resolvedModels.
3. If there is exactly one model, return its name as model and its n_gpus string as nGpus.
4. Run \`amd-smi list\` (or \`rocm-smi\` if amd-smi is missing) once and return the number of GPUs as hostGpus; 0 if neither works.
Do not build an image and do not start a run.`,
  { label: 'resolve', phase: 'Resolve', schema: RESOLVE_SCHEMA, agentType: 'mad-nan-doctor' }
)
if (!resolved) throw new Error('Resolve step returned no result; nothing was run.')
if (!resolved.preflightOk) throw new Error(`Pre-flight failed; nothing was run.\n${resolved.preflightOutput || ''}`)
const resolvedModels = resolved.resolvedModels || []
if (resolvedModels.length !== 1)
  throw new Error(`mad-nan-hunt takes exactly one model, but "${tag}" resolves to ${resolvedModels.length}: [${resolvedModels.join(', ')}]. Pass one model name.`)

const modelName = resolved.model || resolvedModels[0]
const nGpus = parseInt(resolved.nGpus, 10)
const perNode = nGpus > 0 ? nGpus : Math.max(1, Number(resolved.hostGpus) || 1)
const nnodes = Math.max(1, Number((ctxObj && ctxObj.slurm && ctxObj.slurm.nodes) || 1))
const huntName = `mad-${new Date().toISOString().slice(0, 10)}-${safeName(modelName)}`
const artifactDir = `${ndHome}/artifacts/${huntName}`
const nanHuntDir = `nan_hunts/${huntName}`
const intakeCtx = { ...(ctxObj || {}) }
const intake = {
  schema: 1,
  source: 'mad',
  env: 'madengine',
  model_tag: modelName,
  run: `madengine run --tags ${modelName} --live-output`,
  additional_context: intakeCtx,
  deploy_target: deployTarget,
  nan_signal: { regex: '(?i)' + signal },
  world_size: perNode * nnodes,
  nnodes,
  artifact_dir: artifactDir,
  preferences: {
    measure_rate: !skipDetect && repeats > 1,
    capture_data: captureData,
    rootcause,
    fix: false,
    groups: groupsPref,
    budget_minutes: rootcause ? budgetMinutes : 0,
  },
}

function stopped(status, reason, extra) {
  log(reason)
  return { execute, status, reason, model: modelName, deployTarget, nanDoctorHome: ndHome, nanDoctorSha: NAN_DOCTOR_SHA, artifactDir, nanHuntDir, ...extra }
}

if (!execute) {
  log(`Clone, only when ${SKILL_MAP} is missing: ${cloneCommand}`)
  log(`Intake ${artifactDir}/intake.json: ${JSON.stringify(intake)}`)
  if (!skipDetect) for (let i = 1; i <= repeats; i++) log(`Detect ${i}/${repeats} (clean): ${MAD_RUN} start nan_detect_${i} -- ${cleanCommand}; then ${grepCommand}`)
  log(`Capture (up to ${CAPTURE_ATTEMPTS} runs): ${MAD_RUN} start nan_capture_1 -- ${captureCommand}`)
  log(`Replay (debug-nan step 5, groups=${groupsPref}): ${ND_RUN} start <run-id> ${ndHome} node5b CAPSULE=<capsule> OUT_DIR=${artifactDir}/5b/<group> IMAGE=<docker_image from build_manifest.json>`)
  log(`Package (step 6) and report (step 9) into ${artifactDir}; copies in ${nanHuntDir}/`)
  if (rootcause) log(`Root-cause (step 7) within ${budgetMinutes} minutes.`)
  return {
    execute: false, status: 'planned', model: modelName, deployTarget,
    nanDoctorHome: ndHome, nanDoctorSha: NAN_DOCTOR_SHA, cloneCommand,
    artifactDir, nanHuntDir, intake,
    detectCommands: skipDetect ? [] : Array.from({ length: repeats }, () => cleanCommand),
    captureCommand, grepCommand,
  }
}

phase('Setup')
const SETUP_SCHEMA = {
  type: 'object',
  required: ['status'],
  properties: {
    status: { type: 'string', enum: ['ok', 'error'] },
    head: { type: 'string' },
    intakeValid: { type: 'boolean' },
    intakeOutput: { type: 'string' },
    reason: { type: 'string' },
  },
}
const setup = await agent(
  `Prepare nan-doctor for mad-nan-hunt. Run from the MAD repo root. ${RUN_HOW}
NAN_DOCTOR_HOME is ${ndHome}. The pinned commit is ${NAN_DOCTOR_SHA}. The file that must exist is ${SKILL_MAP}.
1. If ${SKILL_MAP} exists and \`git -C ${shellQuote(ndHome)} rev-parse HEAD\` is ${NAN_DOCTOR_SHA}, go to step 4. Do not fetch and do not move the checkout.
2. If ${ndHome} does not exist, run \`git clone ${NAN_DOCTOR_REPO} ${shellQuote(ndHome)}\`, then \`git -C ${shellQuote(ndHome)} checkout ${NAN_DOCTOR_SHA}\`, as two calls.
3. If ${ndHome} exists but HEAD is not ${NAN_DOCTOR_SHA}, or the file is absent, return status "error" with head and the reason. Do not delete, reset or overwrite that directory.
4. Create ${artifactDir} (and its capsules/ subdirectory), and write this exact JSON to ${artifactDir}/intake.json with the Write tool:
${JSON.stringify(intake, null, 2)}
5. Run \`python3 ${ndHome}/scripts/internal/intake.py ${artifactDir}/intake.json\`. Return intakeValid true only if it exits 0, with its output as intakeOutput.
6. Start the ledger as nan-doctor's step1-intake.md says (\`python3 ${ndHome}/scripts/internal/ledger.py ${artifactDir}/ledger.md start step1\`, then a note naming the intake).
Return status "ok" with head when steps 1-6 succeeded.`,
  { label: 'setup', phase: 'Setup', schema: SETUP_SCHEMA, agentType: 'mad-nan-doctor' }
)
if (!setup || setup.status !== 'ok')
  return stopped('error', `nan-doctor setup failed: ${(setup && setup.reason) || 'no result'}`, { setup })
if (!setup.intakeValid)
  return stopped('error', `The intake did not validate:\n${setup.intakeOutput || ''}`, { setup })

phase('Detect')
const DETECT_SCHEMA = {
  type: 'object',
  required: ['runId', 'status', 'signalLines'],
  properties: {
    runId: { type: 'string' },
    status: { type: 'string' },
    signalLines: { type: 'integer' },
    firstMatch: { type: 'string' },
  },
}
const detectRuns = []
if (!skipDetect) {
  for (let i = 1; i <= repeats; i++) {
    const r = await agent(
      `Run clean NaN check ${i} of ${repeats} for ${modelName}. ${RUN_HOW}
1. Start: ${MAD_RUN} start nan_detect_${i} -- ${cleanCommand}
2. Wait until it finishes. A NaN run usually ends "failed"; that is expected, not an error.
3. Run \`${grepCommand.replace('<run-id>', `<the run-id you used>`)}\` and return its number as signalLines (0 when grep prints 0), the first matching line as firstMatch, and the run's final status.
4. Ledger it: \`python3 ${ndHome}/scripts/internal/ledger.py ${artifactDir}/ledger.md note step2 "clean run <run-id> <status> signal_lines=<n>"\`.
Do not interpret the log beyond that count.`,
      { label: `detect-${i}`, phase: 'Detect', schema: DETECT_SCHEMA, agentType: 'mad-nan-doctor' }
    )
    detectRuns.push(r)
  }
  const hits = nanDetected(detectRuns)
  log(`Detect: the NaN signal appeared in ${hits}/${detectRuns.length} clean run(s).`)
  if (hits === 0)
    return stopped('no-nan', `No clean run showed the NaN signal /${signal}/i in ${detectRuns.length} run(s). Nothing to capture. Raise --repeats, pass --nan-signal, or --skip-detect to capture anyway.`, { detectRuns })
}

phase('Capture')
const CAPTURE_SCHEMA = {
  type: 'object',
  required: ['runId', 'status', 'capsules'],
  properties: {
    runId: { type: 'string' },
    status: { type: 'string' },
    capsules: { type: 'integer' },
    manifest: { type: 'boolean' },
    capsuleDir: { type: 'string' },
    image: { type: 'string' },
  },
}
let capture = null
const captureAttempts = []
for (let k = 1; k <= CAPTURE_ATTEMPTS; k++) {
  capture = await agent(
    `Capture attempt ${k} of ${CAPTURE_ATTEMPTS} for ${modelName} (debug-nan step 4 under env-madengine.md: the nan_doctor tool, no source wiring). ${RUN_HOW}
1. Start: ${MAD_RUN} start nan_capture_${k} -- ${captureCommand}
2. Wait until it finishes ("failed" is expected when the capture stops the run).
3. ${deployTarget === 'slurm' ? 'Find every node_<N>/nan_doctor_output/ of this job under the SLURM output directory.' : 'Read nan_doctor_output/summary.txt in the MAD repo root.'} Copy every capsules/ directory into ${artifactDir}/capsules/<run-id>/ (copy, not move).
4. Return the number of .pt capsule files you copied as capsules, whether a manifest*.json came with them, the directory as capsuleDir, and the model's docker_image from build_manifest.json as image.
5. Ledger: \`python3 ${ndHome}/scripts/internal/ledger.py ${artifactDir}/ledger.md note step4 "capture <run-id> <status> capsules=<n>"\`.`,
    { label: `capture-${k}`, phase: 'Capture', schema: CAPTURE_SCHEMA, agentType: 'mad-nan-doctor' }
  )
  captureAttempts.push(capture)
  if (capture && Number(capture.capsules) > 0) break
}
if (!capture || !(Number(capture.capsules) > 0))
  return stopped('no-capsule', `No capsule in ${CAPTURE_ATTEMPTS} capture run(s). The NaN may need the standalone fallback (debug-nan step 5N): run /mad-debug-nan ${tag}.`, { detectRuns, captureAttempts })
log(`Capture: ${capture.capsules} capsule(s) from ${capture.runId} in ${capture.capsuleDir}; image ${capture.image || '(unknown)'}.`)

phase('Replay')
const REPLAY_SCHEMA = {
  type: 'object',
  required: ['groups'],
  properties: {
    groups: {
      type: 'array',
      items: {
        type: 'object',
        required: ['group', 'verdict', 'hits', 'runs'],
        properties: {
          group: { type: 'string' },
          capsules: { type: 'integer' },
          verdict: { type: 'string' },
          hits: { type: 'integer' },
          runs: { type: 'integer' },
          concurrency: { type: 'integer' },
          logDir: { type: 'string' },
        },
      },
    },
  },
}
const replay = await agent(
  `Replay the capsules of ${modelName} (debug-nan step 5). Read ${SKILL_MAP}, then step5-read.md and step5-run.md, with env-madengine.md as the environment. ${RUN_HOW}
Capsules: ${capture.capsuleDir}. Image: ${capture.image || 'the docker_image in build_manifest.json'}. Ledger: ${artifactDir}/ledger.md.
1. Group the capsules (5a) inside the image through the replay helper (\`replay\` job, CMD=the grouping command from step5-run.md).
2. Replay ${groupsPref === 'all' ? 'every replayable group' : 'only the largest replayable group (G1)'} with the \`node5b\` job, OUT_DIR=${artifactDir}/5b/<group>.
3. For each replayed group return: group (G1, G2, ...), capsule count, the verdict from step5-read.md's table (one of ${VERDICTS.join(', ')}), hits and runs from the replay summary, the concurrency, and logDir.
Do not go to step 5N, step 7 or step 8.`,
  { label: 'replay', phase: 'Replay', schema: REPLAY_SCHEMA, agentType: 'mad-nan-doctor' }
)
const groups = (replay && replay.groups) || []
const reproducing = reproducingGroups(groups)
log(`Replay: ${reproducing.length}/${groups.length} group(s) reproduced without the model.`)
if (!reproducing.length)
  return stopped('not-reproduced', `No replayed group reproduced (${groups.map(g => `${g.group} ${g.verdict} ${g.hits}/${g.runs}`).join('; ') || 'no groups'}). Next: /mad-debug-nan ${tag} for pressure (step 5c/5d) or the standalone fallback (5N).`, { detectRuns, captureAttempts, groups })

phase('Package')
const PACKAGE_SCHEMA = {
  type: 'object',
  required: ['status'],
  properties: {
    status: { type: 'string', enum: ['ok', 'error'] },
    tarball: { type: 'string' },
    sha256: { type: 'string' },
    reproduced: { type: 'string' },
    reason: { type: 'string' },
  },
}
const pkg = await agent(
  `Package the reproducer for group ${reproducing[0].group} of ${modelName} (debug-nan step 6). Read step6-finish.md and step6-build.md, with env-madengine.md. ${RUN_HOW}
Its replay: ${JSON.stringify(reproducing[0])}. Image: ${capture.image || 'the docker_image in build_manifest.json'}. Artifact dir: ${artifactDir}.
Build and check the package exactly as step 6 says, through the replay helper for anything that needs a GPU. Return the tarball path, its sha256sum, and the n/N the package's own check reproduced as reproduced.`,
  { label: 'package', phase: 'Package', schema: PACKAGE_SCHEMA, agentType: 'mad-nan-doctor' }
)

let rootCause = null
if (rootcause && pkg && pkg.status === 'ok') {
  phase('Root-cause')
  rootCause = await agent(
    `Root-cause the reproducer ${pkg.tarball} (debug-nan step 7, read step7-rootcause.md). The wall-clock budget is ${budgetMinutes} minutes, already agreed: do not ask for it. ${RUN_HOW}
Stop at the budget. Do not fix anything (no step 8). Return status, the path of ROOT_CAUSE.md, and a two-sentence summary.`,
    {
      label: 'rootcause', phase: 'Root-cause', agentType: 'mad-nan-doctor',
      schema: { type: 'object', required: ['status'], properties: { status: { type: 'string' }, rootCauseFile: { type: 'string' }, summary: { type: 'string' } } },
    }
  )
}

phase('Verify')
const VERIFY_SCHEMA = {
  type: 'object',
  required: ['checks'],
  properties: {
    checks: {
      type: 'array',
      items: {
        type: 'object',
        required: ['group', 'hitsAgree', 'verdictAgree'],
        properties: {
          group: { type: 'string' },
          hitsAgree: { type: 'boolean' },
          verdictAgree: { type: 'boolean' },
          note: { type: 'string' },
        },
      },
    },
  },
}
const verify = await agent(
  `You are an adversarial verifier. Another agent replayed NaN capsules and claims the results below. Check every group against the files, not against the claim. Use Read, Grep and Glob only; run nothing.
For each group read its logDir (summary.txt and the per-copy logs) and ${ndHome}/.claude/skills/user/debug-nan/reference/step5-read.md's verdict table. hitsAgree: the logs show the same hits/runs. verdictAgree: the table, applied to those logs, gives the same verdict. Say why in note when either is false. Default to false when the logs are missing.
Claims: ${JSON.stringify(groups)}`,
  { label: 'verify', phase: 'Verify', schema: VERIFY_SCHEMA, agentType: 'general-purpose' }
)
const verified = verifierAgrees(groups, verify && verify.checks)
log(`Verify: ${verified ? 'the verifier agrees with every replay verdict' : 'the verifier DISAGREES; the verdicts are unverified'}.`)

phase('Report')
const REPORT_SCHEMA = {
  type: 'object',
  required: ['reportPath'],
  properties: { reportPath: { type: 'string' }, copied: { type: 'array', items: { type: 'string' } } },
}
const reportBody = {
  model: modelName, deployTarget, nanDoctorSha: NAN_DOCTOR_SHA, artifactDir,
  detectRuns, captureAttempts, groups, package: pkg, rootCause, verified,
  verifierChecks: (verify && verify.checks) || [],
}
const report = await agent(
  `Close the investigation (debug-nan step 9, read step9-report.md) for ${modelName}. ${RUN_HOW}
Write the report with nan-doctor's make_report.py into ${artifactDir}. Use only the facts below and the ledger; do not invent numbers. Say plainly ${verified ? 'that an independent check agreed with the replay verdicts' : 'that the independent check did NOT agree with the replay verdicts, and where'}. A fix is a proposal, never applied.
Then create ${nanHuntDir}/ in the MAD repo and copy report.md, ledger.md, intake.json and the reproducer tarball into it. Return the report path and the copied paths.
${JSON.stringify(reportBody)}`,
  { label: 'report', phase: 'Report', schema: REPORT_SCHEMA, agentType: 'mad-nan-doctor' }
)

const packaged = !!(pkg && pkg.status === 'ok')
return {
  execute: true,
  status: packaged ? (verified ? 'reproducer' : 'reproducer-unverified') : 'package-failed',
  model: modelName, deployTarget,
  nanDoctorHome: ndHome, nanDoctorSha: NAN_DOCTOR_SHA,
  artifactDir, nanHuntDir,
  detectRuns, captureAttempts, groups,
  package: pkg, rootCause,
  verified, verifierChecks: (verify && verify.checks) || [],
  report,
}
