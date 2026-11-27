import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register } from 'claude-code'

import type { NdStatus } from '../types'

// The investigation this session runs: scripts/ledger.py writes its artifact dir under the
// session id (SESSIONS_TMP there, and <artifacts>/.sessions/), and status.json beside its ledger.
const PANE = 'numeric-doctor'
const SESSIONS_TMP = '/tmp/numeric-doctor-sessions'
const POLL_MS = 3000
const status = atom({ plugin: 'numeric-doctor', key: 'status' } as const, null)

function parent(dir: string): string {
  const i = dir.replace(/\/+$/, '').lastIndexOf('/')
  return i <= 0 ? '/' : dir.slice(0, i)
}

function mb(bytes: number): string {
  return `${(bytes / 1e6).toFixed(1)} MB`
}

// the artifact dir this session's ledger.py calls named: /tmp first, then <ancestor>/artifacts/.sessions
async function findRun($: EngineInterface): Promise<string | null> {
  const sid = await $.session.id()
  const candidates = [`${SESSIONS_TMP}/${sid}`]
  for (let dir = await $.session.cwd(), i = 0; i < 8; dir = parent(dir), i++) {
    candidates.push(`${dir}/artifacts/.sessions/${sid}`)
    if (dir === '/') break
  }
  for (const p of candidates) {
    if (await $.fs.exists(p)) return String(await $.fs.read(p)).trim() || null
  }
  return null
}

export const register: Register = on => {
  let artDir: string | null = null
  let lastText = ''
  let opened = false

  on('session.start', async ($, e, next) => {
    await $.command.register({
      name: 'numeric-doctor',
      description: 'Show the debug-numeric investigation of this session in a pane',
    })
    let polls = 0
    $.clock.every(POLL_MS, () => {
      void (async () => {
        // the session's run can change (a new investigation): look it up again every ~30 s
        if (!artDir || polls++ % 10 === 0) artDir = (await findRun($)) ?? artDir
        if (!artDir) return
        const path = `${artDir}/status.json`
        if (!(await $.fs.exists(path))) return
        const text = String(await $.fs.read(path))
        if (text === lastText) return
        lastText = text
        let st: NdStatus
        try {
          st = JSON.parse(text) as NdStatus
        } catch {
          return // a half-written file: the next poll reads it whole
        }
        await update($, status, () => st)
        $.ui.status(st.line)
        if (!opened) {
          opened = true
          void $.ui.open({ id: PANE, title: 'Numeric Doctor' })
        }
      })()
    })

    return next(e)
  })

  on('command.run', { command: 'numeric-doctor' }, async $ => {
    await $.ui.open({ id: PANE, title: 'Numeric Doctor' })
    const st = await read($, status)

    return { text: st ? `Numeric Doctor: ${st.line}` : 'Numeric Doctor: no investigation in this session yet.' }
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const { Box, Text } = $.ui.resolve(e)
    const st = await read($, status)
    if (!st) {
      return (
        <Box flexDirection="column">
          <Text dimColor>No debug-numeric investigation in this session yet.</Text>
        </Box>
      )
    }
    const rp = st.reproducer
    const rc = st.root_cause
    const fx = st.fix
    const cur = st.current
    const head =
      st.state === 'running'
        ? `● ${cur ? cur.step : 'running'}${cur?.job ? `  job ${cur.job}` : ''}`
        : st.state === 'done'
          ? '✓ done'
          : '■ stopped'

    return (
      <Box flexDirection="column" gap={1}>
        <Box flexDirection="column">
          <Text weight="bold">{st.run}</Text>
          <Text color={st.state === 'running' ? 'suggestion' : 'subtle'}>{head}</Text>
        </Box>

        <Box flexDirection="column">
          <Text weight="bold" color={rp ? 'success' : 'inactive'}>
            {rp ? '✅ REPRODUCER' : '… reproducer'}
            {rp ? `  (${rp.at.slice(11, 16)})` : ''}
          </Text>
          {rp && <Text wrap="truncate-start">{rp.tarball}</Text>}
          {rp && (
            <Text dimColor>
              {mb(rp.size)} · sha {rp.sha256.slice(0, 8)}
              {rp.verdict ? ` · ${rp.verdict}` : ''}
            </Text>
          )}
          {rp?.rate && <Text>{rp.rate}{rp.host ? `  on ${rp.host}` : ''}</Text>}
          {rp && <Text dimColor>bash run.sh  (in the extracted dir)</Text>}
        </Box>

        <Box flexDirection="column">
          <Text weight="bold" color={rc ? (rc.status === 'FOUND' ? 'success' : 'warning') : 'inactive'}>
            {rc ? (rc.status === 'FOUND' ? '✅ ROOT CAUSE' : '⚠ NO ROOT CAUSE') : '… root cause'}
            {rc ? `  (${rc.at.slice(11, 16)})` : ''}
          </Text>
          {rc && <Text>{rc.text}</Text>}
        </Box>

        <Box flexDirection="column">
          <Text weight="bold" color={fx ? (fx.kind === 'NO_FIX' ? 'warning' : 'success') : 'inactive'}>
            {fx ? `✅ ${fx.kind}` : '… fix'}
            {fx ? `  (${fx.at.slice(11, 16)})` : ''}
          </Text>
          {fx && <Text>{fx.text}</Text>}
        </Box>

        <Box flexDirection="column">
          {st.steps.map(s => (
            <Text dimColor>
              {s.step.padEnd(12)} {s.total ?? `skipped${s.skipped ? `: ${s.skipped}` : ''}`}
            </Text>
          ))}
          <Text dimColor wrap="truncate-start">ledger: {st.ledger}</Text>
        </Box>
      </Box>
    )
  })
}
