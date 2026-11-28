// status.json as scripts/ledger.py writes it (schema 1), beside <ARTIFACT_DIR>/ledger.md.
export type NdReproducer = {
  tarball: string
  sha256: string
  size: number
  at: string
  group: string | null
  verdict: string | null
  rate: string | null
  host: string | null
  handover_at: string | null
}

export type NdStatus = {
  schema: number
  run: string
  artifact_dir: string
  ledger: string
  updated: string | null
  run_script: string | null
  state: 'running' | 'done' | 'stopped'
  current: { step: string; since: string; job: string | null } | null
  steps: Array<{ step: string; total?: string; skipped?: string }>
  reproducer: NdReproducer | null
  root_cause: { status: 'FOUND' | 'NO_ROOT_CAUSE'; text: string; at: string } | null
  fix: { text: string; at: string; kind: 'FIX' | 'WORKAROUND' | 'NO_FIX'; tarball: string | null } | null
  line: string
}

declare module 'claude-code' {
  interface PluginState {
    'numeric-doctor': { status: NdStatus | null }
  }
}
