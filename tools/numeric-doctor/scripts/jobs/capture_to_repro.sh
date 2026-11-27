#!/bin/bash
#SBATCH --job-name=nd-capture-to-repro
#SBATCH --nodes=1
#SBATCH --time=02:30:00
# debug-numeric steps 4 -> 5a -> 5b (+ the step-6 verification) in ONE allocation, on any runtime and
# scheduler, so the reproducer does not wait on a queue -- or on anyone reading results --
# between rungs.
#
#   0. preflight   gpu_check.py --torch in the runtime: torch sees the GPUs, the repo is visible
#   1. wave        jobs/wave.sh (SPEC, or an existing WAVE_SCRIPT launcher), early stop at the
#                  first settled capsule; on N > 1 nodes the node-count ladder (NODE_LADDER, below)
#   2. 5a          compare_nan_capsules --recursive [--job] -> groups.json, groups_priority.txt
#   3. package     repro-G<i>/ for the top group (skipping allocator/view false positives and
#                  groups PROPAGATED at capture), tarred and EXTRACTED to verify-G<i>/
#   4. 5b          jobs/replay_ladder.sh FROM THE EXTRACTED COPY -- 5b is also step 6's
#                  verify-from-outside-the-build-dir run
#   5. step 6      README.md (verdict proposed from [C1], marked CHECK), tar, sha256 -> ledger
#   3-5 instead, when no op group is replayable but C1 saved a kernel capsule (capsules_inductor/):
#                  repro-kernel/ (the capsule, replay_inductor_kernel.py, source_graph.txt), replayed
#                  from an extracted copy on one GPU, README (CHECK: the fp64 reference) + tar + sha256
#
#   SPEC=<spec> NAN_SIGNAL='<regex>' [CAPTURE_ENV=...] \
#     ND_ENV=<ARTIFACT_DIR>/env.sh scripts/nd submit --name c2r --gpus 8 --exclusive --step step4 -- capture_to_repro.sh
#
# Knobs: everything wave.sh takes (SPEC | WAVE_SCRIPT + WAVE_LOGS, NAN_SIGNAL (required),
#   NAN_ERROR, CAPTURE_ENV, EARLY_STOP, MIN_CAPSULES, WAIT_CONTROLS, CONTROLS, EARLY_STOP_CMD,
#   LOCATE); every CAPSULE_* knob of replay_ladder.sh; and
#   REPLAY        1 (default) | 0 = stop after the wave
#   REPLAY_GROUP  which priority-ordered replayable group to package and replay (default 1)
#   GROUP         e.g. G2: GROUP-ONLY mode -- skip the wave and 5a (groups_priority.txt must
#                 exist), package GROUP, 5b from the extracted copy, README + tar + sha256
#   COTENANT_FILES  host files a co-tenant reproducer ships in cotenant/ (CAPSULE_COTENANT then
#                 names them as @REPO@/cotenant/<file>; step5-pressure.md)
#   NODE_LADDER   on a job of N > 1 nodes (nd submit --nodes N; a one-line `all` spec written with
#                 $ND_NNODES / $ND_NODE_RANK / $ND_MASTER_ADDR, wave.sh): 1 (default) = steps 1-2
#                 become the node-count ladder -- capture on N, N/2, ... 1 nodes (nodes/n<K>/try<t>/), keep
#                 the fewest nodes whose capture is the same bug as N's (same_site.py), then go on
#                 from that capture; the ledger's CAPTURE_NODES=<K> line.  0 = one wave on N nodes
#   NODE_LADDER_TRIES  attempts per rung before it counts as not reproducing (default 2; the skill
#                 sizes it from the N-node NaN rate, step4-distributed.md)
#   NODE_LADDER_MIN  the fewest nodes to try (default 1): where the data-parallel degree cannot
#                 shrink further without changing another parallelism degree
#   The image comes from the profile (IMAGE / SIF / VENV) -- the same for the wave, the ladder
#   and the package's run.sh.
#
# Last line: CAPTURE_TO_REPRO_RESULT=...  Further groups: <ART>/next_groups.sh holds one
# GROUP-only `nd submit` line per replayable group not yet packaged.
# <ART>/capture.submit: the exact knobs of this job (SPEC, NAN_SIGNAL, CAPTURE_ENV, ...), sourceable,
# with its CAPTURE_TO_REPRO_RESULT appended -- what a later investigation reuses verbatim.
set -uo pipefail
# shellcheck disable=SC1091
source "${ND_REPO_HOST:?run through nd submit, or export ND_REPO_HOST}/scripts/env/lib.sh"
nd_job_init
ART=${ND_ART_HOST:?no artifact dir: use the investigation env.sh as ND_ENV}
if [ -z "${GROUP:-}" ]; then
  [ -n "${SPEC:-}${WAVE_SCRIPT:-}" ] || nd_die "SPEC or WAVE_SCRIPT is required"
  [ -n "${NAN_SIGNAL:-}" ] || nd_die "NAN_SIGNAL is required (the model's own NaN line)"
  nd_check_capture_env
  if [ -n "${SPEC:-}" ]; then
    [ -s "$SPEC" ] || nd_die "SPEC=$SPEC: no such file"
    # A reused capture.submit carries the sha of the spec it ran with: knobs and spec travel together.
    # 2026-11-23: the example's spec (no c0/c1 lines) sat beside an earlier run's knobs (WAIT_ARMS='ctl
    # c0 c1') -- caught by reading a diff, not by the job.
    spec_sha=$(grep -vE '^\s*(#|$)' "$SPEC" | sha256sum | cut -c1-16)
    [ -z "${SPEC_SHA256:-}" ] || [ "$SPEC_SHA256" = "$spec_sha" ] ||
      nd_die "SPEC=$SPEC (trial lines sha $spec_sha) is not the spec these knobs ran with (SPEC_SHA256=$SPEC_SHA256): copy the spec that sat beside the reused capture.submit, or unset SPEC_SHA256 for a new spec"
    spec_arms=" $(grep -vE '^\s*(#|$)' "$SPEC" | cut -d'|' -f2 | sort -u | tr '\n' ' ')"
    for w in ${WAIT_ARMS:-}; do
      [[ $spec_arms == *" $w "* ]] || nd_die "WAIT_ARMS names arm '$w', which no line of SPEC=$SPEC has (arms:$spec_arms)"
    done
  fi
else
  [ -s "$ART/groups_priority.txt" ] || nd_die "GROUP=$GROUP needs $ART/groups_priority.txt (run without GROUP first)"
fi
CAPS=${CAPSULE_DIR:-$ART/capsules}; export CAPSULE_DIR=$CAPS
LEDGER=${LEDGER:-$ART/ledger.md}
tag=$ND_JOB_TAG; host=$(hostname -s)
ART_RT=$(rt_path "$ART"); CAPS_RT=$(rt_path "$CAPS")
say() { echo "[$(date '+%F %T')] $*"; }
LG()  { [ -f "$LEDGER" ] && python3 "$(nd_tool ledger.py)" "$LEDGER" "$@" >/dev/null; return 0; }
finish() {
  LG note capture_to_repro "CAPTURE_TO_REPRO_RESULT=$1  job=$tag host=$host"
  [ -n "${GROUP:-}" ] || [ ! -f "$ART/capture.submit" ] || echo "# CAPTURE_TO_REPRO_RESULT=$1  job=$tag host=$host" >>"$ART/capture.submit"
  echo "CAPTURE_TO_REPRO_RESULT=$1"; exit "${2:-0}"
}
write_next_groups() {
  local nx="$ART/next_groups.sh" g v carry=" "
  # the replay settings this job ran with (a TRIGGER's co-tenant above all: without it a further
  # group replays solo and ends NOT_SET_UP [phase 4, skill run 2])
  for v in CAPSULE_MODE CAPSULE_COTENANT CAPSULE_EXTRA_ARGS CAPSULE_FUTILITY CAPSULE_FUTILITY_GRACE CAPSULE_ITER_COUNT \
           CAPSULE_STOP_HITS CAPSULE_CONCURRENCY COTENANT_FILES; do
    [ -n "${!v:-}" ] && carry+="$v=$(printf '%q' "${!v}") "
  done
  { echo "# GROUP-only 5b+6 jobs, one node each, priority order (written by job $tag)."
    for g in $(grep -vE 'FALSE_POSITIVE\?|PROPAGATED\?' "$ART/groups_priority.txt" 2>/dev/null | cut -d' ' -f1); do
      [ -e "$ART/repro-$g" ] && continue
      echo "GROUP=$g${carry}ND_ENV=${ND_ENV:-} $ND submit --name repro_$g --gpus 8 --exclusive --time 02:00:00 --step 'step5b[$g]' -- capture_to_repro.sh"
    done; } >"$nx"
}
next_count() { awk '/^GROUP=/ {n++} END {print n + 0}' "$ART/next_groups.sh" 2>/dev/null; }

say "host=$host tag=$tag runtime=$RUNTIME art=$ART ${GROUP:+group=$GROUP }capsules=$CAPS"

# The knobs exactly as this job got them: a ledger note abbreviates (a SUSPEND path once lost its
# module that way, and the reuse failed at step 1 of every rank), this file does not.
write_submit() {
  local f="$ART/capture.submit" v val
  { echo "# capture-to-reproducer knobs of job $tag on $host, $(date '+%F %T'), ${SLURM_JOB_NUM_NODES:-1} node(s)."
    echo "# Reuse: copy this file and ITS spec (SPEC_SHA256 checks it) into the new artifact dir, then"
    echo "#   ART=<new artifact dir>; set -a; . \$ART/capture.submit; set +a"
    echo "#   ND_ENV=\$ART/env.sh scripts/nd submit --name c2r --gpus 8 --exclusive --step step4 -- capture_to_repro.sh"
    for v in SPEC WAVE_SCRIPT WAVE_LOGS CONTROL_LOGS NAN_SIGNAL NAN_ERROR CAPTURE_ENV CONTROLS LOCATE \
             EARLY_STOP MIN_CAPSULES WAIT_CONTROLS WAIT_ARMS EARLY_STOP_CMD REPLAY REPLAY_GROUP COTENANT_FILES \
             NODE_LADDER NODE_LADDER_TRIES NODE_LADDER_MIN CAPSULE_MODE CAPSULE_COTENANT CAPSULE_EXTRA_ARGS \
             CAPSULE_FUTILITY CAPSULE_FUTILITY_GRACE CAPSULE_ITER_COUNT CAPSULE_STOP_HITS CAPSULE_CONCURRENCY; do
      [ -n "${!v:-}" ] || continue
      val=${!v}
      if [ "${val#"$ART"/}" != "$val" ]; then      # a file of this artifact dir: of the new one on reuse
        echo "$v=\${ART:?set ART to the new artifact dir}/$(printf '%q' "${val#"$ART"/}")"
      else
        echo "$v=$(printf '%q' "$val")"
      fi
    done
    if [ -n "${spec_sha:-}" ]; then echo "SPEC_SHA256=$spec_sha"; fi   # its trial lines only: comments may change
  } >"$f"
  LG note step4 "knobs saved verbatim: ${f#"$ART"/}"
}
[ -n "${GROUP:-}" ] || write_submit

# ---- 0. preflight -------------------------------------------------------------------------
# gpu_check exits 4 on an untested GPU: a warning (the skill's step0-gates.md), not a failed preflight
"$ND" run --gpus all -- 'python3 -m gpu_check --torch; r=$?; { [ $r = 0 ] || [ $r = 4 ]; } && python3 -c "import importlib.util as u, sys; sys.exit(not u.find_spec(\"replay_nan_op\"))" && echo PREFLIGHT_OK' \
  >"$ART/preflight.$tag.$host.log" 2>&1
grep -q '^PREFLIGHT_OK' "$ART/preflight.$tag.$host.log" || finish "PREFLIGHT_FAILED -- see $ART/preflight.$tag.$host.log" 1
gpu=$(grep -m1 '^GPU=' "$ART/preflight.$tag.$host.log"); dev=$(grep -m1 '^device0' "$ART/preflight.$tag.$host.log")
pstep=step4; [ -n "${GROUP:-}" ] && pstep="step5b[$GROUP]"
LG note "$pstep" "preflight OK on $host: $gpu ($dev)"
grep -q 'VERDICT=UNTESTED' <<<"$gpu" && say "!! untested GPU ($gpu): see the skill's reference/step0-gates.md"

# ---- the wave and the grouping, as functions: the node-count ladder runs them once per rung ---------
# run_wave <dir> <nodes>: wave.sh into <dir>/capsules, <dir>/wave, <dir>/wave_summary.txt (<dir> =
# $ART for a plain capture, as before).  Sets ncap, wstop, wsum, rates.
run_wave() {
  local d=$1 n=$2 caps wout
  caps=$d/capsules; [ "$d" = "$ART" ] && caps=$CAPS
  wout=$d/wave.$tag.$host.out; wsum=$d/wave_summary.$tag.txt
  mkdir -p "$d"
  CAPSULE_DIR=$caps WAVE_DIR=${WAVE_DIR:-$d/wave} WAVE_SUMMARY=$wsum NODES=$n \
    bash "$ND_REPO_HOST/scripts/jobs/wave.sh" >"$wout" 2>&1 &
  wpid=$!
  trap 'say "TERM -- stopping"; kill -TERM $wpid $(jobs -p) 2>/dev/null' TERM INT
  nd_wait_pid "$wpid"
  ncap=$(sed -n 's/^CAPSULES=//p' "$wout" | tail -1); ncap=${ncap:-0}
  wstop=$(sed -n 's/^WAVE_STOP=\([^ ]*\).*/\1/p' "$wout" | tail -1)
  [ -s "$wsum" ] || wsum=$wout
  rates=$(sed -nE 's/^ +(\S+) +rate (\S+) \(excl\. ([0-9]+) ERROR, ([0-9]+) DIED, ([0-9]+) STOPPED\)/\1=\2 err=\3 died=\4 stopped=\5/p' \
            "$wsum" 2>/dev/null | paste -sd ';')
  say "wave done (nodes=$n): $ncap capsule(s), stop=$wstop; $wout summary $wsum"
}
# group_caps <capsule dir> <out dir>: 5a -> <out>/groups.txt, groups.json, groups_priority.txt.
# Priority = capsule count; allocator / view ops are capture false positives.  World size > 1:
# the manifest's origin rank(s) first, represented by the origin rank's own capsule.
group_caps() {
  local caps=$1 out=$2 jobflag=""
  [ -n "$(find "$caps" -name 'manifest*.json' -print -quit 2>/dev/null)" ] && jobflag=--job
  "$ND" run --gpus 0 -- "python3 -m compare_nan_capsules $(rt_path "$caps") --recursive $jobflag --json $(rt_path "$out")/groups.json" \
    >"$out/groups.txt" 2>&1
  sed -i '/^\[nd [0-9:]*\] /d' "$out/groups.txt"      # nd's run banner: the user reads this file
  [ -s "$out/groups.json" ] || return 1
  python3 - "$out/groups.json" "$caps" >"$out/groups_priority.txt" <<'EOF'
import ast, glob, json, os, sys
FP = ('aten.empty', 'aten.new_empty', 'aten.detach', 'aten.alias', 'aten.view', 'aten.as_strided')
# synchronous collectives (op_nan_capture._SYNC_COLLECTIVE_OPS): a group of member ranks' capsules of one call
COLL = ('vllm.all_reduce', 'vllm.all_gather', 'vllm.reduce_scatter')
gs = json.load(open(sys.argv[1]))['groups']
origin = set()
for m in glob.glob(os.path.join(sys.argv[2], '**', 'manifest*.json'), recursive=True):
    try:
        origin |= set(json.load(open(m)).get('verdict', {}).get('origin_ranks') or [])
    except Exception:
        pass
def members(g):
    for s in g.get('operand_sets', []):
        for p, mi in zip(s.get('members', []), s.get('member_info') or [{}] * len(s.get('members', []))):
            yield p, mi
def origin_member(g):
    return next((p for p, mi in members(g) if mi.get('rank') in origin and not mi.get('arrived')
                 and mi.get('nan_origin') != 'uninitialized'), None)
rows = []
for g in sorted(gs, key=lambda g: (origin_member(g) is None, -g['count'])):
    ident = g['identity']
    if isinstance(ident, str):
        try: ident = ast.literal_eval(ident)
        except Exception: ident = {'op': ident}
    op = ident.get('op', '?')
    fin = [o.get('inputs_finite') for s in g.get('operand_sets', []) for o in s.get('outcomes', [])]
    mi = [x for _, x in members(g)]
    om = origin_member(g)
    tag = ' FALSE_POSITIVE?' if op.startswith(FP) or (mi and all(x.get('nan_origin') == 'uninitialized' for x in mi)) else \
          ' COLLECTIVE' if op.startswith(COLL) and not (fin and all(f is False for f in fin)) else \
          ' ORIGIN' if om else \
          ' PROPAGATED?' if (fin and all(f is False for f in fin)) or (mi and all(x.get('arrived') for x in mi)) else ''
    rows.append([g['count'], om or g['representative'], op, tag])
# One COLLECTIVE group per op: its member ranks' inputs differ by design (each rank's partial), so the
# operand digests split one collective call into a group per rank and trial -- one call, one group.
out, seen = [], {}
for r in rows:
    if r[3] == ' COLLECTIVE' and r[2] in seen:
        seen[r[2]][0] += r[0]
        continue
    if r[3] == ' COLLECTIVE':
        seen[r[2]] = r
    out.append(r)
for i, (n, rep, op, tag) in enumerate(out, 1):
    print('G%d %d %s %s%s' % (i, n, rep, op, tag))
EOF
  # groups.txt (compare_nan_capsules) lists a collective's capsules as a group per rank and trial; say
  # what groups_priority.txt made of them, or the two files read as a contradiction
  # (|| true: no COLLECTIVE group is the usual case, and under pipefail grep's 1 was group_caps'
  # return -- GROUPING_FAILED for every single-GPU capture, 2026-11-21)
  { grep ' COLLECTIVE$' "$out/groups_priority.txt" || true; } | while read -r g n _ op _; do
    echo "NOTE (5a): the $op capsules above -- one group per rank and trial, because each rank's input to a"
    echo "  collective differs by design -- are ONE group in groups_priority.txt: $g, $n capsule(s), COLLECTIVE."
  done >>"$out/groups.txt"
}
# The capsule a rung is judged by: the first group that is not an allocator false positive
# (world size > 1: the origin rank's), as a host path.
site_capsule() { rt_hostpath "$(grep -v 'FALSE_POSITIVE?' "$1/groups_priority.txt" | head -1 | cut -d' ' -f3)"; }
# top_capsule_info: capsule_info.py of the top group that is not an allocator false positive --
# replayable or PROPAGATED -- into $ART/capsule_info.<G>.txt, in this allocation: a separate job
# for it cost a fresh node and its image pull (2026-11-19), and 5N starts from it.
top_capsule_info() {
  local top g c f
  top=$(grep -v 'FALSE_POSITIVE?' "$ART/groups_priority.txt" 2>/dev/null | head -1); [ -n "$top" ] || return 0
  g=${top%% *}; c=$(cut -d' ' -f3 <<<"$top"); f=$ART/capsule_info.$g.txt
  "$ND" run --gpus 0 -- "python3 -m capsule_info $c" >"$f" 2>&1
  sed -i '/^\[nd [0-9:]*\] /d' "$f"
  LG note step5a "capsule_info $g: ${f#"$ART"/} ($(grep -m1 '^origin' "$f" | cut -c10-60))"
}

# ---- the runtime a package's run.sh bakes in (5b's repro-<G>, the kernel capsule's repro-kernel) ----
# bake_runtime: img, site_cmd, natv, rt_bake from this job's profile; runtime_lines: the run.sh lines.
bake_runtime() {
  img=${IMAGE:-${SIF:-}}
  # What reproduced, baked into run.sh as defaults [phase 4, skill run 2]: without them a vendor's
  # `sbatch run.sh` (no profile) took RUNTIME=auto -> podman on an apptainer site's node -> a registry pull with
  # no network.  Each default applies only where it can work (the runtime is installed, the SIF
  # exists); the caller's RUNTIME= / SIF= / IMAGE= / APPTAINER_FLAGS= / DOCKER_AS_USER= still override.
  site_gpus=""; [ -n "${SLURM_GPUS_FLAG:-}" ] && site_gpus=${SLURM_GPUS_FLAG//\{n\}/8}
  site_cmd=""; [ "${SCHED:-}" = slurm ] && site_cmd="sbatch ${SLURM_ARGS:-}${site_gpus:+ $site_gpus} run.sh"
  # A native package needs the venv the model ran in -- nothing to auto-detect -- so run.sh records
  # it; EXTRA_ENV (e.g. a library path the model needed) goes in on every runtime: the package's
  # run.sh is verified under env -i below.
  natv=""
  # the SIF the job really used: a profile resolves it from SIF_DIR + IMAGE, so SIF is often
  # unset -- baking only an explicit SIF left the package's run.sh without one
  sif_used=${SIF:-}
  [ "$RUNTIME" = apptainer ] && [ -z "$sif_used" ] && sif_used=$(_apt_sif 2>/dev/null)
  if [ "$RUNTIME" = native ]; then
    img="venv ${VENV:-${CONDA_ENV:+conda:$CONDA_ENV}}"; [ "$img" = "venv " ] && img="host python3 ($(command -v python3))"
    natv="export RUNTIME=\${RUNTIME:-native}   # no container: the model's own Python environment
export VENV=\${VENV:-${VENV:-}}
export CONDA_ENV=\${CONDA_ENV:-${CONDA_ENV:-}}"
  fi
  # Every runtime knob the profile set, so the package's run.sh -- verified below with no profile --
  # runs the container the way the job did: DOCKER_AS_USER=0 alone was once lost and every copy
  # died at `import aiter` (2026-11-19).  Site paths (PROJ, CONTAINER_PROJ, SIF_DIR, EXTRA_MOUNTS)
  # are not baked: another site has other paths; EXTRA_MOUNTS is named in a comment.
  rt_bake=""
  for v in DOCKER_AS_USER DOCKER_PULL DOCKER_SHM DOCKER_ARGS DOCKER_ENTRYPOINT \
           PODMAN_ARGS PODMAN_SHM PODMAN_ENTRYPOINT APPTAINER_FLAGS; do
    [ -n "${!v+x}" ] || continue
    rt_bake+="export $v=\${$v-$(printf '%q' "${!v}")}   # the profile's"$'\n'
  done
  [ -n "${EXTRA_MOUNTS:-}" ] && rt_bake+="# the profile also mounted (site paths, not baked): EXTRA_MOUNTS=$EXTRA_MOUNTS"$'\n'
}
runtime_lines() {
  # a site file only when this job had one: `[ -f "/nonexistent" ] && export SIF=""` read as a
  # broken placeholder to a vendor (2026-11-20)
  local sifl="" tarl=""
  [ -n "${sif_used:-}" ] && sifl="[ -n \"\${SIF:-}\" ] || { [ -f \"$sif_used\" ] && export SIF=\"$sif_used\"; }   # this site's SIF; else built/pulled from IMAGE"
  [ -n "${IMAGE_TAR:-}" ] && tarl="[ -n \"\${IMAGE_TAR:-}\" ] || { [ -f \"$IMAGE_TAR\" ] && export IMAGE_TAR=\"$IMAGE_TAR\"${IMAGE_ID:+ IMAGE_ID=$IMAGE_ID}; }   # this site's image archive (a node without the image, or with another build${IMAGE_ID:+ than IMAGE_ID})"
  cat <<EOF
export ND_REPO_HOST=\$HERE ND_ART_HOST=\$HERE
export IMAGE=\${IMAGE:-${IMAGE:-${SIF:-}}}
# the runtime that reproduced, where it is installed (else the first of docker / podman / apptainer)
[ -z "\${RUNTIME:-}" ] && [ -z "\${ND_ENV:-}" ] && command -v ${RUNTIME:-none} >/dev/null 2>&1 && export RUNTIME=${RUNTIME:-}
${sifl}
${tarl}
[ -n "\${IMAGE_BUILD_DIR:-}" ] || { [ -f "\$HERE/image/Dockerfile" ] && export IMAGE_BUILD_DIR=\$HERE/image; }   # a locally built image: built from image/ where it neither loads nor pulls
$natv
export EXTRA_ENV=\${EXTRA_ENV-'${EXTRA_ENV:-}'}   # env the model needed, passed to every copy
${rt_bake}
EOF
}
# ship_runtime <dir>: the runtime layer (nd + env) and a locally built image's build context
ship_runtime() {
  mkdir -p "$1/scripts/env/profiles" "$1/scripts/jobs"
  cp -- "$ND_REPO_HOST/scripts/nd" "$1/scripts/" && chmod +x "$1/scripts/nd"
  cp -- "$ND_LIB_DIR"/lib.sh "$ND_LIB_DIR"/runtime_*.sh "$ND_LIB_DIR"/sched_*.sh "$ND_LIB_DIR"/zstdcat.py "$1/scripts/env/"
  cp -- "$ND_LIB_DIR"/profiles/local-*.env "$1/scripts/env/profiles/"
  # A locally built image (build_image.sh) is in no registry: ship its build context, so the
  # package's run.sh builds it off-site (runtime_docker.sh IMAGE_BUILD_DIR).  The site's IMAGE_TAR
  # path, baked in, exists only here (2026-11-20, sglang: the package did not run off-site).
  if [ -n "${IMAGE_BUILD_DIR:-}" ] && [ -f "$IMAGE_BUILD_DIR/Dockerfile" ]; then
    cp -r -- "$IMAGE_BUILD_DIR" "$1/image" || return 1
  fi
}

NNODES=${SLURM_JOB_NUM_NODES:-1}
if [ -z "${GROUP:-}" ] && [ "$NNODES" -gt 1 ] && [ "${NODE_LADDER:-1}" = 1 ]; then
  # ---- 1+2. the node-count ladder: the fewest nodes that still capture the same bug ----------------
  # Rung K runs the spec on the job's first K nodes (K = N, N/2, ... 1) into nodes/n<K>/try<t>/,
  # up to NODE_LADDER_TRIES (2) attempts.  A rung passes when it captures and its site capsule
  # is the same bug as rung N's (same_site.py: op + stack digest or model frames); the first rung
  # that fails ends the ladder, and the last rung that passed is replayed and packaged below.
  # Rung N failing every attempt is NO_CAPSULE, as for a plain capture.
  tries=${NODE_LADDER_TRIES:-2}; min=${NODE_LADDER_MIN:-1}; K=$NNODES; best=""; bestdir=""; ref=""
  LG note step4 "node-count ladder: N=$NNODES nodes, tries per rung=$tries, floor=$min"
  while :; do
    passed=""
    for t in $(seq "$tries"); do
      R=$ART/nodes/n$K/try$t
      run_wave "$R" "$K"
      v="NO_CAPSULE"
      if [ "$ncap" -gt 0 ]; then
        if ! group_caps "$R/capsules" "$R"; then v="GROUPING_FAILED ($R/groups.txt)"
        else
          sc=$(site_capsule "$R")
          if [ -z "$sc" ]; then v="NO_SITE (every group FALSE_POSITIVE?)"
          elif [ -z "$ref" ]; then ref=$sc; v="REFERENCE $(head -1 "$R/groups_priority.txt" | cut -d' ' -f4-)"
          else
            v=$("$ND" run --gpus 0 -- "python3 -m same_site $(rt_path "$ref") $(rt_path "$sc")" 2>/dev/null | tee "$R/same_site.txt" | grep '^SITE=' | tail -1)
            v=${v:-SITE=ERROR (see $R/same_site.txt)}
          fi
        fi
      fi
      LG note step4 "node-count ladder n=$K try=$t: capsules=$ncap ${rates:+$rates }-> $v  (${R#"$ART"/})"
      case $v in REFERENCE*|SITE=SAME*) passed=$R; break ;; esac
    done
    [ -n "$passed" ] || break
    best=$K; bestdir=$passed
    [ $((K / 2)) -ge "$min" ] || break
    K=$((K / 2))
  done
  [ -n "$best" ] || { LG end step4; finish "NO_CAPSULE on $NNODES nodes ($tries attempt(s)) -- read $ART/nodes/n$NNODES/try*/wave_summary.*.txt (step5-read.md: empty dir)"; }
  LG note step4 "CAPTURE_NODES=$best (of $NNODES; $([ "$best" = "$K" ] && echo "the ladder reached its floor of $min" || echo "$K node(s) did not capture the same bug in $tries attempt(s)")) -> ${bestdir#"$ART"/}"
  LG end step4
  [ "${REPLAY:-1}" = 1 ] || finish "WAVE_ONLY CAPTURE_NODES=$best capsules in ${bestdir#"$ART"/}/capsules"
  # the passing rung becomes this investigation's capture: its groups at $ART (their capsule
  # paths point into nodes/n<best>/), the rest of the job and the skill read them as usual
  LG start step5a; LG job step5a "$tag" capture-to-reproducer job
  cp -- "$bestdir"/groups.txt "$bestdir"/groups.json "$bestdir"/groups_priority.txt "$ART"/
  CAPS=$bestdir/capsules; export CAPSULE_DIR=$CAPS; CAPS_RT=$(rt_path "$CAPS")
  LG note step5a "groups from the $best-node capture (${bestdir#"$ART"/}): $(wc -l <"$ART/groups_priority.txt") ($(cut -d' ' -f1,2,4 "$ART/groups_priority.txt" | paste -sd ';')) json=groups.json"
  top_capsule_info
  LG end step5a
elif [ -z "${GROUP:-}" ]; then
  # ---- 1. the capture wave ----------------------------------------------------------------
  run_wave "$ART" "$NNODES"
  LG note step4 "wave ${wstop:+stop=$wstop }capsules=$ncap $rates -> $(basename "$wsum")"
  LG end step4
  if [ "$ncap" -eq 0 ]; then
    # A graph-check line (step4-run.md *vLLM with graphs*) that found a NaN only graph replay makes:
    # the capture lines ran graphs off, so their empty dir is that finding, not a broken capture.
    gline=$(grep -h -m1 'GRAPH_CHECK verdict=GRAPH_ONLY' "${WAVE_DIR:-$ART/wave}"/*.log 2>/dev/null | sed 's/^.*GRAPH_CHECK //')
    [ -n "$gline" ] && { LG note step4 "GRAPH_ONLY: $gline"
      finish "NO_CAPSULE GRAPH_ONLY $gline -- the NaN needs graph replay (step5n-compiled.md: Graph-only NaN)"; }
    finish "NO_CAPSULE -- read $wsum (step5-read.md: empty dir)"
  fi
  [ "${REPLAY:-1}" = 1 ] || finish "WAVE_ONLY capsules=$ncap"

  # ---- 2. 5a: group ---------------------------------------------------------------------
  LG start step5a; LG job step5a "$tag" capture-to-reproducer job
  group_caps "$CAPS" "$ART" || finish "GROUPING_FAILED -- see $ART/groups.txt" 1
  LG note step5a "groups=$(wc -l <"$ART/groups_priority.txt") ($(cut -d' ' -f1,2,4 "$ART/groups_priority.txt" | paste -sd ';')) json=groups.json"
  top_capsule_info
  LG end step5a
fi

# ---- NO_REPLAYABLE_GROUP with a kernel capsule (C1 in the wave, step5n-compiled.md) -------------
# Every op capsule PROPAGATED, but C1's Inductor assert saved the kernel that made the NaN: replay it
# here, package repro-kernel/ (step6-finish.md *Kernel capsule reproducer*), verify it from an extracted
# copy -- the agent did all three by hand on 2026-11-20 (vLLM, torchcompile).  Sets KRES for the result
# line.  The verdict still needs the fp64 reference of the kernel's source nodes (5N's job; the
# package's source_graph.txt is Inductor's FX graph of them).
KRES=""
kernel_capsule_repro() {
  local kc K V X kname res hits info x out
  kc=$(ls -t "$ART"/capsules_inductor/inductor_*.pt 2>/dev/null | head -1)
  [ -n "$kc" ] && [ "${REPLAY:-1}" = 1 ] || return 0
  K=$ART/repro-kernel; V=$ART/verify-kernel; out=$ART/replay/kernel; mkdir -p "$out"
  say "no replayable op group; kernel capsule $(basename -- "$kc"): packaging and replaying it"
  LG start "step6[kernel]"; LG job "step6[kernel]" "$tag" capture-to-reproducer job
  for x in "$K" "$V" "$K.tar.gz"; do
    [ -e "$x" ] || continue
    mkdir -p "$ART/old" && mv -- "$x" "$ART/old/$(basename -- "$x").before-$tag" ||
      { KRES=" KERNEL_CAPSULE=REFUSING ($x cannot be moved to old/)"; LG end "step6[kernel]"; return 0; }
  done
  mkdir -p "$K/capsule" && cp -- "$kc" "$K/capsule/" && ship_runtime "$K" &&
    cp -- "$(nd_tool replay_inductor_kernel.py)" "$(nd_tool inductor_ref.py)" "$K/scripts/" ||
    { KRES=" KERNEL_CAPSULE=PACKAGE_FAILED"; LG end "step6[kernel]"; return 0; }
  info=$out/info.$tag.txt
  "$ND" run --gpus 0 -- "python3 -m replay_inductor_kernel $(rt_path "$K/capsule/$(basename -- "$kc")") --info" 2>&1 |
    sed '/^\[nd [0-9:]*\] /d' >"$info"
  kname=$(sed -n 's/^kernel *\([^ ]*\).*/\1/p' "$info" | head -1)
  # Inductor's description of the kernel: from its "Topologically Sorted Source Nodes" comment to its
  # definition in C1's output code (the FX graph of the fused ops -- the recipe for a reference)
  [ -n "$kname" ] && [ -f "$ART/output_code.c1.log" ] && python3 - "$ART/output_code.c1.log" "$kname" >"$K/source_graph.txt" <<'PY'
import re, sys
# torch's log header, after vLLM's [rank0]: prefix (on every line, continuations too)
L = [re.sub(r'^[VIWE]\d{4} .*?\[__output_code\] ?', '', re.sub(r'^\[rank\d+\]:', '', l))
     for l in open(sys.argv[1], errors='replace').read().splitlines()]
d = next((i for i, l in enumerate(L) if l.startswith('%s = async_compile.triton(' % sys.argv[2])), None)
if d is not None:
    s = max((i for i in range(d) if 'Topologically Sorted Source Nodes' in L[i]), default=d)
    print('\n'.join(L[s:d]))
PY
  [ -s "$K/source_graph.txt" ] || rm -f "$K/source_graph.txt"
  bake_runtime
  cat >"$K/run.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=nan-repro-kernel
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=00:30:00
# Reproduce the NaN without the model: recompile the Inductor kernel ${kname:-?} from capsule/ (its
# Triton source, by this image's torch / triton), launch it ITERS times on the model's real inputs on
# one GPU, and count launches whose checked output holds a NaN/Inf.
#
#   bash run.sh          # on a GPU node
#   sbatch [<site args: account, partition, GPU request>] run.sh    # Slurm, from this directory
#   RUNTIME=apptainer|docker|podman|native  IMAGE=...  SIF=...  GPU=0  ITERS=1000
# this site: ${site_cmd:-bash run.sh}
#
# Result: the last line, INDUCTOR_REPLAY_RESULT=<hits>/<iters> ... matches_model=<bool>.  See README.md.
set -uo pipefail
if [ -n "\${SLURM_JOB_ID:-}" ]; then HERE=\$SLURM_SUBMIT_DIR; else HERE=\$(dirname -- "\$0"); fi
HERE=\$(cd -- "\$HERE" && pwd -P) || exit 2
[ -f "\$HERE/scripts/replay_inductor_kernel.py" ] || { echo "run.sh must run from (or be submitted from) its own directory"; exit 2; }
$(runtime_lines)
exec "\$HERE/scripts/nd" run --gpus "\${GPU:-0}" -- "python3 \\\$ND_REPO/scripts/replay_inductor_kernel.py \\\$ND_REPO/capsule/$(basename -- "$kc") --iters \${ITERS:-1000}"
EOF
  chmod +x "$K/run.sh"
  mkdir -p "$V" && tar -cf - -C "$ART" repro-kernel | tar -xf - -C "$V" ||
    { KRES=" KERNEL_CAPSULE=PACKAGE_FAILED (tar/extract)"; LG end "step6[kernel]"; return 0; }
  X=$V/repro-kernel
  # the extracted package's own run.sh in an empty environment, as for 5b (a vendor's `bash run.sh`)
  env -i HOME="$HOME" USER="${USER:-$(id -un)}" LOGNAME="${LOGNAME:-$(id -un)}" PATH="$PATH" \
      LANG="${LANG:-C.UTF-8}" bash -c 'cd -- "$1" && exec bash ./run.sh' _ "$X" >"$out/run.$tag.$host.out" 2>&1
  res=$(sed -n 's/^INDUCTOR_REPLAY_RESULT=//p' "$out/run.$tag.$host.out" | tail -1)
  hits=${res%%/*}; [[ $hits =~ ^[0-9]+$ ]] || hits=0
  LG note "step6[kernel]" "kernel capsule replay from the extracted package on $host: ${res:-no INDUCTOR_REPLAY_RESULT (see ${out#"$ART"/}/run.$tag.$host.out)}"
  if [ "$hits" -gt 0 ]; then
    cat >"$K/README.md" <<EOF
# NaN reproducer: Inductor kernel \`${kname:-?}\`

## What this is

The Triton kernel that \`torch.compile\` (Inductor) generated for part of the model, saved by Inductor's
NaN assert at the launch that produced the NaN: its source, the launch line and every launch argument
-- the inputs with the model's **real values**. \`run.sh\` recompiles the kernel from that source with
this image's torch / triton and launches it on those inputs, **without the model**.

* Capsule: \`capsule/$(basename -- "$kc")\` -- sha256 \`$(sha256sum "$kc" | cut -d' ' -f1)\`
* ${RUNTIME} image: \`$img\`
* What the kernel fuses: \`source_graph.txt\` (Inductor's FX graph of its source nodes)$([ -s "$K/source_graph.txt" ] || echo ' -- not found in the output code')

\`\`\`
$(cat "$info")
\`\`\`

## The one command

\`\`\`bash
bash run.sh            # ${site_cmd:+or: $site_cmd; }GPU=, ITERS=, RUNTIME=, IMAGE= to override
\`\`\`

Measured here, from an extracted copy of this package on node \`$host\`: \`INDUCTOR_REPLAY_RESULT=$res\`.
\`matches_model=True\`: every output of the last launch is bit-identical to the model's.

## Verdict

<!-- CHECK: the kernel turns finite real inputs into NaN on its own. Whether that is the kernel's (or
its lowering's) fault or the arithmetic's needs the fp64 reference of its source nodes on the same
inputs (step5n-compiled.md *Kernel capsule*): fp64 non-finite too = NATIVE_MATH; finite = the kernel. -->
EOF
    python3 "$(nd_tool make_repro_readme.py)" --repro "$K" --repack --ledger "$LEDGER" --step "step6[kernel]" \
      --note "kernel capsule ${kname:-?}: ${res} (README auto; CHECK: the fp64 reference)" >/dev/null
    KRES=" KERNEL_CAPSULE=REPRODUCED $res verified-from-extracted-copy tarball=$K.tar.gz (README auto: its CHECK line needs the fp64 reference of the kernel's source nodes -- step5n-compiled.md *Kernel capsule*, then step6-finish.md *Kernel capsule reproducer*)"
  else
    KRES=" KERNEL_CAPSULE=NOT_REPRODUCED ${res:-error, see ${out#"$ART"/}/run.$tag.$host.out} (0 hits: step5n-compiled.md -- an in_out argument, or the NaN needs neighbouring kernels: C2)"
  fi
  LG end "step6[kernel]"
}

if [ -n "${GROUP:-}" ]; then
  pick=$(grep -vE 'FALSE_POSITIVE\?|PROPAGATED\?' "$ART/groups_priority.txt" | awk -v g="$GROUP" '$1 == g')
  [ -n "$pick" ] || finish "NO_SUCH_GROUP $GROUP (absent, FALSE_POSITIVE? or PROPAGATED?) -- $ART/groups_priority.txt" 1
else
  pick=$(grep -vE 'FALSE_POSITIVE\?|PROPAGATED\?' "$ART/groups_priority.txt" | sed -n "${REPLAY_GROUP:-1}p")
fi
[ -n "$pick" ] || { [ -z "${GROUP:-}" ] && kernel_capsule_repro; finish "NO_REPLAYABLE_GROUP${KRES} -- every group is an allocator/view false positive or PROPAGATED at capture (inputs already non-finite: the bug is upstream, or born in compiled code) -> step 5N, step5n-native.md -- $ART/groups_priority.txt; the top capsule: $(ls "$ART"/capsule_info.G*.txt 2>/dev/null | head -1)"; }
read -r G gcount rep gop <<<"$pick"
rep_host=$(rt_hostpath "$rep")
say "replaying $G (count=$gcount, $gop): $rep_host"

# ---- 3c. a COLLECTIVE group: the member ranks' capsules replayed together ------------------
# A synchronous collective (vllm.all_reduce) that returned non-finite from finite inputs: every
# member rank captured the same call.  The package holds all their capsules (and the manifest)
# and replays the collective on the recorded inputs under torchrun (collective_r3.py: check, no
# GPU; replay, through the framework that ran it) -- step4-distributed.md *Synchronous collectives*.
collective_repro() {
  local tdir ranks iters=${CAPSULE_ITER_COUNT:-1000}
  [ "$iters" -gt 100000 ] && iters=1000       # the capsule ladder's 1M runs: a collective replay is per call
  tdir=$(dirname -- "$rep_host")
  # the member ranks: the trial's manifest (collective_origin), else its capsules' ranks
  ranks=$(python3 - "$tdir" <<'EOF'
import glob, json, os, re, sys
d = sys.argv[1]
for m in sorted(glob.glob(os.path.join(d, 'manifest*.json'))):
    co = (json.load(open(m)).get('verdict') or {}).get('collective_origin')
    if co:
        print(len(co['ranks'])); sys.exit()
print(len({re.match(r'rank(\d+)_', os.path.basename(f)).group(1) for f in glob.glob(os.path.join(d, 'rank*_*.pt'))}) or 2)
EOF
)
  LG start "step5b[$G]"; LG job "step5b[$G]" "$tag" capture-to-reproducer job
  B="$ART/repro-$G"; V="$ART/verify-$G"
  for x in "$B" "$V" "$B.tar.gz"; do
    [ -e "$x" ] || continue
    mkdir -p "$ART/old" && mv -- "$x" "$ART/old/$(basename -- "$x").before-$tag" ||
      finish "REFUSING: $x exists and cannot be moved to $ART/old/" 1
  done
  mkdir -p "$B/capsule"
  cp -- "$tdir"/*.pt "$tdir"/manifest*.json "$B/capsule/" 2>/dev/null
  ship_runtime "$B" || finish "PACKAGE_FAILED: runtime layer" 1
  for f in collective_r3.py collective_arms.sh failure_signature.py op_nan_capture.py math_refs.py capsule_info.py; do
    cp -- "$(nd_tool "$f")" "$B/scripts/" || finish "PACKAGE_FAILED: $f" 1
  done
  bake_runtime
  cat >"$B/run.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=nan-repro-collective
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=00:30:00
# Reproduce the NaN from capsule/ without the model: the collective $gop that returned non-finite from
# finite inputs, replayed on the $ranks member ranks' recorded inputs ($ranks GPUs, one torchrun), through the
# framework that ran it, R3_ITERS times; checked against the fp64 reduction of the inputs.
#
#   bash run.sh          # on a GPU node
#   sbatch [<site args>] run.sh    # Slurm, from this directory
#   RUNTIME=...  IMAGE=...  R3_ITERS=...  R3_GPUS=0,1  (any $ranks GPUs of one node)
# this site: ${site_cmd:-bash run.sh}
#
# Result: the CHECK lines (no GPU needed) and the LADDER_RESULT= line.  See README.md.
set -uo pipefail
if [ -n "\${SLURM_JOB_ID:-}" ]; then HERE=\$SLURM_SUBMIT_DIR; else HERE=\$(dirname -- "\$0"); fi
HERE=\$(cd -- "\$HERE" && pwd -P) || exit 2
$(runtime_lines)
R3_GPUS=\${R3_GPUS:-$(seq -s, 0 $((ranks - 1)))}
exec "\$HERE/scripts/nd" run --gpus "\$R3_GPUS" -- "cd \\\$ND_REPO && python3 scripts/collective_r3.py check capsule; torchrun --nproc_per_node $ranks --master_addr 127.0.0.1 --master_port \${R3_PORT:-29741} scripts/collective_r3.py replay capsule --iters \${R3_ITERS:-$iters}"
EOF
  chmod +x "$B/run.sh"
  mkdir -p "$V" && tar -cf - -C "$ART" "repro-$G" | tar -xf - -C "$V" || finish "PACKAGE_FAILED: tar/extract" 1
  X="$V/repro-$G"; OUT="$ART/replay/$G"; mkdir -p "$OUT"; LADDER_OUT="$OUT/collective.$tag.$host.out"
  env -i HOME="$HOME" USER="${USER:-$(id -un)}" LOGNAME="${LOGNAME:-$(id -un)}" PATH="$PATH" LANG="${LANG:-C.UTF-8}" \
      bash -c 'cd -- "$1" && exec bash ./run.sh' _ "$X" >"$LADDER_OUT" 2>&1 &
  lpid=$!
  trap 'say "TERM -- stopping the replay"; kill -TERM $lpid 2>/dev/null' TERM INT
  nd_wait_pid "$lpid"
  res=$(sed -n 's/^LADDER_RESULT=//p' "$LADDER_OUT" | tail -1)
  chk=$(grep -m1 '^VERDICT' "$LADDER_OUT"); hint=$(grep -m1 '^NARROW_INTERMEDIATE?' "$LADDER_OUT")
  say "collective replay: ${res:-no LADDER_RESULT}; $chk ${hint:+; $hint} ($LADDER_OUT)"
  LG note "step5b[$G]" "COLLECTIVE $gop ranks=$ranks: ${chk:-no CHECK}${hint:+; ${hint%% --*}}; LADDER_RESULT=${res:-none} host=$host out=${LADDER_OUT#"$ART"/}"
  LG end "step5b[$G]"
  if [[ "$res" == REPRODUCED* ]]; then
    LG start "step6[$G]"; LG job "step6[$G]" "$tag" capture-to-reproducer job
    { echo "# NaN reproducer: collective $gop returns non-finite from finite inputs"
      echo
      echo "* Image: \`$img\`"
      echo "* Op: \`$gop\` (a synchronous collective), $ranks ranks; captured on every member rank (capsule/)"
      echo "* Check (fp64 reduction of the recorded inputs, no GPU): \`${chk#VERDICT }\`"
      [ -n "$hint" ] && echo "* Hint: \`$hint\`"
      echo "* Replay: \`$res\` ($ranks GPUs, one torchrun, host $host)"
      echo
      echo "Run: \`${site_cmd:-bash run.sh}\` from this directory (any $ranks GPUs of one node: \`R3_GPUS=\`)."
      echo "The replay runs the recorded op through the framework that ran it (vLLM's TP group and its"
      echo "\`VLLM_*\` env from the capsules), so the same all-reduce implementation is chosen for the same size."
      echo "<!-- CHECK: confirm the verdict and say which implementation the framework picked (README of step6-finish.md) -->"
    } >"$B/README.md"
    cp -- "$B/README.md" "$X/README.md"
    mk=$(python3 "$(nd_tool make_repro_readme.py)" --repro "$B" --repack --ledger "$LEDGER" --step "step6[$G]" 2>&1)
    echo "$mk"
    LG end "step6[$G]"
    write_next_groups
    [ -s "$B.tar.gz" ] || finish "REPRODUCED $G collective verified-from-extracted-copy  PACKAGE_INCOMPLETE: no tarball ($(tail -1 <<<"$mk")) -- $B" 1
    finish "REPRODUCED $G collective ranks=$ranks verified-from-extracted-copy ${chk#VERDICT }${hint:+ ${hint%% --*}}  tarball=$B.tar.gz (README auto: confirm its CHECK line)  next=$(next_count) group(s) in $ART/next_groups.sh"
  fi
  write_next_groups
  finish "${res:-LADDER_ERROR} $G collective -- see $LADDER_OUT; repro-$G did not reproduce: not a reproducer (keep it as evidence; step 5N builds the reproducer)  next=$(next_count) group(s) in $ART/next_groups.sh"
}
case "$gop" in vllm.all_reduce*|vllm.all_gather*|vllm.reduce_scatter*) collective_repro ;; esac

# ---- 3. package (step-6 layout), tar, extract elsewhere ------------------------------------
LG start "step5b[$G]"; LG job "step5b[$G]" "$tag" capture-to-reproducer job
B="$ART/repro-$G"; V="$ART/verify-$G"
# A re-run of the group: keep the earlier package, its tarball and its verification under old/
# (moved, not deleted: a root container's files may not be deletable by the user, and a delivered
# tarball must survive a re-run).
for x in "$B" "$V" "$B.tar.gz"; do
  [ -e "$x" ] || continue
  mkdir -p "$ART/old" && mv -- "$x" "$ART/old/$(basename -- "$x").before-$tag" ||
    finish "REFUSING: $x exists and cannot be moved to $ART/old/" 1
  LG note "step5b[$G]" "earlier $(basename -- "$x") moved to old/$(basename -- "$x").before-$tag"
done
mkdir -p "$B/capsule"
op_pkg=""
if [ "${CAPSULE_MODE:-chain}" = op ] && [ "${SLIM:-1}" = 1 ]; then
  # CAPSULE_MODE=op replays the op alone, so the chain only weighs the package down: package the
  # op-only copy (capsule_slim.py), which is what the ladder below then verifies. The full capsule
  # stays in the capsule dir. No chain in the capsule: capsule_slim refuses, and it is copied.
  so=$("$ND" run --gpus 0 -- "python3 -m capsule_slim $(rt_path "$rep_host") $(rt_path "$B/capsule/$(basename -- "$rep_host")")" 2>&1)
  grep -q '^SLIM=' <<<"$so" && op_pkg=" op-only($(sed -n 's/^SLIM=[^ ]* //p' <<<"$so"))" &&
    LG note "step5b[$G]" "CAPSULE_MODE=op: packaged the op-only copy of $(basename -- "$rep_host")${op_pkg}"
fi
rm -f -- "$B/capsule/"*.tmp
[ -n "$op_pkg" ] || cp -- "$rep_host" "$B/capsule/" || finish "PACKAGE_FAILED: capsule $rep_host" 1
ship_runtime "$B" || finish "PACKAGE_FAILED: runtime layer / build context ${IMAGE_BUILD_DIR:-}" 1
cp -- "$ND_REPO_HOST/scripts/jobs/replay_ladder.sh" "$B/scripts/jobs/"
for f in replay_nan_op.py failure_signature.py kernel_patch.py kernel_mutations.py op_nan_capture.py replay_hipblaslt_log_with_nan.py \
         transformer_hog.py pressure_hog.py capsule_replay_summary.sh replay_verdict.py cotenant_loop.sh kernel_match.py math_refs.py \
         inductor_ref.py twins.py capsule_operands.py waitcnt_scan.py; do
  cp -- "$(nd_tool "$f")" "$B/scripts/" || finish "PACKAGE_FAILED: $f" 1
done
if [ -n "${COTENANT_FILES:-}" ]; then
  mkdir -p "$B/cotenant"
  for f in $COTENANT_FILES; do cp -- "$f" "$B/cotenant/" || finish "PACKAGE_FAILED: $f" 1; done
fi
capname=$(basename -- "$rep_host")
if [ -n "${CAPSULE_COTENANT:-}" ]; then
  what="replay the capsule's op (CAPSULE_MODE=${CAPSULE_MODE:-op}) on every GPU of this node beside the
# co-tenant load it needs (CAPSULE_COTENANT, below): one copy per GPU + the load; without the
# load it does not reproduce."
else
  what="replay the capsule's recorded ${CAPSULE_MODE:-chain} on every GPU of this node,
# CAPSULE_CONCURRENCY copies per GPU, CAPSULE_ITER_COUNT runs per copy.  The copies on one GPU
# are each other's co-tenants -- ONE copy alone gives 0/N and means nothing."
fi
bake_runtime
cat >"$B/run.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=nan-repro
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=02:00:00
# Reproduce the NaN from capsule/ without the model: $what
#
#   bash run.sh          # on a GPU node
#   sbatch [<site args: account, partition, GPU request>] run.sh    # Slurm, from this directory
#   RUNTIME=apptainer|docker|podman|native  IMAGE=...  SIF=...  APPTAINER_FLAGS=...  VENV=...  NGPUS=...
#   ND_ENV=<a profile, scripts/env/profiles/*.env style>  bash run.sh
# this site: ${site_cmd:-bash run.sh}
#
# Result: the LADDER_RESULT= line, and out/c<C>.*/summary.txt.  See README.md.
set -uo pipefail
if [ -n "\${SLURM_JOB_ID:-}" ]; then HERE=\$SLURM_SUBMIT_DIR; else HERE=\$(dirname -- "\$0"); fi
HERE=\$(cd -- "\$HERE" && pwd -P) || exit 2
[ -f "\$HERE/scripts/jobs/replay_ladder.sh" ] || { echo "run.sh must run from (or be submitted from) its own directory"; exit 2; }
$(runtime_lines)
export CAPSULE_CONCURRENCY=\${CAPSULE_CONCURRENCY-@REPRO_CONCURRENCY@}   # what reproduced
export CAPSULE_MODE=\${CAPSULE_MODE:-${CAPSULE_MODE:-chain}}
export CAPSULE_COTENANT=\${CAPSULE_COTENANT-'${CAPSULE_COTENANT:-}'}   # the load it needs ('' = none)
export CAPSULE_EXTRA_ARGS=\${CAPSULE_EXTRA_ARGS-'${CAPSULE_EXTRA_ARGS:-}'}   # e.g. --synthesize (shapes-only capsule)
export CAPSULE_ITER_COUNT=\${CAPSULE_ITER_COUNT:-${CAPSULE_ITER_COUNT:-1000000}}
export CAPSULE_FUTILITY=\${CAPSULE_FUTILITY:-${CAPSULE_FUTILITY:-100}}   # as in the run that reproduced (0 = a rare race)
export CAPSULE_STOP_HITS=\${CAPSULE_STOP_HITS:-${CAPSULE_STOP_HITS:-10}}
export CAPSULE_SECONDS=\${CAPSULE_SECONDS-'${CAPSULE_SECONDS:-}'}   # '' = off
export CAPSULE_KERNEL_ID=\${CAPSULE_KERNEL_ID-'${CAPSULE_KERNEL_ID:-}'}   # '' = 1 (identify the kernel)
export CAPSULE=\$HERE/capsule/$capname
export OUT_DIR=\${OUT_DIR:-\$HERE/out}
exec bash "\$HERE/scripts/jobs/replay_ladder.sh"
EOF
chmod +x "$B/run.sh"
mkdir -p "$V" && tar -cf - -C "$ART" "repro-$G" | tar -xf - -C "$V" || finish "PACKAGE_FAILED: tar/extract" 1
LG note "step5b[$G]" "packaged repro-$G (capsule $capname), extracted to verify-$G; replaying from the extracted copy (= step-6 verification)"

# ---- 4. 5b ladder from the extracted copy ---------------------------------------------------
X="$V/repro-$G"; OUT="$ART/replay/$G"; mkdir -p "$OUT"
LADDER_OUT="$OUT/ladder.$tag.$host.out"
# The extracted package's OWN run.sh, in an empty environment (no profile, no SLURM_*): exactly a
# vendor's `bash run.sh` on this node -- its runtime/SIF defaults are what gets verified, not this
# investigation's ND_ENV.  Only the start concurrency (empty = the ladder's own start), the
# output dir and the caller's identity/PATH are passed in.
# exec all the way down, so $lpid is the ladder itself and the TERM trap below reaches it.
# Its output stays in the package's own out/ (a vendor's default): without a profile the runtime
# mounts only the package dir, so an OUT_DIR outside it is unwritable in the container (progress
# files, hence the hit-count early stop, were lost).
env -i HOME="$HOME" USER="${USER:-$(id -un)}" LOGNAME="${LOGNAME:-$(id -un)}" PATH="$PATH" \
    LANG="${LANG:-C.UTF-8}" CAPSULE_CONCURRENCY="${CAPSULE_CONCURRENCY:-}" \
    bash -c 'cd -- "$1" && exec bash ./run.sh' _ "$X" >"$LADDER_OUT" 2>&1 &
lpid=$!
trap 'say "TERM -- stopping the ladder"; kill -TERM $lpid 2>/dev/null' TERM INT
nd_wait_pid "$lpid"
res=$(sed -n 's/^LADDER_RESULT=//p' "$LADDER_OUT" | tail -1)
say "ladder: ${res:-no LADDER_RESULT} ($LADDER_OUT)"
summ=$(ls -1t "$X"/out/c*/summary.txt 2>/dev/null | head -1)   # the package's out/, the ladder's own tag
[ -n "$summ" ] || summ="$OUT/none/summary.txt"
kern=$(sed -n 's/^KERNEL=//p' "$LADDER_OUT" | tail -1); kern=${kern:-?}
# an Inductor / Triton op IS its kernel (inductor.<kernel>.s<hash>): no library lookup names it, and the
# result line said kernel=? (2026-11-24)
if [[ ${kern%% *} == "?" ]]; then
  gop=$(awk -v g="$G" '$1 == g {print $4; exit}' "$ART/groups_priority.txt" 2>/dev/null)
  [[ $gop == inductor.* || $gop == triton.* ]] && kern=$(cut -d. -f2 <<<"$gop")
fi
[ "$kern" != "?" ] && LG note "step5b[$G]" "KERNEL=$kern"
LG note "step5b[$G]" "LADDER_RESULT=${res:-none}  host=$host  attempts: $(grep -m1 'attempts:' "$LADDER_OUT" | sed 's/.*attempts: //')  summary=${summ#"$ART"/}"
LG end "step5b[$G]"
# ---- 4b. the op-only package: the chain dropped, verified the same way ------------------------
# A real-values chain capsule is mostly the chain's values (2026-11-21, aiter MLA decode: 782 MB, the
# op's own operands 294 MB of it). capsule_slim.py keeps the op's operands; the copy is packaged with
# CAPSULE_MODE=op and replayed from its own extracted copy. If it reproduces it ships as repro-$G,
# and the verified chain package stays beside it as repro-$G-chain (step 7 and a vendor who asks).
# SLIM=0: off. A capsule without a chain fails capsule_slim and changes nothing.
slim_args=(); slim_note=""
slim_try() {
  [ "${SLIM:-1}" = 1 ] && [ "${CAPSULE_MODE:-chain}" = chain ] || return 0
  local S="$ART/repro-$G-op" VS="$ART/verify-$G-op" so lo lp sres ssumm x
  local pv='s/^=== proposed verdict: \([A-Z_0-9?]*\).*/\1/p' cv ov
  cv=$(sed -n "$pv" "$summ" 2>/dev/null | tail -1)
  # Only a kernel-bug verdict can come out of the op alone: a single-op replay whose fp64 reference
  # is non-finite stops before it runs the op (TRUE_RESULT_NONFINITE, an ERROR to the ladder).
  if [[ "$cv" != KERNEL_BUG* ]]; then
    LG note "step5b[$G]" "op-only package not tried: the chain proposes ${cv:-?}"
    return 0
  fi
  for x in "$S" "$VS" "$B-chain" "$V-chain"; do
    [ -e "$x" ] || continue
    mkdir -p "$ART/old" && mv -- "$x" "$ART/old/$(basename -- "$x").before-$tag" || return 0
  done
  mkdir -p "$S/capsule" && tar -cf - -C "$B" --exclude ./out --exclude ./capsule . | tar -xf - -C "$S" || return 0
  so=$("$ND" run --gpus 0 -- "python3 -m capsule_slim $(rt_path "$B/capsule/$capname") $(rt_path "$S/capsule/$capname")" 2>&1)
  if ! grep -q '^SLIM=' <<<"$so"; then
    LG note "step5b[$G]" "op-only package not built: $(grep -v '^\[nd ' <<<"$so" | tail -1)"
    mkdir -p "$ART/old" && mv -- "$S" "$ART/old/$(basename -- "$S").before-$tag"
    return 0
  fi
  sed -i -e 's/^export CAPSULE_MODE=\${CAPSULE_MODE:-chain}/export CAPSULE_MODE=${CAPSULE_MODE:-op}/' \
         -e "s/replay the capsule's recorded chain on every GPU of this node,/replay the capsule's op (alone: CAPSULE_MODE=op) on every GPU of this node,/" "$S/run.sh"
  mkdir -p "$VS" && tar -cf - -C "$ART" "repro-$G-op" | tar -xf - -C "$VS" || return 0
  say "op-only package: $(sed -n 's/^SLIM=[^ ]* //p' <<<"$so"); replaying it from $VS (CAPSULE_MODE=op, start c$C)"
  lo="$OUT/ladder-op.$tag.$host.out"
  env -i HOME="$HOME" USER="${USER:-$(id -un)}" LOGNAME="${LOGNAME:-$(id -un)}" PATH="$PATH" \
      LANG="${LANG:-C.UTF-8}" CAPSULE_CONCURRENCY="$C" \
      bash -c 'cd -- "$1" && exec bash ./run.sh' _ "$VS/repro-$G-op" >"$lo" 2>&1 &
  lp=$!
  trap 'say "TERM -- stopping the op-only ladder"; kill -TERM $lp 2>/dev/null' TERM INT
  nd_wait_pid "$lp"
  sres=$(sed -n 's/^LADDER_RESULT=//p' "$lo" | tail -1)
  ssumm=$(ls -1t "$VS/repro-$G-op"/out/c*/summary.txt 2>/dev/null | head -1)
  LG note "step5b[$G]" "op-only package ($(sed -n 's/^SLIM=[^ ]* //p' <<<"$so")): LADDER_RESULT=${sres:-none}  attempts: $(grep -m1 'attempts:' "$lo" | sed 's/.*attempts: //')  ${ssumm:+summary=${ssumm#"$ART"/}}"
  # Ship it only when the op alone proposes what the chain proposed: the chain's walk is evidence
  # the op-only replay does not have (an op-only replay read a 0/0 RMSNorm row as KERNEL_BUG where
  # the walk said FP32_OVERFLOW?, 2026-11-21). A different verdict keeps the chain package.
  ov=$(sed -n "$pv" "$ssumm" 2>/dev/null | tail -1)
  if [[ "$sres" =~ ^REPRODUCED ]] && [ "$cv" != "$ov" ]; then
    LG note "step5b[$G]" "op-only package NOT shipped: it proposes ${ov:-?}, the chain ${cv:-?} -- the chain package ships"
    say "op-only: proposes ${ov:-?}, the chain ${cv:-?} -- the chain package ships"
    sres="VERDICT_DIFFERS"
  fi
  if [[ "$sres" =~ ^REPRODUCED\ at\ concurrency\ ([0-9]+) ]] && [ -n "$ssumm" ]; then
    sed -i "s/@REPRO_CONCURRENCY@/$C/" "$B/run.sh" "$X/run.sh"          # the chain package, as verified
    mv -- "$B" "$B-chain" && mv -- "$V" "$V-chain" && mv -- "$S" "$B" || { say "op-only: rename failed"; return 0; }
    # the chain attempt's files name their own dir: follow the rename (a summary.txt pointed at
    # verify-G1/..., which then held the op package, 2026-11-23)
    grep -rlF -- "$V/" "$V-chain" --include='*.txt' 2>/dev/null | xargs -r sed -i "s#$V/#$V-chain/#g"
    C=${BASH_REMATCH[1]}; X="$VS/repro-$G-op"; summ=$ssumm
    slim_args=(--mode op --chain-package "repro-$G-chain")
    slim_note=" op-only(chain package kept: repro-$G-chain)"
  else
    say "op-only: ${sres:-no LADDER_RESULT} -- the chain package ships ($lo)"
    mkdir -p "$ART/old" && mv -- "$S" "$ART/old/$(basename -- "$S").before-$tag"
  fi
}
if [[ "$res" =~ ^REPRODUCED\ at\ concurrency\ ([0-9]+) ]]; then
  C=${BASH_REMATCH[1]}
  slim_try
  sed -i "s/@REPRO_CONCURRENCY@/$C/" "$B/run.sh" "$X/run.sh"
  LG start "step6[$G]"; LG job "step6[$G]" "$tag" capture-to-reproducer job
  mk=$(python3 "$(nd_tool make_repro_readme.py)" --repro "$B" --attempt "$(dirname -- "$summ")" \
         --groups "$ART/groups.json" --image "$img" --pack --ledger "$LEDGER" --step "step6[$G]" "${slim_args[@]}" 2>&1)
  echo "$mk"
  LG end "step6[$G]"
  prop=$(sed -n 's/.*proposed=\([A-Z_0-9?]*\).*/\1/p' <<<"$mk" | head -1)
  dns=$(grep -q '^DO_NOT_SHIP' <<<"$mk" && echo ' DO_NOT_SHIP(see README CHECK line)')
  grep -q '^MODEL_NUMERICS' <<<"$mk" && dns=' MODEL_NUMERICS(not a kernel bug: ship it as a model-numerics reproducer, step6-finish.md)'
  write_next_groups
  [ -s "$B.tar.gz" ] || finish "REPRODUCED $G concurrency=$C verified-from-extracted-copy  PACKAGE_INCOMPLETE: no README/tarball (make_repro_readme: $(tail -1 <<<"$mk")) -- $B" 1
  finish "REPRODUCED $G concurrency=$C verified-from-extracted-copy${slim_note}${op_pkg} proposed=${prop:-?}${dns}  kernel=${kern%% *}  tarball=$B.tar.gz (README auto: confirm its CHECK line, step6-finish.md)  next=$(next_count) group(s) in $ART/next_groups.sh"
fi
write_next_groups
finish "${res:-LADDER_ERROR} $G -- see $LADDER_OUT; repro-$G did not reproduce: not a reproducer (keep it as evidence; step 5N builds the reproducer)  next=$(next_count) group(s) in $ART/next_groups.sh"
