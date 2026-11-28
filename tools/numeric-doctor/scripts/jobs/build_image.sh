#!/bin/bash
#SBATCH --job-name=nd-build
#SBATCH --nodes=1
#SBATCH --time=01:00:00
# Build the model's image from a run script's Dockerfile on a node, once per investigation
# (env-runtime.md *Images*; capture_preflight.py's `IMAGE ... BUILDS its image`), then save it as an
# archive the other nodes load (IMAGE_TAR) -- docker / podman.  The node that built it is recorded
# in <ART>/jobs/image_nodes, so the next `nd submit` pins it while it is idle (no load at all).
#
#   ND_ENV=<ARTIFACT_DIR>/env.sh BUILD_DIR=<dir with the Dockerfile> \
#     scripts/nd submit --name build --gpus 1 --time 01:00:00 -- build_image.sh
#
# Knobs: BUILD_DIR (required, host path), IMAGE (the tag to build: the profile's), BUILD_ARGS (extra
# `docker build` flags, e.g. "--build-arg BASE=..."), SAVE (1 default; 0 = build only).  The archive
# is <artifacts>/image-cache/<context key>/<IMAGE with / : -> _>.tar.zst (shared across investigations; a
# cache hit skips build and save) (zstd -T0: 3m43s for the SGLang image's 14 GB archive,
# where gzip -1 took ~6 min, 2026-11-19; .tar.gz with pigz / gzip where zstd is missing).
# Without IMAGE_TAR in the profile the job appends it to ND_ENV.  The build context -- the
# Dockerfile, .dockerignore and what its COPY / ADD lines name -- is kept in <ART>/image/build and
# named in ND_ENV as IMAGE_BUILD_DIR: the image is in no registry, so a step-6/8 package ships that
# dir and its run.sh builds the image where neither IMAGE_TAR nor a pull exists (off-site;
# runtime_docker.sh).  The build's image ID goes in ND_ENV as IMAGE_ID: a node whose tag holds another
# build (an earlier investigation's) reloads IMAGE_TAR, and nd submit pins only nodes recorded with
# this ID (lib.sh nd_image_stale, nd_pick_image_node).  Last line: BUILD_RESULT=OK|FAILED.
set -uo pipefail
# shellcheck disable=SC1091
source "${ND_REPO_HOST:?run through nd submit, or export ND_REPO_HOST}/scripts/env/lib.sh"
nd_job_init
fail() { echo "BUILD_RESULT=FAILED $*"; exit 1; }
[ -n "${BUILD_DIR:-}" ] && [ -f "$BUILD_DIR/Dockerfile" ] || fail "BUILD_DIR=${BUILD_DIR:-} holds no Dockerfile"
[ -n "${IMAGE:-}" ] || fail "no IMAGE in the profile: the tag to build"
case $RUNTIME in
  docker) eng=(docker) ;;
  podman) read -ra eng <<<"podman ${PODMAN_ARGS:-}" ;;
  *) fail "RUNTIME=$RUNTIME: build with docker or podman (apptainer: env-runtime.md *Images*)" ;;
esac
echo "[$(date)] host=$(hostname -s) job=${SLURM_JOB_ID:-local} runtime=$RUNTIME image=$IMAGE dir=$BUILD_DIR"
# Content key of the build: the Dockerfile, what its COPY / ADD lines name, BUILD_ARGS and the tag.
# The same key in an earlier investigation is the same image: its build is reused -- no build, no
# save, and the nodes holding that build (sibling image_nodes, keyed by IMAGE_ID) are pinned.  A
# rebuild of an unchanged Dockerfile cost 5-11 min (save 9-18 GB) plus a 1-2 min load on every node
# holding the previous build (2026-11-23 round).  ND_IMAGE_CACHE=0: build as before.
key=$(python3 - "$BUILD_DIR" "${BUILD_ARGS:-}" "$IMAGE" <<'PY'
import glob, hashlib, os, re, sys
src, bargs, tag = sys.argv[1:]
h = hashlib.sha256(("%s\0%s\0" % (tag, bargs)).encode())
df = open(os.path.join(src, 'Dockerfile'), 'rb').read()
h.update(df)
lines = re.sub(r'\\\n', ' ', df.decode(errors='replace')).splitlines()
want = ['.dockerignore']
for l in lines:
    w = l.split()
    if len(w) >= 3 and w[0].upper() in ('COPY', 'ADD') and not any(x.startswith('--from') for x in w):
        want += [x for x in w[1:-1] if not x.startswith('--') and '://' not in x]
for pat in sorted(set(want)):
    for p in sorted(glob.glob(os.path.join(src, pat))):
        files = [p] if os.path.isfile(p) else sorted(os.path.join(r, f) for r, _, fs in os.walk(p) for f in fs)
        for f in files:
            h.update(os.path.relpath(f, src).encode() + b'\0')
            with open(f, 'rb') as fh:
                for b in iter(lambda: fh.read(1 << 20), b''):
                    h.update(b)
print(h.hexdigest()[:16])
PY
) || fail "hashing the build context"
cache=${ND_IMAGE_CACHE_DIR:-$(dirname "$ND_ART_HOST")/image-cache}/$key
echo "build context key $key ($cache)"
if [ "${ND_IMAGE_CACHE:-1}" = 1 ] && [ -r "$cache/IMAGE_ID" ] && { [ "${SAVE:-1}" = 0 ] || [ -r "$(cat "$cache/IMAGE_TAR" 2>/dev/null)" ]; }; then
  IMAGE_ID=$(cat "$cache/IMAGE_ID"); tar=$(cat "$cache/IMAGE_TAR" 2>/dev/null)
  echo "[$(date)] cache hit: build $IMAGE_ID of $IMAGE from $(cat "$cache/BUILT_BY" 2>/dev/null) -- not rebuilt, not saved"
  here=$("${eng[@]}" image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null)
  [ "${here#sha256:}" = "$IMAGE_ID" ] && nd_note_image_node && echo "this node holds that build (recorded in jobs/image_nodes)"
  if [ -n "${ND_ENV:-}" ]; then
    sed -i '/^IMAGE_ID=/d' "$ND_ENV" && echo "IMAGE_ID=$IMAGE_ID   # build_image.sh cache $key" >>"$ND_ENV"
    if [ -n "$tar" ]; then
      sed -i '/^IMAGE_TAR=/d' "$ND_ENV" && echo "IMAGE_TAR=\"$tar\"   # build_image.sh cache $key" >>"$ND_ENV"
    fi
  fi
  [ -n "$tar" ] && echo "IMAGE_TAR=$tar"
  cached=1
else
  cached=0
fi
# An empty DOCKER_CONFIG: a stored, expired registry login fails even a public base pull; --network=host:
# some daemons have no default bridge (env-runtime.md).
cfg=$(mktemp -d)
if [ "$cached" = 0 ]; then
# shellcheck disable=SC2086
DOCKER_CONFIG=$cfg "${eng[@]}" build --network=host -t "$IMAGE" ${BUILD_ARGS:-} "$BUILD_DIR" ||
  { rm -rf "$cfg"; fail "build"; }
rm -rf "$cfg"
id=$("${eng[@]}" image inspect "$IMAGE" --format '{{.Id}}') || fail "inspect"
IMAGE_ID=${id#sha256:}     # the build, not the tag: lib.sh nd_image_stale / nd_pick_image_node
nd_note_image_node
echo "[$(date)] built $IMAGE $id on $(hostname -s) (recorded in jobs/image_nodes)"
if [ "${SAVE:-1}" = 1 ]; then
  mkdir -p "$ND_ART_HOST/image" || fail "mkdir image/"
  # the archive lives in the shared cache: one copy per build, not per investigation
  mkdir -p "$cache" || fail "mkdir $cache"
  base="$cache/$(tr '/:' '__' <<<"$IMAGE")"
  if command -v zstd >/dev/null; then out=$base.tar.zst; z=(zstd -T0 -3 -q)
  elif command -v pigz >/dev/null; then out=$base.tar.gz; z=(pigz -1)
  else out=$base.tar.gz; z=(gzip -1); fi
  "${eng[@]}" save "$IMAGE" | "${z[@]}" >"$out.tmp" && mv -- "$out.tmp" "$out" || fail "save"
  sha=$(sha256sum "$out" | cut -d' ' -f1)
  echo "[$(date)] saved $out ($(du -h "$out" | cut -f1), sha256 $sha)"
  if [ -n "${ND_ENV:-}" ] && ! grep -q '^IMAGE_TAR=' "$ND_ENV"; then
    echo "IMAGE_TAR=\"$out\"   # build_image.sh, job ${SLURM_JOB_ID:-local}, sha256 $sha" >>"$ND_ENV"
    echo "IMAGE_TAR appended to $ND_ENV"
  fi
  echo "IMAGE_TAR=$out"
fi
if [ -n "${ND_ENV:-}" ]; then      # every build: a rebuild under the same tag is another image
  sed -i '/^IMAGE_ID=/d' "$ND_ENV" && echo "IMAGE_ID=$IMAGE_ID   # build_image.sh, job ${SLURM_JOB_ID:-local}" >>"$ND_ENV"
fi
# Record the build in the cache last, so a reader never sees an ID without its archive.
mkdir -p "$cache" && { echo "$IMAGE_ID" >"$cache/IMAGE_ID.tmp"; [ -n "${out:-}" ] && echo "$out" >"$cache/IMAGE_TAR";
  echo "$ND_ART_HOST job ${SLURM_JOB_ID:-local} $(hostname -s) $(date -u +%FT%TZ)" >"$cache/BUILT_BY";
  mv "$cache/IMAGE_ID.tmp" "$cache/IMAGE_ID"; } || echo "note: could not record the build in $cache"
fi   # cached = 0
# The build context, for the packages (above): file sources only, each under 100 MB
ctx=$ND_ART_HOST/image/build
mkdir -p "$ctx" && python3 - "$BUILD_DIR" "$ctx" <<'PY' || fail "saving the build context"
import glob, os, re, shutil, sys
src, dst = sys.argv[1:]
lines = re.sub(r'\\\n', ' ', open(os.path.join(src, 'Dockerfile')).read()).splitlines()
want = ['Dockerfile', '.dockerignore']
for l in lines:
    w = l.split()
    if len(w) < 3 or w[0].upper() not in ('COPY', 'ADD') or any(x.startswith('--from') for x in w):
        continue
    want += [x for x in w[1:-1] if not x.startswith('--') and '://' not in x]
for pat in want:
    for p in glob.glob(os.path.join(src, pat)):
        rel = os.path.relpath(p, src)
        size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs) \
            if os.path.isdir(p) else os.path.getsize(p)
        if size > 100e6:
            print('build context: %s skipped (%.0f MB)' % (rel, size / 1e6)); continue
        t = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(t), exist_ok=True)
        (shutil.copytree(p, t, dirs_exist_ok=True) if os.path.isdir(p) else shutil.copy2(p, t))
print('build context: %s' % ' '.join(sorted(os.listdir(dst))))
PY
if [ -n "${ND_ENV:-}" ] && ! grep -q '^IMAGE_BUILD_DIR=' "$ND_ENV"; then
  echo "IMAGE_BUILD_DIR=\"$ctx\"   # build_image.sh: the build context a package ships (image/)" >>"$ND_ENV"
fi
echo "IMAGE_BUILD_DIR=$ctx"
echo "BUILD_RESULT=OK image=$IMAGE id=${id:-sha256:$IMAGE_ID} node=$(hostname -s) cached=$cached key=$key"
