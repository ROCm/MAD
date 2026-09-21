#!/bin/bash
# prep_model.sh — stage MiniMax-M3-MXFP4 as a FLAT dir vLLM can load offline.
#
# The HF snapshot of amd/MiniMax-M3-MXFP4 ships weights + config + tokenizer, but the
# config.json auto_map references remote-code Python (configuration_minimax_m3_vl.py) that
# is NOT in the weights-only download. Without it, `--trust-remote-code` fails to import
# the config. This builds a flat dir: symlinks to the snapshot's weights + real copies of
# the remote-code .py files.
#
# Usage: SNAPSHOT=<hf snapshot dir> PYSRC=<dir with the .py files> DEST=<flat out dir> ./prep_model.sh
set -euo pipefail
SNAPSHOT="${SNAPSHOT:?path to models--amd--MiniMax-M3-MXFP4/snapshots/<hash>}"
PYSRC="${PYSRC:?dir containing configuration_minimax_m3_vl.py etc (e.g. an M3 repo with remote code)}"
DEST="${DEST:?flat output dir, e.g. /models/MiniMax-M3-MXFP4}"

mkdir -p "$DEST"
# symlink every real file from the snapshot (weights, config, tokenizer)
for f in "$SNAPSHOT"/*; do ln -sfn "$(readlink -f "$f")" "$DEST/$(basename "$f")"; done
# overlay real copies of the remote-code .py files
for p in configuration_minimax_m3_vl.py image_processor.py processing_minimax.py video_processor.py; do
  [ -f "$PYSRC/$p" ] && cp -f "$PYSRC/$p" "$DEST/$p" && echo "copied $p"
done
echo "flat model dir ready: $DEST"
ls -L "$DEST"/config.json "$DEST"/model.safetensors.index.json >/dev/null && echo "config + index resolve OK"
echo "shards: $(ls -L "$DEST"/model-*-of-*.safetensors 2>/dev/null | wc -l)"
