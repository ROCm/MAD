#!/usr/bin/env bash
# PostToolUse hook (Edit|Write|MultiEdit): when Claude edits a scripts/**/models.json
# registry, validate that file and feed any errors back (exit 2) so they are
# fixed in the same turn instead of surfacing at `madengine run` time.
# Missing Dockerfile/script references are only warnings here, because
# /mad-add-model may write the entry before those files; its final validation
# step (and CI) treats them as errors.
set -uo pipefail

root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
file=$(python3 -c 'import json, sys; print(json.load(sys.stdin).get("tool_input", {}).get("file_path", ""))' 2>/dev/null) || exit 0
case "$file" in
  "$root"/scripts/*/models.json | scripts/*/models.json) ;;
  *) exit 0 ;;
esac
[ -f "$file" ] || exit 0

cd "$root" || exit 0
if ! out=$(python3 tools/validation/model_config_validator.py --models-json "$file" --allow-missing-files 2>&1); then
  {
    echo "tools/validation/model_config_validator.py found errors in $file:"
    echo "$out" | tail -n 60
  } >&2
  exit 2
fi
exit 0
