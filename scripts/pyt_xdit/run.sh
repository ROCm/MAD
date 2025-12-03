#!/bin/bash
###############################################################################
#
# MIT License
#
# Copyright (c) Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################
OPTIONS=$(getopt -o w:v --long workload:,verbose -- "$@")

if [ $? -ne 0 ]; then
  echo "Failed to parse options." >&2
  exit 1
fi

eval set -- "$OPTIONS"

# Parse options
while true; do
  case "$1" in
    -w|--workload)
      WORKLOAD="$2"
      shift 2
      ;;
    -v|--verbose)
      VERBOSE=true
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

SCRIPT="/app/.ci/run.${WORKLOAD}.sh"

if [ ! -e $SCRIPT ]; then
    echo "'$SCRIPT' not found" >&2
    exit 1
fi

# set HF_TOKEN
export HF_TOKEN=$MAD_SECRETS_HFTOKEN

# temporary fix to handle host ROCm version <6.4.2
export HSA_NO_SCRATCH_RECLAIM=1

# run workload
echo "Run instructions:"
bash $SCRIPT --help
echo "Run configurations:"
bash $SCRIPT --mad --dry-run
RECORDS=$(bash $SCRIPT --mad)

if [ $? -ne 0 ]; then
  echo "Failed to run workload" >&2
  exit 1
fi

if [ $VERBOSE ]; then
    echo "$RECORDS"
fi

# save results
echo -e "model,performance,metric" > ../results.csv
echo -e "$RECORDS"  >> ../results.csv
