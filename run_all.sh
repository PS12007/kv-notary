#!/usr/bin/env bash
# Every experiment, one GPU job at a time (8 GB card). Results land in results/*.json.
set -e
PY=${PY:-python}
export PYTHONWARNINGS=ignore
cd "$(dirname "$0")"
mkdir -p logs
run() { echo "== $*"; $PY "$@" > "logs/$(basename "$1" .py)_$3_$(date +%s).log" 2>&1 || echo "FAILED $*"; }
# --from N skips the first N jobs of the rest-queue (to resume after an interruption)
SKIP=0; [ "$2" == "--from" ] && SKIP=$3
i=0; q() { i=$((i+1)); [ $i -le $SKIP ] && { echo "skip $*"; return; }; run "$@"; }
[ "$1" == "--rest" ] || {
  run experiments/forge.py --model llama-1b --n 24
  run experiments/forge.py --model qwen-0.5b --n 24
}
q experiments/adaptive.py --model llama-1b --n 16
q experiments/corrupt.py --model llama-1b --n 12
q experiments/innoise.py --model llama-1b --dtype bf16 --n 12
q experiments/innoise.py --model llama-1b --dtype fp32 --n 8
q experiments/cost.py --model llama-1b
q experiments/forge.py --model llama-1b --n 12 --long 8000
q experiments/forge.py --model llama-1b --n 8 --long 16000
q experiments/adaptive.py --model qwen-0.5b --n 16
q experiments/corrupt.py --model qwen-0.5b --n 12
q experiments/innoise.py --model qwen-0.5b --dtype bf16 --n 12
q experiments/defense.py --model llama-1b --n 12
q experiments/defense.py --model qwen-0.5b --n 12
q experiments/defense.py --model llama-1b --n 12 --start 700 --tag b
q experiments/defense.py --model qwen-0.5b --n 12 --start 700 --tag b
q experiments/crossstack.py --model llama-1b --n 24
q experiments/crossstack.py --model qwen-0.5b --n 24
q experiments/export_demo.py --model llama-1b
q experiments/export_demo.py --model qwen-0.5b
$PY experiments/figures.py
$PY experiments/paper.py
echo all done
