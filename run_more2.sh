#!/usr/bin/env bash
# Round three: consistent cost benchmark and 12 more defense tasks per model.
PY=${PY:-python}
export PYTHONWARNINGS=ignore
cd "$(dirname "$0")"
run() { echo "== $*"; $PY "$@" > "logs/$(basename "$1" .py)_$3_$(date +%s).log" 2>&1 || echo "FAILED $*"; }
run experiments/cost.py --model llama-1b
run experiments/defense.py --model llama-1b --n 12 --start 700 --tag b
run experiments/defense.py --model qwen-0.5b --n 12 --start 700 --tag b
echo all done
