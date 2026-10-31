#!/usr/bin/env bash
# Round two: defenses against adaptive forgers, and the ~4K-token long-document run.
PY=${PY:-python}
export PYTHONWARNINGS=ignore
cd "$(dirname "$0")"
run() { echo "== $*"; $PY "$@" > "logs/$(basename "$1" .py)_$3_$(date +%s).log" 2>&1 || echo "FAILED $*"; }
run experiments/defense.py --model llama-1b --n 12
run experiments/defense.py --model qwen-0.5b --n 12
run experiments/forge.py --model llama-1b --n 6 --long 16000
echo all done
