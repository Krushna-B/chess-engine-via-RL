! usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CUTECHESS="$ROOT_DIR/build/cutechess/cutechess-cli"
ENGINE="$ROOT_DIR/build/engine"
RESULTS_DIR="$ROOT_DIR/results"

mkdir -p "$RESULTS_DIR"

if [[ ! -x "$CUTECHESS" ]]; then
    echo "Cute Chess not found at: $CUTECHESS" >&2
    exit 1
fi

if [[ ! -x "$ENGINE" ]]; then
    echo "Engine not found at: $ENGINE" >&2
    exit 1
fi

#Games are 10s with 0.1s increment
"$CUTECHESS" \
    -engine name=RandomWhite cmd="$ENGINE" proto=uci \
    -engine name=RandomBlack cmd="$ENGINE" proto=uci \
    -each tc=10+0.1 timemargin=100 \
    -rounds 200 \
    -repeat \
    -concurrency 1 \
    -ratinginterval 20 \
    -pgnout "$RESULTS_DIR/random-selfplay.pgn"