#!/usr/bin/env bash
# Play the trained net (engine_neural + a JIT checkpoint) against the random
# mover and then up a Stockfish Elo ladder, all through cutechess-cli.
#
#   MODEL=artifacts/eval/gen_0005_jit.pt scripts/run_eval.sh
#
# Grab a checkpoint first with:  modal run modal_app.py::download --model ...
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CUTECHESS="${CUTECHESS:-$ROOT_DIR/build/cutechess/cutechess-cli}"
RANDOM_ENGINE="${RANDOM_ENGINE:-$ROOT_DIR/build/engine}"
NEURAL_ENGINE="${NEURAL_ENGINE:-$ROOT_DIR/build/engine_neural}"
STOCKFISH="${STOCKFISH:-$(command -v stockfish 2>/dev/null || true)}"
MODEL="${MODEL:-$ROOT_DIR/artifacts/checkpoints/chess_model_jit.pt}"

# 0 sims = raw policy (the net's gut); >0 runs MCTS, capped here but also bound
# by the time control so a fast tc naturally searches shallow.
NET_SIMS="${NET_SIMS:-160}"

GAMES="${GAMES:-100}"
# MOVETIME (seconds per move) gives every move the same budget, so NET_SIMS
# actually fires -- the right knob for strength eval. Leave it unset to fall
# back to a whole-game TIME_CONTROL, where the per-move budget shrinks as the
# clock drains and deep search never happens.
MOVETIME="${MOVETIME:-}"
TIME_CONTROL="${TIME_CONTROL:-10+0.1}"
TIME_MARGIN_MS="${TIME_MARGIN_MS:-500}"
CONCURRENCY="${CONCURRENCY:-2}"

if [[ -n "$MOVETIME" ]]; then
    TIME_FLAG=(st="$MOVETIME")
else
    TIME_FLAG=(tc="$TIME_CONTROL")
fi

STOCKFISH_LEVELS=(1350 1500 1700 1900 2100)

TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
RUN_DIR="$ROOT_DIR/benchmarks/results/eval_$TIMESTAMP"

fail() { echo "ERROR: $*" >&2; exit 1; }

require_executable() {
    [[ -n "$1" && -x "$1" ]] || fail "$2 is missing or not executable: $1"
}

run_match() {
    local match_name="$1"; shift

    local match_dir="$RUN_DIR/$match_name"
    mkdir -p "$match_dir"

    echo
    echo "=================================================================="
    echo "Running: $match_name  ($GAMES games, ${TIME_FLAG[*]}, sims=$NET_SIMS)"
    echo "Output:  $match_dir"
    echo "=================================================================="

    set -o pipefail
    "$CUTECHESS" \
        "$@" \
        -each "${TIME_FLAG[@]}" timemargin="$TIME_MARGIN_MS" \
        -rounds "$GAMES" \
        -repeat \
        -concurrency "$CONCURRENCY" \
        -ratinginterval 10 \
        -pgnout "$match_dir/games.pgn" \
        2>&1 | tee "$match_dir/cutechess.log"
    local exit_code="${PIPESTATUS[0]}"
    set +o pipefail

    [[ "$exit_code" -eq 0 ]] || echo "Match '$match_name' exited $exit_code" >&2
}

require_executable "$CUTECHESS" "cutechess-cli"
require_executable "$NEURAL_ENGINE" "Neural engine"
require_executable "$RANDOM_ENGINE" "Random engine"
[[ -f "$MODEL" ]] || fail "Checkpoint not found: $MODEL"
# cutechess runs engines from its own cwd, so hand it an absolute checkpoint path.
MODEL="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"

mkdir -p "$RUN_DIR"

{
    echo "Timestamp: $TIMESTAMP"
    echo "Neural engine: $NEURAL_ENGINE"
    echo "Checkpoint: $MODEL"
    echo "Simulations: $NET_SIMS"
    echo "Random engine: $RANDOM_ENGINE"
    echo "Stockfish: ${STOCKFISH:-<not found>}"
    echo "Games per match: $GAMES"
    echo "Time control: $TIME_CONTROL (margin ${TIME_MARGIN_MS} ms)"
    echo "Stockfish levels: ${STOCKFISH_LEVELS[*]}"
    if git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "Git commit: $(git -C "$ROOT_DIR" rev-parse HEAD)"
    fi
} > "$RUN_DIR/metadata.txt"

net_engine=(-engine name=NeuralNet cmd="$NEURAL_ENGINE" arg="$MODEL" \
    proto=uci option.Simulations="$NET_SIMS")

# 1. The baseline we should already be beating: the random mover.
run_match "neural-vs-random" \
    "${net_engine[@]}" \
    -engine name=Random cmd="$RANDOM_ENGINE" proto=uci

# 2. Climb the Stockfish limited-strength ladder.
if [[ -n "$STOCKFISH" && -x "$STOCKFISH" ]]; then
    for elo in "${STOCKFISH_LEVELS[@]}"; do
        run_match "neural-vs-sf-$elo" \
            "${net_engine[@]}" \
            -engine name="Stockfish-$elo" cmd="$STOCKFISH" proto=uci \
                option.UCI_LimitStrength=true \
                option.UCI_Elo="$elo" \
                option.Threads=1 \
                option.Hash=16
    done
else
    echo "Stockfish not found; skipping the Elo ladder." >&2
fi

echo
echo "Eval complete -> $RUN_DIR"
