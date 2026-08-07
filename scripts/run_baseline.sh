#!/usr/bin/env bash

set -uo pipefail


# Configuration
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CUTECHESS="${CUTECHESS:-$ROOT_DIR/build/cutechess/cutechess-cli}"
ENGINE="${ENGINE:-$ROOT_DIR/build/engine}"
STOCKFISH="${STOCKFISH:-$(command -v stockfish 2>/dev/null || true)}"

GAMES="${GAMES:-1000}"
TIME_CONTROL="${TIME_CONTROL:-1+0.1}"
TIME_MARGIN_MS="${TIME_MARGIN_MS:-100}"
CONCURRENCY="${CONCURRENCY:-4}"

# Begin at Stockfish's low limited-strength range.
STOCKFISH_LEVELS=(
    1350
    1500
    1700
    1900
    2100
    2300
    2500
)

TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
RUN_DIR="$ROOT_DIR/benchmarks/results/$TIMESTAMP"
SUMMARY_CSV="$RUN_DIR/summary.csv"

# SUMMARIZER="$ROOT_DIR/benchmarks/summarize_match.py"


# Helpers
fail() {
    echo "ERROR: $*" >&2
    exit 1
}

require_executable() {
    local path="$1"
    local description="$2"

    if [[ -z "$path" || ! -x "$path" ]]; then
        fail "$description is missing or not executable: $path"
    fi
}

sha256_file() {
    local path="$1"

    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$path" | awk '{print $1}'
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$path" | awk '{print $1}'
    else
        echo "unavailable"
    fi
}

run_match() {
    local match_name="$1"
    shift

    local match_dir="$RUN_DIR/$match_name"
    local pgn_file="$match_dir/games.pgn"
    local log_file="$match_dir/cutechess.log"
    local command_file="$match_dir/command.txt"

    mkdir -p "$match_dir"

    printf '%q ' "$CUTECHESS" "$@" > "$command_file"
    printf '\n' >> "$command_file"

    echo
    echo "=================================================================="
    echo "Running: $match_name"
    echo "Games: $GAMES"
    echo "Output: $match_dir"
    echo "=================================================================="

    # pipefail ensures a Cute Chess failure is not hidden by tee.
    set -o pipefail

    "$CUTECHESS" \
        "$@" \
        -each tc="$TIME_CONTROL" timemargin="$TIME_MARGIN_MS" \
        -rounds "$GAMES" \
        -repeat \
        -concurrency "$CONCURRENCY" \
        -ratinginterval 20 \
        -pgnout "$pgn_file" \
        2>&1 | tee "$log_file"

    local cutechess_exit="${PIPESTATUS[0]}"

    set +o pipefail

    if [[ "$cutechess_exit" -ne 0 ]]; then
        echo "Match failed with exit code $cutechess_exit" >&2

    #     python3 "$SUMMARIZER" \
    #         --name "$match_name" \
    #         --log "$log_file" \
    #         --pgn "$pgn_file" \
    #         --games-requested "$GAMES" \
    #         --status "failed:$cutechess_exit" \
    #         >> "$SUMMARY_CSV"

    #     return "$cutechess_exit"
    fi

    # python3 "$SUMMARIZER" \
    #     --name "$match_name" \
    #     --log "$log_file" \
    #     --pgn "$pgn_file" \
    #     --games-requested "$GAMES" \
    #     --status "completed" \
    #     "$@" \
    #     >> "$SUMMARY_CSV"
}


# Validation
require_executable "$CUTECHESS" "cutechess-cli"
require_executable "$ENGINE" "Your UCI engine"
require_executable "$STOCKFISH" "Stockfish"

mkdir -p "$RUN_DIR"


# CSV header
cat > "$SUMMARY_CSV" <<'CSV'
match,opponent_anchor_elo,games_requested,games_completed,wins,losses,draws,score,score_percent,relative_elo,estimated_engine_elo,draw_percent,white_wins,black_wins,status,pgn,log
CSV


# Metadata
{
    echo "Benchmark timestamp: $TIMESTAMP"
    echo "Root directory: $ROOT_DIR"
    echo "Engine: $ENGINE"
    echo "Engine SHA-256: $(sha256_file "$ENGINE")"
    echo "Stockfish: $STOCKFISH"
    echo "Stockfish SHA-256: $(sha256_file "$STOCKFISH")"
    echo "Cute Chess: $CUTECHESS"
    echo "Cute Chess SHA-256: $(sha256_file "$CUTECHESS")"
    echo "Games per match: $GAMES"
    echo "Time control: $TIME_CONTROL"
    echo "Time margin: ${TIME_MARGIN_MS} ms"
    echo "Concurrency: $CONCURRENCY"
    echo "Stockfish levels: ${STOCKFISH_LEVELS[*]}"
    echo "Operating system: $(uname -a)"

    if git -C "$ROOT_DIR" rev-parse --is-inside-work-tree \
        >/dev/null 2>&1; then
        echo "Git commit: $(git -C "$ROOT_DIR" rev-parse HEAD)"
        echo "Git branch: $(git -C "$ROOT_DIR" branch --show-current)"
        echo "Git dirty:"
        git -C "$ROOT_DIR" status --porcelain
    fi
} > "$RUN_DIR/metadata.txt"

# Capture engine handshakes.
{
    printf "uci\nisready\nquit\n" | "$ENGINE"
} > "$RUN_DIR/engine-uci.txt" 2>&1

{
    printf "uci\nisready\nquit\n" | "$STOCKFISH"
} > "$RUN_DIR/stockfish-uci.txt" 2>&1


# 1. Random engine self-play
run_match \
    "selfplay-random-v0" \
    -engine name=RandomA cmd="$ENGINE" proto=uci \
    -engine name=RandomB cmd="$ENGINE" proto=uci \
    || echo "Self-play failed; continuing." >&2


# 2. Stockfish Elo ladder
for elo in "${STOCKFISH_LEVELS[@]}"; do
    run_match \
        "random-v0-vs-sf-$elo" \
        -engine name=RandomV0 cmd="$ENGINE" proto=uci \
        -engine name="Stockfish-$elo" \
            cmd="$STOCKFISH" \
            proto=uci \
            option.UCI_LimitStrength=true \
            option.UCI_Elo="$elo" \
            option.Threads=1 \
            option.Hash=16 \
        || echo "Stockfish $elo match failed; continuing." >&2
done

echo
echo "Benchmark complete."
echo "Summary: $SUMMARY_CSV"
echo "Metadata: $RUN_DIR/metadata.txt"

# # Pretty-print the summary if column is available.
# if command -v column >/dev/null 2>&1; then
#     echo
#     column -s, -t "$SUMMARY_CSV"
# fi