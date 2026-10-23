#!/usr/bin/env bash
# Run this on the HOST. Uses the Gufo binary baked into the gufo-rocm-nix image.
# It does not build Gufo, install Nix, alter host profiles, or tune any kernel.
set -euo pipefail

if test -e /run/.toolboxenv; then
    printf '%s\n' \
        'Run this launcher on the HOST, not inside Toolbx.' \
        'Type exit to leave the toolbox, then run scripts/gufo-nix/launch-hermes-qwen27b.sh on the host.' >&2
    exit 1
fi

usage() {
    cat <<'HELP'
Usage: scripts/gufo-nix/launch-hermes-qwen27b.sh [--help]
Run on the HOST with toolbox, curl, python3, and hermes available.
Starts Gufo in gufo-rocm-nix, waits for the expected /v1/models ID,
configures Hermes, and opens the project directory with Hermes.

Defaults:
  Gufo: /usr/local/bin/gufo inside the toolbox
  Model: /home/esckobear/VMl/Qwen3.8-27B-Uncensored-Q4_K_M.gguf
  Projector: /home/esckobear/VMl/mmproj-Qwen3.8-27B-Uncensored-BF16.gguf
  Context: 262144; speculation: off; thinking: off; port: 8080
  Served model ID: qwen38-27b-uncensored

Convert the original F16 projector to the BF16 filename above before first use.
The input model/projector files remain unchanged. This launcher does not tune
Q4 or vision kernels and does not request a plan cache by default.

Overrides (environment variables):
  GUFO_TOOLBOX_NAME  GUFO_BIN           GUFO_PROJECT_ROOT
  GUFO_MODEL_PATH    GUFO_VISION_PATH   GUFO_PORT
  GUFO_CONTEXT       GUFO_CACHE_RAM_BYTES (default 4294967296)
  GUFO_LOG_FILE      GUFO_STARTUP_TIMEOUT (default 300 seconds)
  GUFO_TUNE_CACHE    Explicit opt-in path to a readable, nonempty plan cache.
                    An unavailable file is ignored; current vision ignores it.
HELP
}

if (( $# != 0 )); then
    if (( $# == 1 )) && [[ "$1" == --help || "$1" == -h ]]; then
        usage
        exit 0
    fi
    usage >&2
    exit 2
fi

PORT="${GUFO_PORT:-8080}"
TOOLBOX_NAME="${GUFO_TOOLBOX_NAME:-gufo-rocm-nix}"
PROJECT_ROOT="${GUFO_PROJECT_ROOT:-/home/esckobear/Downloads/Downloads_raw/micro_never_ends/Kimi_Agent_Quaternion Sphere 16 Centroids}"

GUFO_BIN="${GUFO_BIN:-/usr/local/bin/gufo}"
MODEL_PATH="${GUFO_MODEL_PATH:-/home/esckobear/VMl/Qwen3.8-27B-Uncensored-Q4_K_M.gguf}"
VISION_PATH="${GUFO_VISION_PATH:-/home/esckobear/VMl/mmproj-Qwen3.8-27B-Uncensored-BF16.gguf}"
TUNE_CACHE="${GUFO_TUNE_CACHE:-}"
LOG_FILE="${GUFO_LOG_FILE:-/tmp/gufo-qwen27b-server.log}"

SERVED_NAME="qwen38-27b-uncensored"
CONTEXT="${GUFO_CONTEXT:-262144}"
CACHE_RAM_BYTES="${GUFO_CACHE_RAM_BYTES:-4294967296}"
STARTUP_TIMEOUT="${GUFO_STARTUP_TIMEOUT:-300}"
BASE_URL="http://127.0.0.1:${PORT}/v1"

for program in toolbox curl python3 hermes; do
    command -v "$program" >/dev/null || {
        printf 'Missing host command: %s\n' "$program" >&2
        exit 1
    }
done

api_models() {
    curl -fsS --connect-timeout 2 --max-time 5 "$BASE_URL/models" 2>/dev/null
}

has_expected_model() {
    python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
    ok = any(item.get("id") == sys.argv[1] for item in data.get("data", []))
except (ValueError, AttributeError, TypeError):
    ok = False
sys.exit(0 if ok else 1)
' "$SERVED_NAME" <<< "$1"
}

show_failure() {
    printf 'Gufo failed to become ready. Log: %s\n' "$LOG_FILE" >&2
    tail -n 50 "$LOG_FILE" >&2 || true
    exit 1
}

if MODELS_JSON=$(api_models); then
    if ! has_expected_model "$MODELS_JSON"; then
        printf 'Port %s is already serving a different model. Stop that server or choose GUFO_PORT.\n' "$PORT" >&2
        printf 'Expected API model ID: %s\n' "$SERVED_NAME" >&2
        exit 1
    fi
    printf 'Reusing %s on port %s.\n' "$SERVED_NAME" "$PORT"
else
    # Validate against the installed image, with host dynamic-loader overrides removed.
    toolbox run -c "$TOOLBOX_NAME" env -u LD_LIBRARY_PATH -u LD_PRELOAD bash -c '
        test -x "$1" || { printf "Missing installed Gufo executable: %s\n" "$1" >&2; exit 1; }
        test -r "$2" || { printf "Cannot read model: %s\n" "$2" >&2; exit 1; }
        test -r "$3" || { printf "Cannot read BF16 projector: %s\n" "$3" >&2; exit 1; }
    ' bash "$GUFO_BIN" "$MODEL_PATH" "$VISION_PATH"

    # Set the variable INSIDE Toolbx. A host export need not propagate there.
    # This cache is optional; current vision explicitly ignores it.
    ENV_ARGS=(env -u LD_LIBRARY_PATH -u LD_PRELOAD -u GUFO_HIPBLASLT_PLAN_CACHE)
    if [[ -n "$TUNE_CACHE" ]] && toolbox run -c "$TOOLBOX_NAME" \
        env -u LD_LIBRARY_PATH -u LD_PRELOAD bash -c 'test -r "$1" && test -s "$1"' bash "$TUNE_CACHE"; then
        ENV_ARGS+=("GUFO_HIPBLASLT_PLAN_CACHE=$TUNE_CACHE")
        printf 'Using optional hipBLASLt cache: %s\n' "$TUNE_CACHE"
    else
        if [[ -n "$TUNE_CACHE" ]]; then
            printf 'Optional plan cache is unavailable or empty: %s; using Gufo defaults.\n' "$TUNE_CACHE"
        else
            printf 'No optional hipBLASLt cache requested; using Gufo defaults.\n'
        fi
    fi

    CMD=(
        "$GUFO_BIN" serve --host 0.0.0.0 --port "$PORT" --sessions 1 llm
        --model "$MODEL_PATH"
        --mmproj "$VISION_PATH"
        --served-model-name "$SERVED_NAME"
        --context "$CONTEXT"
        --cache-ram-bytes "$CACHE_RAM_BYTES"
        --speculative off
        --think off
    )

    printf 'Starting %s inside %s...\n' "$SERVED_NAME" "$TOOLBOX_NAME"
    nohup toolbox run -c "$TOOLBOX_NAME" "${ENV_ARGS[@]}" "${CMD[@]}" \
        > "$LOG_FILE" 2>&1 < /dev/null &
    SERVER_PID=$!
    printf 'Waiting up to %s seconds (launcher PID %s)...\n' "$STARTUP_TIMEOUT" "$SERVER_PID"

    READY=0
    DEADLINE=$((SECONDS + STARTUP_TIMEOUT))
    while (( SECONDS < DEADLINE )); do
        if MODELS_JSON=$(api_models); then
            if has_expected_model "$MODELS_JSON"; then
                READY=1
                break
            fi
            printf 'Another model appeared on port %s during startup.\n' "$PORT" >&2
            show_failure
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            show_failure
        fi
        sleep 1
    done
    if (( READY == 0 )); then
        printf 'The process may still be loading; inspect the log before starting another copy.\n' >&2
        show_failure
    fi
    printf 'Gufo is ready.\n'
fi

printf 'Configuring Hermes for %s...\n' "$SERVED_NAME"
hermes config set model.provider custom >/dev/null
hermes config set model.base_url "$BASE_URL" >/dev/null
hermes config set model.default "$SERVED_NAME" >/dev/null

# Current Hermes schema: top-level model_overrides -> provider -> model.
hermes config set "model_overrides.custom.${SERVED_NAME}.supports_tools" true >/dev/null
hermes config set "model_overrides.custom.${SERVED_NAME}.supports_vision" true >/dev/null
hermes config set "model_overrides.custom.${SERVED_NAME}.context_window" "$CONTEXT" >/dev/null

mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"
printf 'Starting Hermes in %s\n' "$PWD"
exec hermes
