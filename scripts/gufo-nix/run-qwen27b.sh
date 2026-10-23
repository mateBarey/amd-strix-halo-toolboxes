#!/usr/bin/env bash
# Start and check the installed model from either the host or its Toolbx.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/gufo-nix/run-qwen27b.sh [--replace-flash-next]
       bash scripts/gufo-nix/run-qwen27b.sh --help

Works inside the existing Gufo Nix Toolbx, including its development shell.
From the host, routes Gufo commands through the existing gufo-rocm-nix Toolbx.
Starts/reuses qwen38-27b-uncensored on port 8081, then requires real text and
image completions. Default context 262144, one session, thinking/speculation off.

--replace-flash-next explicitly stops your own Gufo server for the exact old
Flash Next model path before starting 27B. Other models/processes are preserved.
Without this flag, no existing inference process is stopped.

Overrides: GUFO_TOOLBOX_NAME, GUFO_BIN, GUFO_MODEL_PATH, GUFO_VISION_PATH,
GUFO_PORT, GUFO_CONTEXT, GUFO_CACHE_RAM_BYTES, GUFO_STARTUP_TIMEOUT,
GUFO_TUNE_CACHE, GUFO_TEST_IMAGE, GUFO_PREVIOUS_MODEL_PATH (replacement only).
Reports/logs: $HOME/.cache/gufo/qwen27b/run-<unique>/
The server remains running after successful checks. Hermes runs on the host;
the helper prints the matching command after both checks pass.
HELP
}

replace_previous=0
if (( $# )); then
    (( $# == 1 )) || { usage >&2; exit 2; }
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --replace-flash-next) replace_previous=1 ;;
        *) usage >&2; exit 2 ;;
    esac
fi

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TOOLBOX_NAME=${GUFO_TOOLBOX_NAME:-gufo-rocm-nix}
GUFO_BIN=${GUFO_BIN:-/usr/local/bin/gufo}
MODEL=${GUFO_MODEL_PATH:-/home/esckobear/VMl/Qwen3.8-27B-Uncensored-Q4_K_M.gguf}
VISION=${GUFO_VISION_PATH:-/home/esckobear/VMl/mmproj-Qwen3.8-27B-Uncensored-BF16.gguf}
PORT=${GUFO_PORT:-8081}
CONTEXT=${GUFO_CONTEXT:-262144}
CACHE_BYTES=${GUFO_CACHE_RAM_BYTES:-4294967296}
TIMEOUT=${GUFO_STARTUP_TIMEOUT:-300}
SERVED_NAME=qwen38-27b-uncensored
BASE_URL="http://127.0.0.1:$PORT/v1"
PREVIOUS_MODEL=${GUFO_PREVIOUS_MODEL_PATH:-/home/esckobear/VMl/Qwen3.8-Flash-Next-Uncensored-Gufo-Q4Mix.gguf}

[[ "$PORT" =~ ^[1-9][0-9]{0,4}$ ]] && (( PORT <= 65535 )) || {
    printf 'GUFO_PORT must be an integer from 1 to 65535.\n' >&2; exit 2;
}
[[ "$CONTEXT" =~ ^[1-9][0-9]*$ && "$TIMEOUT" =~ ^[1-9][0-9]*$ && "$CACHE_BYTES" =~ ^[0-9]+$ ]] || {
    printf 'Context and timeout must be positive integers; cache bytes must be nonnegative.\n' >&2; exit 2;
}
for program in curl python3 flock mktemp; do
    command -v "$program" >/dev/null || { printf 'Missing command: %s\n' "$program" >&2; exit 1; }
done

RUNNER=()
if [[ ! -e /run/.toolboxenv && ! -e /run/.containerenv ]]; then
    command -v toolbox >/dev/null || { printf 'Missing host command: toolbox\n' >&2; exit 1; }
    RUNNER=(toolbox run -c "$TOOLBOX_NAME")
fi
in_runtime() {
    "${RUNNER[@]}" env -u LD_LIBRARY_PATH -u LD_PRELOAD "$@"
}

state_dir="$HOME/.cache/gufo/qwen27b"
mkdir -p -- "$state_dir"
exec 9> "$state_dir/port-$PORT.lock"
flock -n 9 || { printf 'Another startup/check is using port %s.\n' "$PORT" >&2; exit 1; }
run_dir=$(mktemp -d "$state_dir/run-XXXXXXXX")
log_file="$run_dir/server.log"
printf 'Startup reports: %s\n' "$run_dir"

# Validate the new files before any explicitly requested previous-model stop.
in_runtime bash -c '
    test -x "$1" || { printf "Cannot execute Gufo: %s\n" "$1" >&2; exit 1; }
    for path in "$2" "$3"; do
        test -f "$path" && test -r "$path" && test -s "$path" || {
            printf "Cannot read nonempty model/projector: %s\n" "$path" >&2; exit 1;
        }
    done
' bash "$GUFO_BIN" "$MODEL" "$VISION"
in_runtime /usr/local/bin/gufo-convert-projector "$VISION" --check-only
in_runtime "$GUFO_BIN" --version | tee "$run_dir/version.txt"

snapshot_memory() {
    local destination=$1
    if ! GUFO_TOOLBOX_NAME="$TOOLBOX_NAME" bash "$REPO/scripts/gufo-nix/inspect-memory.sh" \
        > "$destination" 2> "$destination.stderr"; then
        printf 'Memory inspection reported an error; details: %s\n' "$destination.stderr" >&2
        cat "$destination.stderr" >&2
    fi
}
api_models() {
    curl --noproxy '*' -fsS --connect-timeout 2 --max-time 5 "$BASE_URL/models" 2>/dev/null
}
expected_model() {
    python3 -c '
import json, sys
try:
    value = json.load(sys.stdin)
    ok = any(item.get("id") == sys.argv[1] for item in value.get("data", []))
except (ValueError, AttributeError, TypeError):
    ok = False
sys.exit(0 if ok else 1)
' "$SERVED_NAME" <<< "$1"
}
failure() {
    printf 'Gufo startup/inference checks failed. Reports: %s\n' "$run_dir" >&2
    [[ ! -f "$log_file" ]] || tail -n 60 "$log_file" >&2
    snapshot_memory "$run_dir/memory-failure.json"
    printf 'Current memory report: %s\n' "$run_dir/memory-failure.json" >&2
    python3 - "$run_dir/memory-failure.json" <<'PY' >&2
import json, sys
try:
    with open(sys.argv[1]) as source:
        report = json.load(source)
    if not isinstance(report, dict):
        raise ValueError("report is not a JSON object")
except (OSError, ValueError) as error:
    print(f"Memory summary unavailable: {error}")
    sys.exit(0)

def rows(value):
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []

def byte_count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0

def size(value):
    return f"{value} bytes ({value / 2**30:.2f} GiB)" if byte_count(value) else "unavailable"

hip = report.get("hip")
if isinstance(hip, dict):
    print(f"HIP inspection status: {hip.get('status', 'unavailable')}")
    for device in rows(hip.get("devices")):
        print(f"HIP GPU {device.get('index', '?')}: free={size(device.get('free_bytes'))}; total={size(device.get('total_bytes'))}")
host = report.get("host_memory")
print(f"Host MemAvailable: {size(host.get('MemAvailable') if isinstance(host, dict) else None)}")
processes = rows(report.get("gufo_processes"))
print(f"Verified Gufo processes reported: {len(processes)}")
for process in processes:
    print(f"Gufo PID {process.get('pid', '?')}: RSS={size(process.get('rss_bytes'))}; model={json.dumps(process.get('model'))}")
clients = rows(report.get("kfd_clients"))
print(f"KFD clients reported: {len(clients)}")
for client in clients:
    counters = client.get("accounted_vram_bytes_by_gpu")
    values = [value for value in counters.values() if byte_count(value)] if isinstance(counters, dict) else []
    total = sum(values) if values else None
    print(f"KFD PID {client.get('pid', '?')}: accounted VRAM counters total={size(total)}")
PY
    exit 1
}

if catalog=$(api_models); then
    expected_model "$catalog" || {
        printf 'Port %s serves another model; no process on that port was selected for reuse.\n' "$PORT" >&2
        failure
    }
    printf 'Checking existing %s on port %s.\n' "$SERVED_NAME" "$PORT"
else
    # Detect an occupied non-API port before replacing any prior model. This
    # is an early check, not a reservation; Gufo still owns the actual bind.
    python3 - "$PORT" <<'PY'
import socket, sys
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", int(sys.argv[1])))
except OSError as error:
    sys.exit(f"Cannot use port {sys.argv[1]}: {error}. Choose GUFO_PORT before replacing a model.")
PY
    if (( replace_previous )); then
        snapshot_memory "$run_dir/memory-before-replacement.json"
        in_runtime python3 "$REPO/scripts/gufo-nix/stop-previous-gufo.py" --model "$PREVIOUS_MODEL"
    fi
    snapshot_memory "$run_dir/memory-before.json"
    if ! in_runtime "$GUFO_BIN" diagnose --json --section inventory \
        > "$run_dir/inventory-before.json" 2> "$run_dir/inventory-before.stderr"; then
        cat "$run_dir/inventory-before.stderr" >&2
    fi
    SERVER_ENV=(env -u LD_LIBRARY_PATH -u LD_PRELOAD -u GUFO_HIPBLASLT_PLAN_CACHE)
    if [[ -n "${GUFO_TUNE_CACHE:-}" ]]; then
        if in_runtime bash -c 'test -r "$1" && test -s "$1"' bash "$GUFO_TUNE_CACHE"; then
            SERVER_ENV+=("GUFO_HIPBLASLT_PLAN_CACHE=$GUFO_TUNE_CACHE")
        else
            printf 'Optional tuning cache unavailable; using Gufo defaults: %s\n' "$GUFO_TUNE_CACHE" >&2
        fi
    fi
    CMD=("$GUFO_BIN" serve --host 0.0.0.0 --port "$PORT" --sessions 1 llm
         --model "$MODEL" --mmproj "$VISION" --served-model-name "$SERVED_NAME"
         --context "$CONTEXT" --cache-ram-bytes "$CACHE_BYTES" --speculative off --think off)
    printf '%q ' "${CMD[@]}" > "$run_dir/command.txt"
    printf '\n' >> "$run_dir/command.txt"
    nohup "${RUNNER[@]}" "${SERVER_ENV[@]}" "${CMD[@]}" > "$log_file" 2>&1 < /dev/null 9>&- &
    server_pid=$!
    printf '%s\n' "$server_pid" > "$run_dir/launcher.pid"
    printf 'Starting Gufo; launcher PID %s, log %s\n' "$server_pid" "$log_file"
    deadline=$((SECONDS + TIMEOUT))
    ready=0
    while (( SECONDS < deadline )); do
        if catalog=$(api_models); then
            expected_model "$catalog" || failure
            ready=1
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            wait "$server_pid" || true
            failure
        fi
        sleep 1
    done
    if (( ! ready )); then
        printf 'Readiness timeout; the process may still be loading. Inspect its log before another start.\n' >&2
        failure
    fi
fi

CHECK=(python3 "$REPO/scripts/gufo-nix/check-server.py" --base-url "$BASE_URL"
       --model "$SERVED_NAME" --output "$run_dir/inference-check.json")
[[ -z "${GUFO_TEST_IMAGE:-}" ]] || CHECK+=(--image "$GUFO_TEST_IMAGE")
"${CHECK[@]}" || failure
snapshot_memory "$run_dir/memory-ready.json"
printf 'Text and vision checks passed. Gufo is serving at %s\n' "$BASE_URL"
if [[ "${GUFO_PRINT_HERMES_COMMAND:-1}" != 0 ]]; then
    printf 'Start Hermes from the HOST (exit Toolbx first if currently inside):\n'
    printf '  GUFO_TOOLBOX_NAME=%q GUFO_PORT=%q GUFO_CONTEXT=%q bash %q\n' \
        "$TOOLBOX_NAME" "$PORT" "$CONTEXT" "$REPO/scripts/gufo-nix/launch-hermes-qwen27b.sh"
fi
