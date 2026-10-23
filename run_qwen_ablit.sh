#!/usr/bin/env bash
# One entry point for the already-installed Gufo Nix model and host Hermes.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: ./run_qwen_ablit.sh [start|stop|status]

  start   Start/check Qwen 27B, then open Hermes. Default; run on the HOST.
  stop    Gracefully stop your verified Gufo server for this exact model.
  status  Show its verified PIDs and the local API state. No model is loaded.

Gufo continues in the background after Hermes exits. Use this file's stop
command to release the model. Stop/status also work inside Toolbx.

Defaults: gufo-rocm-nix, port 8081, context 262144, Q4_K_M model and the
already-validated BF16 projector under $HOME/VMl.

GUFO_REPO selects the checkout. A copied file defaults to
$HOME/VMl/gufo-nix-mateBarey-github. Existing GUFO_MODEL_PATH,
GUFO_VISION_PATH, GUFO_PORT, GUFO_CONTEXT, GUFO_TOOLBOX_NAME,
GUFO_PROJECT_ROOT and GUFO_TUNE_CACHE overrides are preserved.
HELP
}

(( $# <= 1 )) || { usage >&2; exit 2; }
action=${1:-start}
case "$action" in
    --help|-h|help) usage; exit 0 ;;
    start|stop|status) ;;
    *) usage >&2; exit 2 ;;
esac

if [[ "$action" == start ]] && [[ -e /run/.toolboxenv || -e /run/.containerenv ]]; then
    printf '%s\n' \
        'Start this file from your Fedora HOST terminal so Hermes uses its host installation.' \
        'Type exit to leave Toolbx, then run: ~/VMl/run_qwen_ablit.sh' \
        'The stop and status commands also work here inside Toolbx.' >&2
    exit 1
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -n "${GUFO_REPO:-}" ]]; then
    repo=$GUFO_REPO
elif [[ -d "$script_dir/scripts/gufo-nix" ]]; then
    repo=$script_dir
else
    repo="$HOME/VMl/gufo-nix-mateBarey-github"
fi
[[ -d "$repo/scripts/gufo-nix" ]] || {
    printf 'Gufo checkout not found: %s\nSet GUFO_REPO to your checkout.\n' "$repo" >&2
    exit 1
}
repo=$(cd -- "$repo" && pwd)

export GUFO_TOOLBOX_NAME=${GUFO_TOOLBOX_NAME:-gufo-rocm-nix}
export GUFO_PORT=${GUFO_PORT:-8081}
export GUFO_CONTEXT=${GUFO_CONTEXT:-262144}
export GUFO_MODEL_PATH=${GUFO_MODEL_PATH:-"$HOME/VMl/Qwen3.8-27B-Uncensored-Q4_K_M.gguf"}
export GUFO_VISION_PATH=${GUFO_VISION_PATH:-"$HOME/VMl/mmproj-Qwen3.8-27B-Uncensored-BF16.gguf"}

[[ "$GUFO_PORT" =~ ^[1-9][0-9]{0,4}$ ]] && (( GUFO_PORT <= 65535 )) || {
    printf 'GUFO_PORT must be an integer from 1 to 65535.\n' >&2; exit 2;
}

if [[ "$action" == start ]]; then
    command -v hermes >/dev/null || {
        printf 'Hermes is missing from this host PATH; Gufo was not started.\n' >&2
        exit 1
    }
    for helper in run-qwen27b.sh launch-hermes-qwen27b.sh; do
        [[ -r "$repo/scripts/gufo-nix/$helper" ]] || {
            printf 'Missing launcher helper: %s\n' "$repo/scripts/gufo-nix/$helper" >&2
            exit 1
        }
    done
    printf 'Starting Qwen 27B in %s on port %s, context %s.\n' \
        "$GUFO_TOOLBOX_NAME" "$GUFO_PORT" "$GUFO_CONTEXT"
    printf 'Hermes opens after the text and image checks pass.\n'
    printf 'Gufo stays in the background after Hermes exits.\n'
    printf 'Stop it with: %q stop\nCheck it with: %q status\n' \
        "$script_dir/$(basename -- "${BASH_SOURCE[0]}")" \
        "$script_dir/$(basename -- "${BASH_SOURCE[0]}")"
    GUFO_PRINT_HERMES_COMMAND=0 bash "$repo/scripts/gufo-nix/run-qwen27b.sh"
    printf 'Opening Hermes with the checked Qwen 27B server...\n'
    export GUFO_REQUIRE_RUNNING=1
    exec bash "$repo/scripts/gufo-nix/launch-hermes-qwen27b.sh"
fi

# /proc identifies the real Gufo task; saved launcher PIDs are never signalled.
# This works on the host as well as inside its Toolbx and does not start a container.
command -v python3 >/dev/null || { printf 'Missing command: python3\n' >&2; exit 1; }
control=(python3 "$repo/scripts/gufo-nix/stop-previous-gufo.py" --model "$GUFO_MODEL_PATH")
[[ "$action" != status ]] || control+=(--status)
if report=$(env -u LD_LIBRARY_PATH -u LD_PRELOAD "${control[@]}"); then
    :
else
    result=$?
    printf '%s\n' "$report" >&2
    exit "$result"
fi
python3 -c '
import json, sys
action = sys.argv[1]
try:
    data = json.load(sys.stdin)
    pids = data["running_pids" if action == "status" else "stopped_pids"]
    if not isinstance(pids, list) or any(type(pid) is not int or pid <= 0 for pid in pids):
        raise ValueError("invalid PID list")
    print("Model:", data["model"])
    if pids:
        print(("Running Gufo PID(s): " if action == "status" else "Stopped Gufo PID(s): ")
              + ", ".join(map(str, pids)))
    else:
        print("No matching Gufo server is running." if action == "status"
              else "Already stopped: no matching Gufo server was running.")
except (ValueError, TypeError, KeyError) as error:
    sys.exit(f"Invalid Gufo {action} result: {error}")
' "$action" <<< "$report"

if [[ "$action" == status ]]; then
    url="http://127.0.0.1:$GUFO_PORT/v1"
    if catalog=$(curl --noproxy '*' -fsS --connect-timeout 2 --max-time 5 "$url/models" 2>/dev/null); then
        python3 -c '
import json, sys
try:
    names = [item.get("id") for item in json.load(sys.stdin).get("data", [])]
    if "qwen38-27b-uncensored" in names:
        print("API ready:", sys.argv[1], "(qwen38-27b-uncensored)")
    else:
        print("API responds but does not advertise qwen38-27b-uncensored:", sys.argv[1])
except (ValueError, AttributeError, TypeError):
    print("API returned an unreadable model catalog:", sys.argv[1])
' "$url" <<< "$catalog"
    else
        printf 'API not ready: %s\n' "$url"
    fi
fi
