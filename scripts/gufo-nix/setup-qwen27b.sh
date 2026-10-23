#!/usr/bin/env bash
# Run the checked-out Gufo Nix setup on the Fedora HOST as your ordinary user.
# Builds a new image/container; never modifies a previous installer checkout.
set -Eeuo pipefail

readonly GUFO_SETUP_ID='mateBarey-gufo-nix-repository-v2'
readonly GUFO_LEGACY_SETUP_ID='mateBarey-gufo-nix-v1:167275137ec25be536ea7bb74c5c612602e44efd6dfb697e84f1ff71c74eaea3'
readonly GUFO_SOURCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

say() { printf '\n==> %s\n' "$*"; }
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

help_text() {
    cat <<'HELP'
Usage: bash scripts/gufo-nix/setup-qwen27b.sh
       bash scripts/gufo-nix/setup-qwen27b.sh --help

Run on the Fedora HOST, as your normal user, from a fresh clone of the
repository's gufo-nix branch. Builds and CPU-smoke-tests its pinned image,
creates a separate Toolbx, converts/checks the 27B projector, saves a rerun
launcher, and opens Hermes with terminal input. No Git clone, patch, commit,
push, host package changes or automatic GPU tuning is performed here.

Default work/log directory: $HOME/VMl/gufo-nix-mateBarey
An existing recognized v1 installer directory is reused under a lock; its old
repo directory and existing containers remain unchanged.
Default model: /home/esckobear/VMl/Qwen3.8-27B-Uncensored-Q4_K_M.gguf
Input projector: /home/esckobear/VMl/mmproj-Qwen3.8-27B-Uncensored-f16.gguf
Output projector: /home/esckobear/VMl/mmproj-Qwen3.8-27B-Uncensored-BF16.gguf
Container: an unused gufo-rocm-nix[-N]; port: first free port from 8080.
Context: 262144; thinking/speculation: off.

Optional environment overrides:
  GUFO_SETUP_DIR, GUFO_MODEL_PATH, GUFO_F16_PROJECTOR, GUFO_VISION_PATH,
  GUFO_PROJECT_ROOT, GUFO_CONTEXT, GUFO_PORT, GUFO_BUILD_CORES,
  GUFO_NIX_MAX_JOBS, GUFO_STARTUP_TIMEOUT, GUFO_CACHE_RAM_BYTES.
Requires existing host podman, toolbox, python3, curl, hermes, flock, mktemp,
your model/projector files, and GPU devices. Rerun this command after an
interrupted setup; completed build layers are reused and logs are retained.
HELP
}

require_host() {
    [[ ! -e /run/.toolboxenv && ! -e /run/.containerenv ]] ||
        fail 'Run this on the Fedora HOST. Type exit to leave Toolbx first.'
    (( EUID != 0 )) || fail 'Run as your normal user, without sudo.'
    for program in podman toolbox python3 curl hermes flock mktemp; do
        command -v "$program" >/dev/null || fail "Missing host command: $program"
    done
    ( : </dev/tty ) 2>/dev/null ||
        fail 'Run this script in a terminal so Toolbx and Hermes can read input.'
    [[ -c /dev/kfd && -d /dev/dri ]] ||
        fail 'The host must expose /dev/kfd and /dev/dri for ROCm GPU execution.'
    [[ -f "$MODEL" && -r "$MODEL" && -s "$MODEL" ]] || fail "Cannot read model: $MODEL"
    [[ -f "$F16" && -r "$F16" && -s "$F16" ]] || fail "Cannot read projector: $F16"
    [[ "$F16" != "$BF16" ]] || fail 'Input and output projector paths must be different.'
    [[ "$CONTEXT" =~ ^[1-9][0-9]*$ ]] || fail 'GUFO_CONTEXT must be a positive integer.'
    [[ "$FIRST_PORT" =~ ^[1-9][0-9]*$ ]] && (( FIRST_PORT >= 1024 && FIRST_PORT <= 65535 )) ||
        fail 'GUFO_PORT must be a number between 1024 and 65535.'
    podman info >/dev/null
}

prepare_directory() {
    local previous_id='' marker_temp
    if [[ -e "$SETUP_DIR" ]]; then
        [[ -d "$SETUP_DIR" && ! -L "$SETUP_DIR" && -f "$SETUP_DIR/setup-id" ]] ||
            fail "Existing path is not a recognized Gufo work directory: $SETUP_DIR"
        previous_id=$(cat "$SETUP_DIR/setup-id")
        [[ "$previous_id" == "$GUFO_SETUP_ID" || "$previous_id" == "$GUFO_LEGACY_SETUP_ID" ]] ||
            fail "Different setup version at $SETUP_DIR. Set GUFO_SETUP_DIR to a new directory."
    else
        mkdir -p -- "$SETUP_DIR"
    fi
    exec 9> "$SETUP_DIR/setup.lock"
    flock -n 9 || fail "Another setup is using $SETUP_DIR"
    # Recheck under the lock. Only the exact previous installer is migrated;
    # its dirty repo is never read, reset, patched or used as this build context.
    if [[ -f "$SETUP_DIR/setup-id" ]]; then
        previous_id=$(cat "$SETUP_DIR/setup-id")
        [[ "$previous_id" == "$GUFO_SETUP_ID" || "$previous_id" == "$GUFO_LEGACY_SETUP_ID" ]] ||
            fail 'The setup marker changed while acquiring its lock.'
    fi
    if [[ "$previous_id" != "$GUFO_SETUP_ID" ]]; then
        marker_temp=$(mktemp "$SETUP_DIR/setup-id.XXXXXX")
        printf '%s\n' "$GUFO_SETUP_ID" > "$marker_temp"
        mv -- "$marker_temp" "$SETUP_DIR/setup-id"
    fi
}

choose_container_name() {
    local candidate='gufo-rocm-nix' suffix=0 status
    while true; do
        if podman container exists "$candidate"; then
            (( suffix += 1 ))
            candidate="gufo-rocm-nix-$suffix"
        else
            status=$?
            (( status == 1 )) || fail "Podman could not check container names (exit $status)."
            TOOLBOX_NAME=$candidate
            return
        fi
    done
}

choose_port() {
    python3 - "$FIRST_PORT" <<'PY'
import errno
import socket
import sys
first = int(sys.argv[1])
for port in range(first, min(first + 100, 65536)):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("0.0.0.0", port))
        except OSError as exc:
            if exc.errno in (errno.EADDRINUSE, errno.EACCES):
                continue
            raise
        print(port)
        break
else:
    sys.exit("No available port in the requested range; set GUFO_PORT to another starting port.")
PY
}

in_toolbox() {
    toolbox run -c "$TOOLBOX_NAME" env -u LD_LIBRARY_PATH -u LD_PRELOAD "$@"
}

save_launcher() {
    local temporary_launcher
    temporary_launcher=$(mktemp "$SETUP_DIR/start-qwen27b.XXXXXX")
    {
        printf '#!/usr/bin/env bash\n# Generated by repository scripts/gufo-nix/setup-qwen27b.sh\nset -euo pipefail\n'
        printf 'export GUFO_TOOLBOX_NAME=%q\n' "$TOOLBOX_NAME"
        printf 'export GUFO_BIN=%q\n' '/usr/local/bin/gufo'
        printf 'export GUFO_MODEL_PATH=%q\n' "$MODEL"
        printf 'export GUFO_VISION_PATH=%q\n' "$BF16"
        printf 'export GUFO_PROJECT_ROOT=%q\n' "$PROJECT"
        printf 'export GUFO_PORT=%q\n' "$PORT"
        printf 'export GUFO_CONTEXT=%q\n' "$CONTEXT"
        printf 'export GUFO_LOG_FILE=%q\n' "$SETUP_DIR/server.log"
        printf 'export GUFO_STARTUP_TIMEOUT=%q\n' "${GUFO_STARTUP_TIMEOUT:-300}"
        printf 'export GUFO_CACHE_RAM_BYTES=%q\n' "${GUFO_CACHE_RAM_BYTES:-4294967296}"
        printf 'export GUFO_TUNE_CACHE=%q\n' ''
        printf 'exec bash %q\n' "$REPO/scripts/gufo-nix/launch-hermes-qwen27b.sh"
    } > "$temporary_launcher"
    chmod 755 "$temporary_launcher"
    mv -- "$temporary_launcher" "$LAUNCHER"
    {
        printf 'Source checkout: %s\n' "$REPO"
        printf 'Container: %s\nImage: %s\nPort: %s\n' "$TOOLBOX_NAME" "$IMAGE" "$PORT"
        printf 'Model: %s\nProjector: %s\n' "$MODEL" "$BF16"
        printf '\nStart again from the host, without rebuilding:\n  bash %q\n' "$LAUNCHER"
        printf '\nOpen Nix development shell:\n  toolbox run -c %q /usr/local/bin/nix develop /opt/gufo-shell\n' "$TOOLBOX_NAME"
        printf '\nOptional hipBLASLt tuner (synthetic BF16 GEMM, not Q4/vision autotuning):\n'
        printf '  toolbox run -c %q /usr/local/bin/gufo-tune-hipblaslt --quick\n' "$TOOLBOX_NAME"
    } > "$SETUP_DIR/SETUP-RESULT.txt"
}

main() {
    if (( $# )); then
        [[ "$#" == 1 && ( "$1" == --help || "$1" == -h ) ]] ||
            fail 'Usage: bash scripts/gufo-nix/setup-qwen27b.sh [--help]'
        help_text
        return
    fi
    SETUP_DIR=${GUFO_SETUP_DIR:-"$HOME/VMl/gufo-nix-mateBarey"}
    [[ "$SETUP_DIR" == /* ]] || fail 'GUFO_SETUP_DIR must be an absolute path.'
    REPO=$GUFO_SOURCE_ROOT
    LAUNCHER="$SETUP_DIR/start-qwen27b.sh"
    IMAGE='localhost/gufo-nix:latest'
    MODEL=${GUFO_MODEL_PATH:-'/home/esckobear/VMl/Qwen3.8-27B-Uncensored-Q4_K_M.gguf'}
    F16=${GUFO_F16_PROJECTOR:-'/home/esckobear/VMl/mmproj-Qwen3.8-27B-Uncensored-f16.gguf'}
    BF16=${GUFO_VISION_PATH:-'/home/esckobear/VMl/mmproj-Qwen3.8-27B-Uncensored-BF16.gguf'}
    PROJECT=${GUFO_PROJECT_ROOT:-'/home/esckobear/Downloads/Downloads_raw/micro_never_ends/Kimi_Agent_Quaternion Sphere 16 Centroids'}
    CONTEXT=${GUFO_CONTEXT:-262144}
    FIRST_PORT=${GUFO_PORT:-8080}
    trap 'status=$?; printf "\nSetup stopped at line %s (exit %s). Work is kept at: %s\nRerun this script after resolving the error.\n" "$LINENO" "$status" "$SETUP_DIR" >&2; exit "$status"' ERR

    say 'Checking host tools, model paths and GPU devices'
    require_host
    prepare_directory
    say 'Building Gufo + Nix + matching ROCm inside the new image'
    printf 'The first build downloads dependencies and compiles Gufo. Keep this terminal open.\n'
    CONTAINER_ENGINE=podman bash "$REPO/scripts/gufo-nix/build-local.sh" "$IMAGE" 2>&1 |
        tee "$SETUP_DIR/build.log"

    choose_container_name
    say "Creating Toolbx $TOOLBOX_NAME"
    toolbox create -c "$TOOLBOX_NAME" --image "$IMAGE"
    in_toolbox /usr/local/bin/gufo --version
    in_toolbox /usr/local/bin/nix --version
    in_toolbox /usr/local/bin/gufo diagnose --fingerprint --json > "$SETUP_DIR/fingerprint.json"
    cat "$SETUP_DIR/fingerprint.json"

    if [[ -e "$BF16" || -L "$BF16" ]]; then
        say 'Checking the existing BF16 projector'
        in_toolbox /usr/local/bin/gufo-convert-projector "$BF16" --check-only
    else
        say 'Converting the 27B F16 projector into a separate BF16/F32 file'
        in_toolbox /usr/local/bin/gufo-convert-projector "$F16" "$BF16"
    fi

    PORT=$(choose_port)
    save_launcher
    say 'Image and projector are ready; starting Gufo and Hermes'
    printf 'Container: %s\nAPI: http://127.0.0.1:%s/v1\n' "$TOOLBOX_NAME" "$PORT"
    printf 'For every later launch: bash %q\n' "$LAUNCHER"
    printf 'Setup details: %s/SETUP-RESULT.txt\n' "$SETUP_DIR"
    # Do not leave the installer lock open in the detached server or Hermes.
    flock -u 9
    exec 9>&-
    exec bash "$LAUNCHER" </dev/tty
}


if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
