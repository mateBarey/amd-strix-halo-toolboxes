#!/usr/bin/env bash
# Enter the pinned official development environment in an existing Toolbx.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: scripts/gufo-nix/develop.sh
       scripts/gufo-nix/develop.sh --command CMD [ARG...]
       scripts/gufo-nix/develop.sh --help

Run on the HOST from this checkout; reuses the existing gufo-rocm-nix Toolbx.
Checks that the checkout's Gufo output matches the installed package before
realizing dependencies. The pinned interactive Bash is fetched from a binary
cache only; missing official development dependencies may download or build.
Preserves the official development environment and its shell hooks.

Overrides: GUFO_TOOLBOX_NAME (default gufo-rocm-nix), GUFO_BUILD_CORES (4),
GUFO_NIX_MAX_JOBS (1). No host Nix installation, image build, container creation,
or Gufo package build is performed by this helper.
HELP
}

if (( $# )); then
    case "$1" in
        --help|-h)
            (( $# == 1 )) || { usage >&2; exit 2; }
            usage
            exit 0
            ;;
        --command)
            (( $# >= 2 )) || { usage >&2; exit 2; }
            ;;
        *) usage >&2; exit 2 ;;
    esac
fi

if [[ -e /run/.toolboxenv || -e /run/.containerenv ]]; then
    printf 'Run this helper on the HOST. Type exit to leave Toolbx first.\n' >&2
    exit 1
fi

toolbox_name=${GUFO_TOOLBOX_NAME:-gufo-rocm-nix}
cores=${GUFO_BUILD_CORES:-4}
max_jobs=${GUFO_NIX_MAX_JOBS:-1}
[[ "$toolbox_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ && ${#toolbox_name} -le 128 ]] || {
    printf 'GUFO_TOOLBOX_NAME must be a container name, without spaces or slashes.\n' >&2
    exit 2
}
[[ "$cores" =~ ^[1-9][0-9]*$ && "$max_jobs" =~ ^[1-9][0-9]*$ ]] || {
    printf 'GUFO_BUILD_CORES and GUFO_NIX_MAX_JOBS must be positive integers.\n' >&2
    exit 2
}
command -v toolbox >/dev/null || { printf 'Missing host command: toolbox\n' >&2; exit 1; }
flake_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../toolboxes/gufo-nix" && pwd)

exec toolbox run -c "$toolbox_name" env -u LD_LIBRARY_PATH -u LD_PRELOAD \
    /bin/bash -euo pipefail -c '
        flake="path:$1"
        container=$2
        cores=$3
        max_jobs=$4
        shift 4

        installed=$(readlink -f /opt/gufo)
        expected=$(/usr/local/bin/nix path-info --offline --no-update-lock-file "$flake#default")
        if [[ "$installed" != "$expected" ]]; then
            printf "Checkout Gufo output does not match the installed package.\nInstalled: %s\nCheckout: %s\n" \
                "$installed" "$expected" >&2
            exit 1
        fi

        cache_dir="$HOME/.cache/gufo/nix-devshell/$container"
        mkdir -p -- "$cache_dir"
        printf "Fetching pinned interactive Bash from the binary cache; source builds are disabled for this step.\n"
        /usr/local/bin/nix build --no-update-lock-file --max-jobs 0 \
            --out-link "$cache_dir/bash" "$flake#bash-interactive"
        printf "Entering the official Gufo development environment. Missing development dependencies may download or build.\n"
        exec /usr/local/bin/nix develop --no-update-lock-file --cores "$cores" --max-jobs "$max_jobs" \
            --profile "$cache_dir/environment" "$flake" "$@"
    ' bash "$flake_dir" "$toolbox_name" "$cores" "$max_jobs" "$@"
