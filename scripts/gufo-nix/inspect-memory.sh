#!/usr/bin/env bash
# Query the existing Gufo runtime without loading a model or building packages.
set -euo pipefail

if (( $# )); then
    if (( $# == 1 )) && [[ "$1" == --help || "$1" == -h ]]; then
        printf '%s\n' \
            'Usage: bash scripts/gufo-nix/inspect-memory.sh' \
            'Prints JSON memory/runtime/process evidence using the installed Gufo stack.' \
            'Works inside Toolbx or from the host; GUFO_TOOLBOX_NAME defaults to gufo-rocm-nix.' \
            'HIP is initialized, but no model or benchmark buffer is allocated.'
        exit 0
    fi
    printf 'Unexpected arguments. Use --help.\n' >&2
    exit 2
fi

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ ! -e /run/.toolboxenv && ! -e /run/.containerenv ]]; then
    command -v toolbox >/dev/null || { printf 'Missing host command: toolbox\n' >&2; exit 1; }
    exec toolbox run -c "${GUFO_TOOLBOX_NAME:-gufo-rocm-nix}" \
        env -u LD_LIBRARY_PATH -u LD_PRELOAD \
        bash "$repo/scripts/gufo-nix/inspect-memory.sh"
fi

unset LD_LIBRARY_PATH LD_PRELOAD
flake="path:$repo/toolboxes/gufo-nix"
installed=$(readlink -f /opt/gufo)
expected=$(/usr/local/bin/nix path-info --offline --no-update-lock-file "$flake#default")
if [[ "$installed" != "$expected" ]]; then
    printf 'Cannot inspect a different runtime: checkout Gufo differs from installed /opt/gufo.\n' >&2
    exit 1
fi
clr=$(/usr/local/bin/nix eval --offline --no-update-lock-file --raw \
    "$flake#packages.x86_64-linux.default.rocmPackages.clr.outPath")
rocr=$(/usr/local/bin/nix eval --offline --no-update-lock-file --raw \
    "$flake#packages.x86_64-linux.default.rocmRuntime.outPath")
[[ -r "$clr/lib/libamdhip64.so" && -d "$rocr/lib" ]] || {
    printf 'Installed HIP/ROCr library paths are unavailable.\n' >&2
    exit 1
}
# Set the loader environment before Python starts, matching Gufo's wrapper.
exec env -u LD_PRELOAD LD_LIBRARY_PATH="$rocr/lib" \
    /usr/bin/python3 "$repo/toolboxes/gufo-nix/gufo_memory.py" \
    --hip-library "$clr/lib/libamdhip64.so" --rocr-library-dir "$rocr/lib"
