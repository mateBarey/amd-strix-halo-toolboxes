#!/usr/bin/env bash
# Host-only additive image build; never enters, updates, or removes an existing Toolbx.
set -euo pipefail

if test -e /run/.toolboxenv; then
    printf '%s\n' 'Run this build helper on the HOST. Type exit to leave Toolbx first.' >&2
    exit 1
fi
if (( $# > 1 )) || [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: scripts/gufo-nix/build-local.sh [IMAGE]
Build the pinned Gufo Nix image, then run its CPU-only smoke checks.
Default image: localhost/gufo-nix:latest
CONTAINER_ENGINE=podman (default) or docker chooses the host image engine.
GUFO_BUILD_CORES=4 and GUFO_NIX_MAX_JOBS=1 bound Nix build parallelism.
Run on the host. No existing toolbox or model file is changed.
The Nix/ROCm build needs substantial free disk/RAM; it does not require a GPU.
HELP
    (( $# <= 1 )) && exit 0
    exit 2
fi
engine=${CONTAINER_ENGINE:-podman}
case "$engine" in
    podman|docker) ;;
    *) printf 'CONTAINER_ENGINE must be podman or docker.\n' >&2; exit 2 ;;
esac
command -v "$engine" >/dev/null || { printf 'Missing host command: %s\n' "$engine" >&2; exit 1; }
image=${1:-localhost/gufo-nix:latest}
[[ "$image" != -* ]] || { printf 'IMAGE must be a container image reference.\n' >&2; exit 2; }
cores=${GUFO_BUILD_CORES:-4}
jobs=${GUFO_NIX_MAX_JOBS:-1}
[[ "$cores" =~ ^[1-9][0-9]*$ && "$jobs" =~ ^[1-9][0-9]*$ ]] || {
    printf 'GUFO_BUILD_CORES and GUFO_NIX_MAX_JOBS must be positive integers.\n' >&2
    exit 2
}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
printf 'Building %s with %s (Nix cores=%s, max-jobs=%s).\n' "$image" "$engine" "$cores" "$jobs"
printf 'Build context: %s\n' "$repo_root"
"$engine" build \
    --build-arg "NIX_BUILD_CORES=$cores" \
    --build-arg "NIX_MAX_JOBS=$jobs" \
    --tag "$image" \
    --file "$repo_root/toolboxes/Dockerfile.gufo-nix" \
    "$repo_root"
CONTAINER_ENGINE="$engine" "$script_dir/smoke-image.sh" "$image"
printf 'Image built and CPU smoke checks passed: %s\n' "$image"
printf 'GPU inference remains a separate check on the Strix Halo host.\n'
