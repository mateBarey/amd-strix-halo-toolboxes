#!/usr/bin/env bash
# CPU-only image verification in a disposable, unmounted container. No Toolbx or GPU needed.
set -euo pipefail

if (( $# != 1 )) || [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: CONTAINER_ENGINE=podman scripts/gufo-nix/smoke-image.sh IMAGE
CONTAINER_ENGINE may be podman (default) or docker.
Checks the pinned Gufo version, installed command help, and a fresh non-root
Nix daemon build. Uses one disposable container without mounts, network or GPU
access. A successful result is CPU/image verification, not GPU qualification.
HELP
    if (( $# == 1 )) && [[ "$1" == --help || "$1" == -h ]]; then exit 0; fi
    exit 2
fi
engine=${CONTAINER_ENGINE:-podman}
case "$engine" in
    podman|docker) ;;
    *) printf 'CONTAINER_ENGINE must be podman or docker.\n' >&2; exit 2 ;;
esac
command -v "$engine" >/dev/null || { printf 'Missing container engine: %s\n' "$engine" >&2; exit 1; }
image=$1
[[ "$image" != -* ]] || { printf 'IMAGE must be a container image reference.\n' >&2; exit 2; }
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
revision=$(tr -d '\r\n' < "$repo_root/toolboxes/gufo-nix/gufo-revision")
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || { printf 'Invalid pinned Gufo revision.\n' >&2; exit 1; }
name="gufo-nix-smoke-$$-${RANDOM}"
cleanup() { "$engine" rm --force "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'Checking %s in disposable container %s...\n' "$image" "$name"
"$engine" run --interactive --rm --name "$name" --network=none --user 0 \
    --entrypoint /bin/bash "$image" -s -- "$revision" <<'CONTAINER'
set -euo pipefail
expected_revision=$1
version=$(/usr/local/bin/gufo --version)
[[ "$version" == "gufo version development ($expected_revision)" ]] || {
    printf 'Unexpected Gufo version: %s\n' "$version" >&2
    exit 1
}
printf '%s\n' "$version"
/usr/local/bin/gufo --help >/dev/null
/opt/gufo/bin/gufo-kernel-bench --help >/dev/null
/opt/gufo/bin/tune_hipblaslt --help >/dev/null
/usr/local/bin/gufo-tune-hipblaslt --help >/dev/null
/usr/local/bin/gufo-convert-projector --help >/dev/null
/usr/local/bin/gufo-profile --help >/dev/null
# Use the real Nix binary here: the first wrapper invocation must be non-root.
/opt/nix/bin/nix --version
[[ "$(stat -c %u /nix)" == 0 && "$(stat -c %u /nix/store)" == 0 ]]
store_mode=$(stat -c %a /nix/store)
(( (8#$store_mode & 2) == 0 )) || { printf '/nix/store is world-writable.\n' >&2; exit 1; }
[[ -x /usr/local/bin/nix && -x /usr/local/sbin/gufo-nix-start-daemon ]]
for comm in /proc/[0-9]*/comm; do
    process_name=''
    IFS= read -r process_name < "$comm" 2>/dev/null || continue
    [[ "$process_name" != nix-daemon ]] || {
        printf 'Smoke requires a fresh daemon start through the non-root wrapper.\n' >&2
        exit 1
    }
done
useradd --create-home --uid 20001 --shell /bin/bash gufo-smoke
# Resolve the trusted image's Nix-store bash; no flake download or host /bin/sh
# dependency is needed by the tiny build, including when sandboxing is enabled.
IFS= read -r shebang < /opt/gufo/bin/gufo
builder=${shebang#\#!}
builder=${builder# }
builder=${builder%% *}
[[ "$builder" == /nix/store/*/bin/bash && -x "$builder" ]] || {
    printf 'Could not resolve the installed Gufo wrapper Nix-store bash.\n' >&2
    exit 1
}
user_nix() {
    runuser -u gufo-smoke -- env \
        HOME=/home/gufo-smoke USER=gufo-smoke LOGNAME=gufo-smoke \
        GUFO_SMOKE_BASH="$builder" \
        /usr/local/bin/nix --extra-experimental-features nix-command --store daemon "$@"
}
user_nix store ping
# Nix develop catches interactive-Bash realization errors and may still exit0
# by using a fallback Bash. Check its separately selected Bash first using a
# command that propagates the failure; prohibit both builds and substitution.
offline_shell_options=(--offline --max-jobs 0 --option substitute false)
user_nix build "${offline_shell_options[@]}" --no-link nixpkgs#bashInteractive
shell_log=$(mktemp)
if ! shell_version=$(user_nix develop "${offline_shell_options[@]}" \
    /opt/gufo-shell --command gufo --version 2>"$shell_log"); then
    cat "$shell_log" >&2
    rm -f "$shell_log"
    exit 1
fi
if grep -Eq 'error \(ignored\)|derivations? will be built|building .*/nix/store/' "$shell_log"; then
    printf 'Offline developer shell attempted a build or used an error fallback:\n' >&2
    cat "$shell_log" >&2
    rm -f "$shell_log"
    exit 1
fi
cat "$shell_log" >&2
rm -f "$shell_log"
[[ "$shell_version" == "$version" ]]
expression='let
  builder = builtins.storePath (builtins.getEnv "GUFO_SMOKE_BASH");
in derivation {
  name = "gufo-nix-daemon-smoke";
  system = "x86_64-linux";
  inherit builder;
  args = [ "-euc" "test \"$UID\" -gt 0; printf \"gufo-nix-smoke-ok\\nbuild_uid=%s\\n\" \"$UID\" > \"$out\"" ];
}'
result=$(user_nix build --offline --impure --no-link --print-out-paths --expr "$expression")
[[ "$result" == /nix/store/*-gufo-nix-daemon-smoke && -f "$result" ]]
[[ "$(head -n 1 "$result")" == gufo-nix-smoke-ok ]]
builder_uid=$(sed -n 's/^build_uid=//p' "$result")
[[ "$builder_uid" =~ ^[1-9][0-9]*$ && "$builder_uid" != 20001 ]]
builder_account=$(getent passwd "$builder_uid")
[[ "${builder_account%%:*}" == nixbld* ]] || {
    printf 'Expected a nixbld build user; got: %s\n' "$builder_account" >&2
    exit 1
}
printf 'Non-root Nix wrapper started the daemon and built successfully as %s.\n' "${builder_account%%:*}"
printf 'CPU image smoke checks passed. GPU execution was not tested.\n'
CONTAINER
