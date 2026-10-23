#!/usr/bin/env bash
# Run only inside the disposable Fedora/Nix bootstrap CI container.
# Evaluates the real developer flake, then tests shell GC/offline behavior
# with a small environment; it does not build Gufo, ROCm or PyTorch.
set -euo pipefail

[[ -x /opt/nix/bin/nix && $EUID -eq 0 ]]
export PATH="/opt/nix/bin:$PATH"
export GUFO_CHECK_FLAKE="path:$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../toolboxes/gufo-nix" && pwd)"

nix eval --impure --raw --expr '
  let
    toolbox = builtins.getFlake (builtins.getEnv "GUFO_CHECK_FLAKE");
    system = "x86_64-linux";
    official = toolbox.inputs.gufo.devShells.${system}.default;
    actual = toolbox.devShells.${system}.default;
    lib = toolbox.inputs.nixpkgs.lib;
    strings = builtins.map builtins.toString;
    containsAll = expected: available:
      builtins.all (item: builtins.elem item (strings available)) (strings expected);
    baselines = [ "ds4-reference" "llama-cpp-reference" "llama-cpp-mtp-reference" ];
  in
  assert containsAll official.nativeBuildInputs actual.nativeBuildInputs;
  assert containsAll official.buildInputs actual.buildInputs;
  assert builtins.elem (builtins.toString toolbox.packages.${system}.default)
    (strings actual.nativeBuildInputs);
  assert actual.env == official.env;
  assert official.env ? LD_LIBRARY_PATH && official.env ? TORCH_HOME;
  assert lib.hasPrefix official.shellHook actual.shellHook;
  assert builtins.all (name:
    builtins.toString toolbox.packages.${system}.${name} ==
    builtins.toString toolbox.inputs.gufo.packages.${system}.${name}) baselines;
  "Official developer environment, hooks and optional baseline outputs retained.\n"
'

work=$(mktemp -d)
nixpkgs_source=$(nix eval --raw "$GUFO_CHECK_FLAKE#lib.nixpkgsSource")
mkdir -p "$work/fixture" /nix/var/nix/gcroots
cat > "$work/fixture/flake.nix" <<EOF
{
  inputs.nixpkgs.url = "path:$nixpkgs_source";
  outputs = { self, nixpkgs }: let
    pkgs = import nixpkgs { system = "x86_64-linux"; };
  in {
    devShells.x86_64-linux.default = pkgs.mkShellNoCC {
      packages = [ pkgs.bashInteractive ];
    };
  };
}
EOF
nix flake lock "path:$work/fixture"
nix develop --no-update-lock-file --max-jobs 1 --cores 2 \
  --profile "$work/environment" "path:$work/fixture" --command true
nix registry add --registry /etc/nix/registry.json nixpkgs "path:$nixpkgs_source"

# Exercise the default output set used by Nix develop, including Bash's man
# output. Executable-only closure retention caused the reported fallback.
interactive_outputs=$(nix build --no-update-lock-file --max-jobs 0 \
  --no-link --print-out-paths nixpkgs#bashInteractive)
[[ -n "$interactive_outputs" ]]
index=0
while IFS= read -r output; do
  [[ "$output" == /nix/store/* && -d "$output" ]]
  ln -s "$output" "/nix/var/nix/gcroots/ci-interactive-bash-$index"
  index=$((index + 1))
done <<< "$interactive_outputs"
ln -s "$nixpkgs_source" /nix/var/nix/gcroots/ci-shell-nixpkgs
ln -s "$(readlink -f "$work/environment")" /nix/var/nix/gcroots/ci-shell-profile
nix-store --gc --option keep-derivations false --option keep-outputs false

offline=(--offline --max-jobs 0 --option substitute false)
nix build "${offline[@]}" --no-link nixpkgs#bashInteractive
if ! nix develop "${offline[@]}" "$work/environment" \
  --command bash -c 'printf "Offline shell closure verified.\n"' \
  > "$work/stdout" 2> "$work/stderr"; then
  cat "$work/stderr" >&2
  exit 1
fi
if grep -Eq 'error \(ignored\)|derivations? will be built|building .*/nix/store/' "$work/stderr"; then
  cat "$work/stderr" >&2
  exit 1
fi
cat "$work/stdout" "$work/stderr"
