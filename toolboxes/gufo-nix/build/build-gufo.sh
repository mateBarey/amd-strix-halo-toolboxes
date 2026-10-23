#!/bin/bash
test -s "$NIX_SSL_CERT_FILE"
[[ "$NIX_BUILD_CORES" =~ ^[1-9][0-9]*$ ]]
[[ "$NIX_MAX_JOBS" =~ ^[1-9][0-9]*$ ]]
flake=path:/opt/gufo-toolbox/flake
/opt/nix/bin/nix build -L --no-update-lock-file \
  --cores "$NIX_BUILD_CORES" --max-jobs "$NIX_MAX_JOBS" \
  --out-link /opt/gufo "$flake#default"
/opt/nix/bin/nix develop --no-update-lock-file \
  --cores "$NIX_BUILD_CORES" --max-jobs "$NIX_MAX_JOBS" \
  --profile /opt/gufo-shell "$flake" --command true
ln -s "$(readlink -f /opt/gufo)" /nix/var/nix/gcroots/gufo-runtime
ln -s "$(readlink -f /opt/gufo-shell)" /nix/var/nix/gcroots/gufo-tools-shell
for input in gufoSource nixpkgsSource; do
  source_path=$(/opt/nix/bin/nix eval --raw "$flake#lib.$input")
  ln -s "$source_path" "/nix/var/nix/gcroots/$input"
done
# A profile-path `nix develop` resolves its interactive Bash via the registry.
# Pin that lookup to the already-rooted source rather than a moving channel.
nixpkgs_source=$(/opt/nix/bin/nix eval --raw "$flake#lib.nixpkgsSource")
/opt/nix/bin/nix registry add --registry /etc/nix/registry.json \
  nixpkgs "path:$nixpkgs_source"
# Nix develop independently realizes this installable, whose default outputs
# include Bash's manual output as well as the executable. A shell environment
# retaining only the executable is insufficient after GC. Capture status before
# processing the paths; process substitution would hide a failed Nix build.
interactive_outputs=$(/opt/nix/bin/nix build --no-update-lock-file \
  --cores "$NIX_BUILD_CORES" --max-jobs "$NIX_MAX_JOBS" \
  --no-link --print-out-paths nixpkgs#bashInteractive)
test -n "$interactive_outputs"
interactive_index=0
while IFS= read -r output; do
  [[ "$output" == /nix/store/* && -d "$output" ]]
  ln -s "$output" "/nix/var/nix/gcroots/gufo-interactive-bash-$interactive_index"
  interactive_index=$((interactive_index + 1))
done <<< "$interactive_outputs"
/opt/nix/bin/nix eval --json "$flake#lib" > /opt/gufo-toolbox/build-inputs.json
# Retain only rooted runtime, tools shell, Nix and pinned source closures.
/opt/nix/bin/nix-store --gc --option keep-derivations false --option keep-outputs false
