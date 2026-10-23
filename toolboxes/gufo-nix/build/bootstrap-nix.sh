#!/bin/bash
test "$(uname -m)" = x86_64
test -s /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
install -d /etc/nix /nix/store /nix/var/nix/gcroots /opt
cat > /etc/nix/nix.conf <<'NIXCONF'
experimental-features = nix-command flakes
sandbox = false
build-users-group =
accept-flake-config = false
NIXCONF
archive="nix-${NIX_VERSION}-x86_64-linux.tar.xz"
curl --fail --location --retry 3 \
  "https://releases.nixos.org/nix/nix-${NIX_VERSION}/${archive}" \
  --output "/tmp/${archive}"
printf '%s  %s\n' "$NIX_SHA256" "/tmp/${archive}" | sha256sum --check --strict
tar -xJf "/tmp/${archive}" -C /tmp
bootstrap="/tmp/nix-${NIX_VERSION}-x86_64-linux"
cp -a "$bootstrap/store/." /nix/store/
nix_roots=(/nix/store/*-nix-"$NIX_VERSION")
test "${#nix_roots[@]}" -eq 1
ln -s "${nix_roots[0]}" /opt/nix
/opt/nix/bin/nix-store --load-db < "$bootstrap/.reginfo"
ln -s "${nix_roots[0]}" /nix/var/nix/gcroots/gufo-nix-client
/opt/nix/bin/nix --version
rm -rf "$bootstrap" "/tmp/${archive}"
