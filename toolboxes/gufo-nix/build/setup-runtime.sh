#!/bin/bash
test -s /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
# Fixed, non-login build accounts belong to the container, independent of the
# UID Toolbx creates for its user. /nix/store is never made user-writable.
groupadd --system --gid 30000 nixbld
for i in {1..16}; do
  useradd --system --uid "$((30000 + i))" --gid nixbld --groups nixbld \
    --no-create-home --home-dir /var/empty --shell /usr/sbin/nologin "nixbld$i"
done
install -d -m 0755 /var/empty /usr/local/bin /usr/local/sbin /etc/sudoers.d
chown root:nixbld /nix/store
chmod 1775 /nix/store
chmod 0755 /nix /nix/var /nix/var/nix
install -m 0755 /tmp/gufo-bin/gufo-nix-start-daemon /usr/local/sbin/gufo-nix-start-daemon
for program in nix gufo-profile gufo-tune-hipblaslt gufo-convert-projector; do
  install -m 0755 "/tmp/gufo-bin/$program" "/usr/local/bin/$program"
done
for program in gufo gufo-server gufo-kernel-bench tune_hipblaslt; do
  ln -s "/opt/gufo/bin/$program" "/usr/local/bin/$program"
done
cat > /etc/sudoers.d/gufo-nix-daemon <<'SUDOERS'
ALL ALL=(root) NOPASSWD: /usr/local/sbin/gufo-nix-start-daemon ""
SUDOERS
chmod 0440 /etc/sudoers.d/gufo-nix-daemon
visudo --check --file /etc/sudoers.d/gufo-nix-daemon
python3 -m venv /opt/gufo-tools/venv
/opt/gufo-tools/venv/bin/python3 -m pip install --no-cache-dir 'gguf==0.19.0'
chmod -R a+rX /opt/gufo-tools
rm -rf /tmp/gufo-bin
/usr/local/bin/gufo --version
/usr/local/bin/gufo-convert-projector --help >/dev/null
/usr/local/bin/gufo-profile --help >/dev/null
/usr/local/bin/nix --version
