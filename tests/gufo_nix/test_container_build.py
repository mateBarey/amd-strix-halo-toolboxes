"""Regressions for Fedora's CA layout and OCI build-shell error handling.

No container engine, network, /nix writes, or image build is required. Build
script probes exit before any original side effect, while retaining the real
RUN command and script body. OCI builders may ignore Dockerfile SHELL, so an
implicit RUN shell is deliberately represented by the OCI /bin/sh default.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / 'toolboxes/Dockerfile.gufo-nix'
CANONICAL_CA = '/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem'
REMOVED_CA = '/etc/pki/tls/certs/ca-bundle.crt'


def run_heredocs():
    text = DOCKERFILE.read_text()
    pattern = re.compile(
        r"^RUN(?P<command>[^\n]*?)\s*<<'(?P<delimiter>\w+)'\s*\n"
        r"(?P<body>.*?)^(?P=delimiter)\s*$", re.MULTILINE | re.DOTALL)
    return [(match.group('command').strip(), match.group('body'))
            for match in pattern.finditer(text)]


def probe_build_script(command, body, failing_command):
    # Preserve leading comments/blank lines and any local `set` preamble. Do
    # not execute the original body: the unconditional exit protects the host
    # even when the regression causes the failure to be ignored.
    lines = body.splitlines(keepends=True)
    preamble_end = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or stripped.startswith('set '):
            preamble_end += 1
        else:
            break
    injection = failing_command + "\nprintf 'BUILD_CONTINUED\\n'\nexit 0\n"
    script = ''.join(lines[:preamble_end]) + injection + ''.join(lines[preamble_end:])
    # SHELL is intentionally ignored here, matching the reported Podman OCI
    # behavior. Only the executable on this individual RUN can select Bash.
    argv = shlex.split(command) if command else ['/bin/sh', '-s']
    environment = dict(os.environ)
    for name in ('BASH_ENV', 'ENV', 'SHELLOPTS', 'BASHOPTS'):
        environment.pop(name, None)
    return subprocess.run(argv, input=script, capture_output=True, text=True,
                          env=environment, timeout=5)


class ContainerBuildTests(unittest.TestCase):
    def test_all_nix_certificate_settings_use_fedora44_extracted_bundle(self):
        dockerfile = DOCKERFILE.read_text()
        config = (ROOT / 'toolboxes/gufo-nix/nix.conf').read_text()
        daemon = (ROOT / 'toolboxes/gufo-nix/bin/gufo-nix-start-daemon').read_text()
        # These paths are consumed by Nix/OpenSSL inside Fedora 44, where the
        # historical ca-bundle.crt compatibility path is no longer provided.
        settings = re.findall(r'NIX_SSL_CERT_FILE=([^\s)]+)', dockerfile + '\n' + daemon)
        self.assertEqual(len(settings), 3, 'builder, runtime, and clean daemon environment')
        for path in settings:
            self.assertEqual(path, CANONICAL_CA)
        self.assertEqual(re.findall(r'^ssl-cert-file\s*=\s*(\S+)\s*$', config, re.MULTILINE),
                         [CANONICAL_CA])
        active_lines = '\n'.join(line for line in (dockerfile + config + daemon).splitlines()
                                 if not line.lstrip().startswith('#'))
        self.assertNotIn(REMOVED_CA, active_lines)

    def test_bootstrap_checks_certificate_before_first_network_request(self):
        scripts = run_heredocs()
        self.assertEqual(len(scripts), 3)
        bootstrap = scripts[0][1]
        guard = re.search(r'^test -s [\"\']?' + re.escape(CANONICAL_CA) + r'[\"\']?\s*$',
                          bootstrap, re.MULTILINE)
        self.assertIsNotNone(guard, 'fail immediately when the installed CA bundle is missing')
        self.assertLess(guard.start(), bootstrap.index('curl --fail'))

    def test_each_actual_run_aborts_on_early_command_failure(self):
        scripts = run_heredocs()
        self.assertEqual(len(scripts), 3)
        for index, (command, body) in enumerate(scripts, 1):
            with self.subTest(run=index):
                result = probe_build_script(command, body, 'false')
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertNotIn('BUILD_CONTINUED', result.stdout)

    def test_each_actual_run_aborts_when_left_side_of_pipeline_fails(self):
        scripts = run_heredocs()
        self.assertEqual(len(scripts), 3)
        for index, (command, body) in enumerate(scripts, 1):
            with self.subTest(run=index):
                result = probe_build_script(command, body, 'false | true')
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertNotIn('BUILD_CONTINUED', result.stdout)


if __name__ == '__main__':
    unittest.main()
