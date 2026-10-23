"""Exercise wrapper routing and daemon startup without modifying /etc or /nix."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / 'toolboxes/gufo-nix/bin'


class NixRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runtime = self.root / 'run'
        self.real = self.root / 'real-nix'
        self.daemon = self.root / 'real-daemon'
        self.calls = self.root / 'calls.jsonl'
        self.daemon_env = self.root / 'daemon-env.json'
        self.ready = self.root / 'ready'
        self.pid = self.root / 'daemon.pid'
        self.env = {**os.environ, 'NIX_CONFIG': 'trusted-users = *',
                    'NIX_USER_CONF_FILES': '/tmp/untrusted-nix.conf',
                    'NIX_REMOTE': 'attacker', 'LD_LIBRARY_PATH': '/tmp/untrusted-libs'}
        self.write(self.real, f'''#!/usr/bin/python3
import json, os, pathlib, sys
if sys.argv[1:3] == ['store', 'ping']:
    sys.exit(0 if pathlib.Path({str(self.ready)!r}).exists() else 1)
with open({str(self.calls)!r}, 'a') as f:
    f.write(json.dumps({{'args':sys.argv[1:], 'remote':os.environ.get('NIX_REMOTE')}})+'\\n')
''')
        self.write(self.daemon, f'''#!/usr/bin/python3
import json, os, pathlib, time
pathlib.Path({str(self.daemon_env)!r}).write_text(json.dumps(dict(os.environ)))
pathlib.Path({str(self.pid)!r}).write_text(str(os.getpid()))
pathlib.Path({str(self.ready)!r}).write_text('ready')
time.sleep(30)
''')
        # Production has no environment overrides for privileged executable
        # paths. Rewrite fixed paths only in isolated copies for this fixture.
        helper = (BIN / 'gufo-nix-start-daemon').read_text()
        helper = helper.replace('/opt/nix/bin/nix-daemon', str(self.daemon))
        helper = helper.replace('/opt/nix/bin/nix', str(self.real))
        helper = helper.replace('/run/gufo-nix', str(self.runtime))
        helper = helper.replace('NIX_CONF_DIR=/etc/nix', f'NIX_CONF_DIR={self.root}/etc')
        self.helper = self.root / 'helper'
        self.write(self.helper, helper)

    def write(self, path, text):
        path.write_text(text)
        path.chmod(0o755)

    def tearDown(self):
        if self.pid.exists():
            try:
                os.kill(int(self.pid.read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.tmp.cleanup()

    def run_helper(self, *args):
        return subprocess.run([str(self.helper), *args], env=self.env,
                              capture_output=True, text=True, timeout=20)

    @unittest.skipUnless(os.geteuid() == 0, 'root-only helper; image smoke covers normal-user sudo')
    def test_concurrent_start_is_single_and_privileged_environment_is_clean(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.run_helper(), range(4)))
        for result in results:
            self.assertEqual(result.returncode, 0, result.stderr)
        environment = json.loads(self.daemon_env.read_text())
        for name in ('NIX_CONFIG', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'NIX_REMOTE'):
            self.assertNotIn(name, environment)
        self.assertEqual(environment['NIX_USER_CONF_FILES'], '/dev/null')
        self.assertEqual(environment['HOME'], '/var/empty')
        self.assertEqual(environment['NIX_CONF_DIR'], str(self.root / 'etc'))
        self.assertFalse((self.runtime / 'daemon.previous.log').exists())
        # A subsequent client must not restart an already reachable daemon.
        before = self.pid.read_text()
        self.assertEqual(self.run_helper().returncode, 0)
        self.assertEqual(self.pid.read_text(), before)

    def test_helper_rejects_arguments_before_startup(self):
        result = self.run_helper('--anything')
        self.assertEqual(result.returncode, 2)
        self.assertIn('no arguments', result.stderr)
        self.assertFalse(self.runtime.exists())

    @unittest.skipUnless(os.geteuid() == 0, 'root-only helper')
    def test_failed_daemon_reports_log_and_fails(self):
        self.write(self.daemon, '#!/bin/bash\necho startup-failed >&2\nexit 7\n')
        result = self.run_helper()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('startup-failed', result.stderr)

    def wrapper(self, user_mode=False):
        text = (BIN / 'nix').read_text().replace('/opt/nix/bin/nix', str(self.real))
        helper_stub = self.root / 'helper-stub'
        self.write(helper_stub, '#!/bin/bash\nexit 0\n')
        text = text.replace('/usr/local/sbin/gufo-nix-start-daemon', str(helper_stub))
        sudo = self.root / 'sudo'
        sudo_calls = self.root / 'sudo.json'
        self.write(sudo, f'''#!/usr/bin/python3
import json, pathlib, subprocess, sys
pathlib.Path({str(sudo_calls)!r}).write_text(json.dumps(sys.argv[1:]))
sys.exit(subprocess.call(sys.argv[3:]))
''')
        text = text.replace('/usr/bin/sudo', str(sudo))
        if user_mode:
            # EUID is a readonly shell value; replace only the privilege branch
            # to test CLI forwarding on hosts without a test sudo account.
            text = text.replace('if (( EUID == 0 )); then', 'if false; then')
        path = self.root / 'wrapper'
        self.write(path, text)
        return path, sudo_calls

    def test_nonroot_route_and_arguments_are_preserved(self):
        wrapper, sudo_calls = self.wrapper(user_mode=True)
        result = subprocess.run([str(wrapper), 'build', 'path with spaces#default'],
                                env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(sudo_calls.read_text())[:2], ['-n', '--'])
        self.assertEqual(json.loads(self.calls.read_text()),
                         {'args': ['build', 'path with spaces#default'], 'remote': 'daemon'})

    def test_information_command_does_not_start_daemon(self):
        wrapper, sudo_calls = self.wrapper(user_mode=True)
        result = subprocess.run([str(wrapper), '--version'], env=self.env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(sudo_calls.exists())
        self.assertEqual(json.loads(self.calls.read_text())['args'], ['--version'])

    def test_startup_failure_does_not_run_client_against_another_store(self):
        wrapper, _ = self.wrapper(user_mode=True)
        self.write(self.root / 'sudo', '#!/bin/bash\necho sudo-denied >&2\nexit 1\n')
        result = subprocess.run([str(wrapper), 'build', '.#default'], env=self.env,
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('sudo-denied', result.stderr)
        self.assertFalse(self.calls.exists())

    @unittest.skipUnless(os.geteuid() == 0, 'tests actual root CLI branch')
    def test_root_route_uses_local_store(self):
        wrapper, sudo_calls = self.wrapper()
        result = subprocess.run([str(wrapper), 'eval', '--expr', '1'], env=self.env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(sudo_calls.exists())
        self.assertEqual(json.loads(self.calls.read_text())['remote'], 'local')


if __name__ == '__main__':
    unittest.main()
