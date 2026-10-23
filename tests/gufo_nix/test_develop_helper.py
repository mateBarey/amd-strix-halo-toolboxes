"""Run the development helper's actual inner Bash using isolated command fixtures."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/gufo-nix/develop.sh'


class DevelopHelperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='gufo-develop-check-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.home = self.root / 'home with spaces'
        self.home.mkdir()
        self.repo = self.root / 'checkout with spaces'
        self.script = self.repo / 'scripts/gufo-nix/develop.sh'
        self.script.parent.mkdir(parents=True)
        self.flake = self.repo / 'toolboxes/gufo-nix'
        self.flake.mkdir(parents=True)
        self.env = os.environ.copy()
        self.env.update(PATH=str(self.bin) + os.pathsep + self.env['PATH'],
                        HOME=str(self.home), TEST_ROOT=str(self.root))
        for key in ('GUFO_TOOLBOX_NAME', 'GUFO_BUILD_CORES', 'GUFO_NIX_MAX_JOBS'):
            self.env.pop(key, None)
        installed = self.root / 'installed-gufo'
        installed.mkdir()
        (self.root / 'gufo-profile').symlink_to(installed)
        self.make('toolbox', '''import json, os, sys
from pathlib import Path
r = Path(os.environ['TEST_ROOT'])
a = sys.argv[1:]
(r/'toolbox.json').write_text(json.dumps(a))
assert a[0:2] == ['run', '-c'], a
assert a[3:11] == ['env', '-u', 'LD_LIBRARY_PATH', '-u', 'LD_PRELOAD', '/bin/bash', '-euo', 'pipefail'], a
assert a[11] == '-c', a
# Fixed production paths are translated only inside this test fixture.
a[12] = a[12].replace('/usr/local/bin/nix', str(r/'bin/fake-nix'))
a[12] = a[12].replace('/opt/gufo', str(r/'gufo-profile'))
os.execvp(a[3], a[3:])
''')
        self.make('fake-nix', '''import json, os, sys
from pathlib import Path
r = Path(os.environ['TEST_ROOT'])
a = sys.argv[1:]
with (r/'nix.jsonl').open('a') as f:
    f.write(json.dumps({'args': a, 'loader': {k: os.environ.get(k) for k in ('LD_LIBRARY_PATH', 'LD_PRELOAD')}}) + '\\n')
if a[0] == 'path-info':
    print(r / ('different-gufo' if os.environ.get('TEST_MISMATCH') else 'installed-gufo'))
elif a[0] == 'build' and os.environ.get('TEST_BASH_FAILURE'):
    sys.exit(23)
''')

    def make(self, name, body):
        path = self.bin / name
        path.write_text('#!/usr/bin/env python3\n' + body)
        path.chmod(0o755)

    def run_helper(self, *args):
        self.assertTrue(SCRIPT.is_file(), 'existing-container develop helper must exist')
        shutil.copyfile(SCRIPT, self.script)
        return subprocess.run(['bash', str(self.script), *args], env=self.env,
                              text=True, capture_output=True, timeout=10)

    def calls(self):
        log = self.root / 'nix.jsonl'
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    def test_command_arguments_paths_and_clean_loader_reach_official_environment(self):
        self.env.update(GUFO_TOOLBOX_NAME='gufo-rocm-nix-2', GUFO_BUILD_CORES='6',
                        GUFO_NIX_MAX_JOBS='2', LD_LIBRARY_PATH='/old/rocm/lib', LD_PRELOAD='libm.so.6')
        command = ['python3', '-c', 'print("$HOME; `false`")', 'argument with spaces', '']
        result = self.run_helper('--command', *command)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        self.assertEqual([c['args'][0] for c in calls], ['path-info', 'build', 'develop'])
        flake = 'path:' + str(self.flake)
        self.assertEqual(calls[0]['args'], ['path-info', '--offline', '--no-update-lock-file', flake + '#default'])
        cache = self.home / '.cache/gufo/nix-devshell/gufo-rocm-nix-2'
        self.assertEqual(calls[1]['args'], ['build', '--no-update-lock-file', '--max-jobs', '0',
                                          '--out-link', str(cache/'bash'), flake + '#bash-interactive'])
        self.assertEqual(calls[2]['args'], ['develop', '--no-update-lock-file', '--cores', '6',
                                          '--max-jobs', '2', '--profile', str(cache/'environment'),
                                          flake, '--command', *command])
        self.assertTrue(cache.is_dir())
        self.assertEqual(json.loads((self.root/'toolbox.json').read_text())[2], 'gufo-rocm-nix-2')
        for call in calls:
            self.assertEqual(call['loader'], {'LD_LIBRARY_PATH': None, 'LD_PRELOAD': None})

    def test_interactive_defaults_do_not_add_a_command(self):
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = self.calls()[-1]['args']
        self.assertEqual(args[args.index('--cores') + 1], '4')
        self.assertEqual(args[args.index('--max-jobs') + 1], '1')
        self.assertNotIn('--command', args)
        self.assertEqual(json.loads((self.root/'toolbox.json').read_text())[2], 'gufo-rocm-nix')

    def test_package_mismatch_stops_before_any_realization(self):
        self.env['TEST_MISMATCH'] = '1'
        result = self.run_helper()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('does not match', result.stderr)
        self.assertEqual([c['args'][0] for c in self.calls()], ['path-info'])

    def test_bash_dependency_failure_stops_before_develop(self):
        self.env['TEST_BASH_FAILURE'] = '1'
        result = self.run_helper()
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
        self.assertEqual([c['args'][0] for c in self.calls()], ['path-info', 'build'])

    def test_help_and_invalid_input_do_not_contact_toolbox(self):
        result = self.run_helper('--help')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('--command', result.stdout)
        for args in (('--command',), ('unexpected',)):
            result = self.run_helper(*args)
            self.assertNotEqual(result.returncode, 0)
        for key, value in (('GUFO_BUILD_CORES', '0'), ('GUFO_NIX_MAX_JOBS', 'bad'),
                           ('GUFO_TOOLBOX_NAME', '../other')):
            with self.subTest(key=key):
                self.env[key] = value
                result = self.run_helper()
                self.assertNotEqual(result.returncode, 0)
                del self.env[key]
        self.assertFalse((self.root/'toolbox.json').exists())

    def test_container_guard_stops_before_toolbox_execution(self):
        marker = self.root / 'toolboxenv'
        marker.touch()
        # Substitute the sentinel's location, then execute the actual guard.
        self.script.write_text(SCRIPT.read_text().replace('/run/.toolboxenv', str(marker)))
        result = subprocess.run(['bash', str(self.script)], env=self.env,
                                text=True, capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('HOST', result.stderr)
        self.assertFalse((self.root/'toolbox.json').exists())


if __name__ == '__main__':
    unittest.main()
