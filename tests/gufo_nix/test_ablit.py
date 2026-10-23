"""Exercise the public launcher without Toolbx, HTTP, or real Gufo processes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / 'run_qwen_ablit.sh'


class AblitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='gufo-ablit-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.home = self.root / 'home with spaces'
        self.home.mkdir()
        self.events_path = self.root / 'events.jsonl'
        self.markers = {name: self.root / name for name in ('toolboxenv', 'containerenv')}
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(('GUFO_', 'TEST_'))}
        # A private PATH makes the missing-Hermes check deterministic, even on
        # a developer machine that has Hermes installed elsewhere.
        self.env.update(PATH=str(self.bin), HOME=str(self.home),
                        TEST_ROOT=str(self.root),
                        TEST_BOUNDARY=str(self.root / 'boundary.py'))
        for command in ('bash', 'dirname', 'basename', 'cat', 'readlink', 'realpath', 'env'):
            executable = shutil.which(command)
            if executable:
                (self.bin / command).symlink_to(executable)
        (self.bin / 'python3').symlink_to(sys.executable)
        (self.root / 'boundary.py').write_text('''import json, os, sys
from pathlib import Path
action, helper, *args = sys.argv[1:]
root = Path(os.environ['TEST_ROOT'])
record = {'action': action, 'helper': helper, 'args': args, 'pid': os.getpid(),
          'env': {key: value for key, value in os.environ.items() if key.startswith('GUFO_')}}
with (root / 'events.jsonl').open('a') as stream:
    stream.write(json.dumps(record) + '\\n')
if action == 'start':
    print('startup fixture stdout')
    print('startup fixture stderr', file=sys.stderr)
    sys.exit(int(os.environ.get('TEST_START_EXIT', '0')))
if action == 'hermes':
    print('Hermes fixture session')
    sys.exit(int(os.environ.get('TEST_HERMES_EXIT', '0')))
if action == 'process':
    model = args[args.index('--model') + 1]
    code = int(os.environ.get('TEST_PROCESS_EXIT', '0'))
    if code:
        print('process fixture diagnostic', file=sys.stderr)
        print(json.dumps({'model': model, 'error': 'fixture failure', 'stopped_pids': []}))
        sys.exit(code)
    key = 'running_pids' if '--status' in args else 'stopped_pids'
    print(json.dumps({'model': model, key: json.loads(os.environ.get('TEST_PIDS', '[2147483646]'))}))
    sys.exit(0)
if action == 'curl':
    code = int(os.environ.get('TEST_CURL_EXIT', '0'))
    if code:
        sys.exit(code)
    print(json.dumps({'object': 'list', 'data': [
        {'id': os.environ.get('TEST_API_MODEL', 'qwen38-27b-uncensored'),
         'object': 'model', 'owned_by': 'gufo'}]}))
    sys.exit(0)
print('unexpected external action: ' + action, file=sys.stderr)
sys.exit(99)
''')
        for command in ('hermes', 'toolbox', 'podman', 'gufo', 'pkill', 'killall', 'curl'):
            self.shell_boundary(self.bin / command, command)
        # Intercept Bash's builtin kill too, so a regression cannot signal any
        # real process while testing the read-only status branch.
        guard = self.root / 'shell-guard.sh'
        guard.write_text('kill() { python3 "$TEST_BOUNDARY" signal "$0" "$@"; }\n')
        self.env['BASH_ENV'] = str(guard)
        self.repo = self.make_repo(self.root / 'repo with spaces')

    def shell_boundary(self, path, action):
        path.write_text('#!/usr/bin/env bash\n'
                        f'exec python3 "$TEST_BOUNDARY" {action} "$0" "$@"\n')
        path.chmod(0o755)

    def make_repo(self, repo):
        scripts = repo / 'scripts/gufo-nix'
        scripts.mkdir(parents=True)
        self.shell_boundary(scripts / 'run-qwen27b.sh', 'start')
        self.shell_boundary(scripts / 'launch-hermes-qwen27b.sh', 'hermes')
        (scripts / 'stop-previous-gufo.py').write_text('''import os, runpy, sys
sys.argv = [os.environ['TEST_BOUNDARY'], 'process', __file__, *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name='__main__')
''')
        return repo

    def run_script(self, *args, location=None, inside=None):
        self.assertTrue(SCRIPT.is_file(), 'the public run_qwen_ablit.sh entry point must exist')
        for name, marker in self.markers.items():
            marker.unlink(missing_ok=True)
            if name == inside:
                marker.touch()
        destination = location or self.repo / SCRIPT.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        body = SCRIPT.read_text()
        for name, marker in self.markers.items():
            body = body.replace('/run/.' + name, str(marker))
        destination.write_text(body)
        self.events_path.unlink(missing_ok=True)
        command = [str(self.bin / 'bash'), str(destination), *args]
        process = subprocess.Popen(command, cwd=self.root, env=self.env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.last_pid = process.pid
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

    def events(self):
        if not self.events_path.exists():
            return []
        return [json.loads(line) for line in self.events_path.read_text().splitlines()]

    def test_default_and_start_export_defaults_then_exec_hermes(self):
        self.env.update(TEST_HERMES_EXIT='23', GUFO_PRINT_HERMES_COMMAND='1')
        expected = {
            'GUFO_PORT': '8081', 'GUFO_CONTEXT': '262144',
            'GUFO_TOOLBOX_NAME': 'gufo-rocm-nix',
            'GUFO_MODEL_PATH': str(self.home / 'VMl/Qwen3.8-27B-Uncensored-Q4_K_M.gguf'),
            'GUFO_VISION_PATH': str(self.home / 'VMl/mmproj-Qwen3.8-27B-Uncensored-BF16.gguf'),
        }
        for args in ((), ('start',)):
            with self.subTest(args=args):
                result = self.run_script(*args)
                self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
                calls = self.events()
                self.assertEqual([call['action'] for call in calls], ['start', 'hermes'])
                for call in calls:
                    self.assertEqual(call['args'], [])
                    self.assertEqual({key: call['env'].get(key) for key in expected}, expected)
                self.assertEqual(calls[0]['env']['GUFO_PRINT_HERMES_COMMAND'], '0')
                self.assertNotEqual(calls[0]['env'].get('GUFO_REQUIRE_RUNNING'), '1')
                self.assertEqual(calls[1]['env'].get('GUFO_REQUIRE_RUNNING'), '1')
                self.assertEqual(calls[1]['pid'], self.last_pid, 'Hermes must replace the wrapper')
                self.assertIn('Hermes fixture session', result.stdout)

    def test_explicit_repository_and_custom_settings_preserve_spaces(self):
        alternate = self.make_repo(self.root / 'alternate repo with spaces')
        overrides = {
            'GUFO_REPO': str(alternate), 'GUFO_PORT': '18091', 'GUFO_CONTEXT': '65536',
            'GUFO_TOOLBOX_NAME': 'my-gufo',
            'GUFO_MODEL_PATH': str(self.root / 'models/custom model.gguf'),
            'GUFO_VISION_PATH': str(self.root / 'models/custom projector BF16.gguf'),
            'GUFO_PROJECT_ROOT': str(self.root / 'project with spaces'),
        }
        self.env.update(overrides)
        result = self.run_script('start')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.events()
        self.assertEqual([call['action'] for call in calls], ['start', 'hermes'])
        for call, filename in zip(calls, ('run-qwen27b.sh', 'launch-hermes-qwen27b.sh')):
            self.assertEqual(Path(call['helper']), alternate / 'scripts/gufo-nix' / filename)
            self.assertEqual({key: call['env'].get(key) for key in overrides}, overrides)

    def test_copied_vml_script_finds_the_existing_checkout(self):
        fallback = self.make_repo(self.home / 'VMl/gufo-nix-mateBarey-github')
        result = self.run_script(location=self.home / 'VMl/run_qwen_ablit.sh')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.events()
        self.assertEqual([call['action'] for call in calls], ['start', 'hermes'])
        self.assertTrue(all(Path(call['helper']).parent == fallback / 'scripts/gufo-nix'
                            for call in calls))

    def test_container_and_missing_hermes_fail_before_startup(self):
        for marker in self.markers:
            with self.subTest(marker=marker):
                result = self.run_script(inside=marker)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('host', result.stderr.lower())
                self.assertEqual(self.events(), [])
        (self.bin / 'hermes').unlink()
        result = self.run_script('start')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('hermes', result.stderr.lower())
        self.assertEqual(self.events(), [])

    def test_startup_failure_preserves_output_and_never_runs_hermes(self):
        self.env['TEST_START_EXIT'] = '37'
        result = self.run_script()
        self.assertEqual(result.returncode, 37, result.stdout + result.stderr)
        self.assertIn('startup fixture stdout', result.stdout)
        self.assertIn('startup fixture stderr', result.stderr)
        self.assertEqual([call['action'] for call in self.events()], ['start'])

    def test_stop_calls_the_exact_model_helper_directly_on_host_or_inside(self):
        (self.bin / 'hermes').unlink()
        model = str(self.root / 'models/selected model.gguf')
        self.env['GUFO_MODEL_PATH'] = model
        for inside, pids, code in ((None, [2147483646], 0), ('toolboxenv', [], 0),
                                   ('containerenv', [2147483645], 0), (None, [], 29)):
            with self.subTest(inside=inside, pids=pids, code=code):
                self.env.update(TEST_PIDS=json.dumps(pids), TEST_PROCESS_EXIT=str(code))
                result = self.run_script('stop', inside=inside)
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                calls = self.events()
                self.assertEqual([call['action'] for call in calls], ['process'])
                self.assertEqual(calls[0]['args'], ['--model', model])
                for pid in pids:
                    self.assertIn(str(pid), result.stdout)
                if code:
                    self.assertIn('process fixture diagnostic', result.stderr)

    def test_status_is_observational_even_when_api_is_wrong_or_unavailable(self):
        (self.bin / 'hermes').unlink()
        model = str(self.root / 'models/observed model.gguf')
        self.env.update(GUFO_MODEL_PATH=model, GUFO_PORT='18092')
        scenarios = (('qwen38-27b-uncensored', '0', '0'),
                     ('different-model', '0', '0'),
                     ('qwen38-27b-uncensored', '7', '0'),
                     ('qwen38-27b-uncensored', '0', '31'))
        for alias, curl_exit, helper_exit in scenarios:
            with self.subTest(alias=alias, curl_exit=curl_exit, helper_exit=helper_exit):
                self.env.update(TEST_API_MODEL=alias, TEST_CURL_EXIT=curl_exit,
                                TEST_PROCESS_EXIT=helper_exit)
                result = self.run_script('status', inside='toolboxenv')
                self.assertEqual(result.returncode, int(helper_exit), result.stdout + result.stderr)
                calls = self.events()
                self.assertTrue(calls)
                self.assertEqual(calls[0]['action'], 'process')
                self.assertCountEqual(calls[0]['args'], ['--status', '--model', model])
                self.assertEqual(calls[0]['args'][calls[0]['args'].index('--model') + 1], model)
                self.assertEqual(sum(call['action'] == 'process' for call in calls), 1)
                self.assertTrue(all(call['action'] in ('process', 'curl') for call in calls))
                for call in calls[1:]:
                    urls = [arg for arg in call['args'] if arg.startswith(('http:', 'https:'))]
                    self.assertEqual(urls, ['http://127.0.0.1:18092/v1/models'])
                if helper_exit == '0':
                    self.assertIn('2147483646', result.stdout)
                    if any(call['action'] == 'curl' for call in calls):
                        if curl_exit != '0':
                            self.assertIn('API not ready:', result.stdout)
                        elif alias != 'qwen38-27b-uncensored':
                            self.assertIn('does not advertise', result.stdout)
                            self.assertNotIn('API ready:', result.stdout)
                        else:
                            self.assertIn('API ready:', result.stdout)
                else:
                    self.assertIn('process fixture diagnostic', result.stderr)

    def test_help_and_bad_arguments_never_execute_helpers(self):
        (self.bin / 'hermes').unlink()
        for args, success in ((('--help',), True), (('-h',), True),
                              (('restart',), False), (('start', 'extra'), False),
                              (('stop', 'extra'), False)):
            with self.subTest(args=args):
                result = self.run_script(*args, inside='containerenv')
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    for action in ('start', 'stop', 'status'):
                        self.assertIn(action, result.stdout.lower())
                else:
                    self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.events(), [])


if __name__ == '__main__':
    unittest.main()
