"""Execute the startup helper against isolated process and HTTP boundaries."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/gufo-nix/run-qwen27b.sh'


class RunQwenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='gufo-startup-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.repo = self.root / 'repo with spaces'
        self.scripts = self.repo / 'scripts/gufo-nix'
        self.scripts.mkdir(parents=True)
        self.script = self.scripts / SCRIPT.name
        self.marker = self.root / 'toolboxenv'
        self.env = dict(os.environ)
        for key in tuple(self.env):
            if key.startswith('GUFO_'):
                self.env.pop(key)
        self.env.update(PATH=str(self.bin) + os.pathsep + self.env['PATH'],
                        TEST_ROOT=str(self.root), HOME=str(self.root / 'home'),
                        GUFO_BIN=str(self.bin / 'gufo'), GUFO_STARTUP_TIMEOUT='3',
                        GUFO_MODEL_PATH=str(self.root / 'model with spaces.gguf'),
                        GUFO_VISION_PATH=str(self.root / 'projector with spaces.gguf'))
        for key in ('GUFO_MODEL_PATH', 'GUFO_VISION_PATH'):
            Path(self.env[key]).write_bytes(b'fixture')
        self.write(self.bin / 'toolbox', '''import json, os, sys
from pathlib import Path
r=Path(os.environ['TEST_ROOT']); a=sys.argv[1:]
with (r/'toolbox.jsonl').open('a') as f: f.write(json.dumps(a)+'\\n')
assert a[:3]==['run','-c','gufo-rocm-nix'], a
os.execvp(a[3],a[3:])
''')
        self.write(self.bin / 'gufo', '''import json, os, sys, time
from pathlib import Path
r=Path(os.environ['TEST_ROOT']); a=sys.argv[1:]
if a==['--version']:
 print('gufo version fixture'); sys.exit(0)
if a[:1]==['diagnose']:
 print('{"inventory":{}}'); sys.exit(0)
(r/'server.pid').write_text(str(os.getpid()))
(r/'server.json').write_text(json.dumps({'args':a,'loader':os.environ.get('LD_LIBRARY_PATH'),'cache':os.environ.get('GUFO_HIPBLASLT_PLAN_CACHE')}))
if os.environ.get('TEST_LOAD_FAILURE'):
 print('text runner request-state claim exceeds state capacity', flush=True)
 sys.exit(7)
(r/'ready').write_text('qwen38-27b-uncensored')
while True: time.sleep(0.05)
''')
        self.write(self.bin / 'projector-check', '''import os, sys
from pathlib import Path
r=Path(os.environ['TEST_ROOT']); (r/'projector-checked').touch()
sys.exit(9 if os.environ.get('TEST_BAD_PROJECTOR') else 0)
''')
        self.write(self.bin / 'curl', '''import json, os, sys
from pathlib import Path
assert '--noproxy' in sys.argv and sys.argv[sys.argv.index('--noproxy')+1]=='*'
p=Path(os.environ['TEST_ROOT'])/'ready'
if not p.exists(): sys.exit(7)
print(json.dumps({'data':[{'id':p.read_text()}]}))
''')
        self.write(self.scripts / 'check-server.py', '''import json, os, sys
from pathlib import Path
r=Path(os.environ['TEST_ROOT']); (r/'checked.json').write_text(json.dumps(sys.argv[1:]))
if os.environ.get('TEST_CHECK_FAILURE'):
 print('vision request failed',file=sys.stderr); sys.exit(11)
''')
        self.write(self.scripts / 'stop-previous-gufo.py', '''import json, os, sys
from pathlib import Path
r=Path(os.environ['TEST_ROOT']); (r/'stopped.json').write_text(json.dumps(sys.argv[1:]))
''')
        memory = {
            'hip': {'status': 'ok', 'devices': [{'index': 0, 'free_bytes': 1073741824,
                                               'total_bytes': 4294967296}]},
            'host_memory': {'MemAvailable': 123456},
            'gufo_processes': [{'pid': 123, 'rss_bytes': 987654321, 'model': '/models/previous.gguf'}],
            'kfd_clients': [{'pid': 123, 'accounted_vram_bytes_by_gpu': {'0': 777, '1': 888}}],
        }
        (self.scripts / 'inspect-memory.sh').write_text(
            "#!/bin/bash\nprintf '%s\\n' '" + json.dumps(memory) + "'\n")
        self.addCleanup(self.stop_server)

    def write(self, path, body):
        path.write_text('#!/usr/bin/env python3\n' + body)
        path.chmod(0o755)

    def stop_server(self):
        path = self.root / 'server.pid'
        if path.exists():
            try:
                os.kill(int(path.read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass

    def run_helper(self, *args, inside=False):
        self.assertTrue(SCRIPT.is_file(), 'complete startup helper must exist')
        body = SCRIPT.read_text().replace('/run/.toolboxenv', str(self.marker))
        body = body.replace('/run/.containerenv', str(self.root / 'containerenv'))
        body = body.replace('/usr/local/bin/gufo-convert-projector', str(self.bin / 'projector-check'))
        self.script.write_text(body)
        if inside:
            self.marker.touch()
        return subprocess.run(['bash', str(self.script), *args], env=self.env,
                              capture_output=True, text=True, timeout=15)

    def test_inside_starts_directly_with_full_context_and_runs_checks(self):
        self.env['LD_LIBRARY_PATH'] = '/wrong/rocm/lib'
        self.env['GUFO_HIPBLASLT_PLAN_CACHE'] = '/unexpected/cache'
        self.env['GUFO_TEST_IMAGE'] = str(self.root / 'actual image.png')
        result = self.run_helper(inside=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((self.root / 'toolbox.jsonl').exists())
        call = json.loads((self.root / 'server.json').read_text())
        for flag, value in (('--port', '8081'), ('--context', '262144'),
                            ('--sessions', '1'), ('--speculative', 'off'),
                            ('--think', 'off'), ('--cache-ram-bytes', '4294967296'),
                            ('--model', self.env['GUFO_MODEL_PATH']),
                            ('--mmproj', self.env['GUFO_VISION_PATH'])):
            self.assertEqual(call['args'][call['args'].index(flag) + 1], value)
        self.assertIsNone(call['loader'])
        self.assertIsNone(call['cache'])
        checked = json.loads((self.root / 'checked.json').read_text())
        self.assertEqual(checked[checked.index('--base-url') + 1], 'http://127.0.0.1:8081/v1')
        self.assertEqual(checked[checked.index('--image') + 1], self.env['GUFO_TEST_IMAGE'])
        self.assertFalse((self.root / 'stopped.json').exists())
        self.assertIn('Hermes', result.stdout)

    def test_host_routes_to_existing_container(self):
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [json.loads(line) for line in (self.root / 'toolbox.jsonl').read_text().splitlines()]
        self.assertTrue(calls)
        self.assertTrue(all(a[:3] == ['run', '-c', 'gufo-rocm-nix'] for a in calls))
        self.assertNotIn('create', [word for a in calls for word in a])
        self.assertTrue((self.root / 'checked.json').exists())

    def test_wrong_model_on_port_is_not_reused_or_stopped(self):
        (self.root / 'ready').write_text('different-model')
        result = self.run_helper('--replace-flash-next', inside=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / 'server.json').exists())
        self.assertFalse((self.root / 'checked.json').exists())
        self.assertFalse((self.root / 'stopped.json').exists())

    def test_matching_model_is_checked_without_duplicate_start(self):
        (self.root / 'ready').write_text('qwen38-27b-uncensored')
        result = self.run_helper('--replace-flash-next', inside=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((self.root / 'server.json').exists())
        self.assertTrue((self.root / 'checked.json').exists())
        self.assertFalse((self.root / 'stopped.json').exists())

    def test_non_api_port_conflict_prevents_replacement(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            self.env['GUFO_PORT'] = str(listener.getsockname()[1])
            result = self.run_helper('--replace-flash-next', inside=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Cannot use port', result.stderr)
        self.assertFalse((self.root / 'stopped.json').exists())
        self.assertFalse((self.root / 'server.json').exists())

    def test_load_error_preserves_failure_and_does_not_claim_success(self):
        self.env['TEST_LOAD_FAILURE'] = '1'
        result = self.run_helper(inside=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('request-state claim exceeds state capacity', result.stderr)
        self.assertIn('HIP GPU 0: free=1073741824 bytes (1.00 GiB); total=4294967296 bytes', result.stderr)
        self.assertIn('Host MemAvailable: 123456 bytes', result.stderr)
        self.assertIn('Gufo PID 123: RSS=987654321 bytes', result.stderr)
        self.assertIn('model="/models/previous.gguf"', result.stderr)
        self.assertIn('KFD PID 123: accounted VRAM counters total=1665 bytes', result.stderr)
        self.assertFalse((self.root / 'checked.json').exists())
        self.assertNotIn('Text and vision checks passed', result.stdout)
        reports = list((self.root / 'home/.cache/gufo/qwen27b').glob('run-*/memory-failure.json'))
        self.assertEqual(len(reports), 1)

    def test_invalid_memory_report_keeps_original_failure_visible(self):
        self.env['TEST_LOAD_FAILURE'] = '1'
        (self.scripts / 'inspect-memory.sh').write_text("#!/bin/bash\nprintf 'not JSON\\n'\nexit 1\n")
        result = self.run_helper(inside=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('request-state claim exceeds state capacity', result.stderr)
        self.assertIn('Memory summary unavailable:', result.stderr)
        self.assertNotIn('Traceback', result.stderr)

    def test_ready_catalog_does_not_hide_inference_check_failure(self):
        self.env['TEST_CHECK_FAILURE'] = '1'
        result = self.run_helper(inside=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('vision request failed', result.stderr)
        self.assertNotIn('Text and vision checks passed', result.stdout)

    def test_previous_flash_next_stop_requires_explicit_flag(self):
        result = self.run_helper('--replace-flash-next', inside=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads((self.root / 'stopped.json').read_text()),
                         ['--model', '/home/esckobear/VMl/Qwen3.8-Flash-Next-Uncensored-Gufo-Q4Mix.gguf'])

    def test_bad_projector_and_help_do_not_start(self):
        result = self.run_helper('--help')
        self.assertEqual(result.returncode, 0)
        self.assertIn('--replace-flash-next', result.stdout)
        self.assertFalse((self.root / 'toolbox.jsonl').exists())
        self.env['TEST_BAD_PROJECTOR'] = '1'
        result = self.run_helper(inside=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / 'server.json').exists())
        self.assertFalse((self.root / 'checked.json').exists())


if __name__ == '__main__':
    unittest.main()
