import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/gufo-nix/launch-hermes-qwen27b.sh"

class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), "the adapted launcher must exist")
        self.tmp = tempfile.TemporaryDirectory(prefix="gufo-launch-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        bins = self.root / "bin"
        bins.mkdir()
        self.env = os.environ.copy()
        self.env.update({
            "PATH": str(bins) + os.pathsep + self.env["PATH"],
            "TEST_ROOT": str(self.root),
            "GUFO_MODEL_PATH": str(self.root / "model with spaces.gguf"),
            "GUFO_VISION_PATH": str(self.root / "projector with spaces.gguf"),
            "GUFO_BIN": str(bins / "fake-gufo"),
            "GUFO_PROJECT_ROOT": str(self.root / "project with spaces"),
            "GUFO_LOG_FILE": str(self.root / "server.log"),
            "GUFO_TUNE_CACHE": str(self.root / "plans.bin"),
            "GUFO_STARTUP_TIMEOUT": "5",
        })
        for key in ("GUFO_MODEL_PATH", "GUFO_VISION_PATH", "GUFO_TUNE_CACHE"):
            Path(self.env[key]).touch()
        Path(self.env["GUFO_TUNE_CACHE"]).write_bytes(b"synthetic-plan-cache")
        self.make(bins / "toolbox", """import os,sys,json
from pathlib import Path
p=Path(os.environ['TEST_ROOT'])/'calls.jsonl'
with p.open('a') as f: f.write(json.dumps(['toolbox', *sys.argv[1:]])+chr(10))
a=sys.argv[1:]
assert a[:3] == ['run','-c','gufo-rocm-nix'], a
os.execvp(a[3],a[3:])
""")
        self.make(bins / "fake-gufo", """import os,sys,json,time
from pathlib import Path
r=Path(os.environ['TEST_ROOT'])
(r/'gpu.pid').write_text(str(os.getpid()))
(r/'gpu.json').write_text(json.dumps({'argv':sys.argv[1:],'cache':os.getenv('GUFO_HIPBLASLT_PLAN_CACHE'),'ld_library_path':os.getenv('LD_LIBRARY_PATH'),'ld_preload':os.getenv('LD_PRELOAD')}))
if os.environ.get('TEST_SCENARIO') == 'load-failure':
 print('invalid BF16 Qwen vision tensor: test', flush=True)
 sys.exit(7)
(r/'ready').write_text('qwen38-27b-uncensored')
while True: time.sleep(0.05)
""")
        self.make(bins / "curl", """import json,os,sys
from pathlib import Path
r=Path(os.environ['TEST_ROOT'])
if not (r/'ready').exists(): sys.exit(7)
print(json.dumps({'data':[{'id':(r/'ready').read_text()}]}))
""")
        self.make(bins / "hermes", """import json,os,sys
from pathlib import Path
with (Path(os.environ['TEST_ROOT'])/'calls.jsonl').open('a') as f:
 f.write(json.dumps(['hermes',*sys.argv[1:]])+chr(10))
""")
        self.addCleanup(self.stop_gpu)

    def make(self, path, text):
        path.write_text('#!/usr/bin/env python3\n' + text)
        path.chmod(0o755)

    def stop_gpu(self):
        p = self.root / 'gpu.pid'
        if p.exists():
            try: os.kill(int(p.read_text()), signal.SIGTERM)
            except ProcessLookupError: pass

    def run_script(self):
        return subprocess.run(['bash', str(SCRIPT)], env=self.env, text=True,
                              capture_output=True, timeout=15)

    def calls(self):
        p = self.root / 'calls.jsonl'
        return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []

    def test_start_passes_paths_cache_and_consistent_hermes_context(self):
        r = self.run_script()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        g = json.loads((self.root/'gpu.json').read_text())
        self.assertEqual(g['cache'], self.env['GUFO_TUNE_CACHE'])
        a = g['argv']
        for key, value in [('--model',self.env['GUFO_MODEL_PATH']),('--mmproj',self.env['GUFO_VISION_PATH']),('--context','262144'),('--speculative','off')]:
            self.assertEqual(a[a.index(key)+1], value)
        self.assertEqual(a[a.index('--think')+1], 'off')
        self.assertEqual(a[a.index('--served-model-name')+1], 'qwen38-27b-uncensored')
        self.assertNotIn('--mtp-model', a)
        self.assertIn(['hermes','config','set','model_overrides.custom.qwen38-27b-uncensored.context_window','262144'], self.calls())
        self.assertIn(['hermes','config','set','model_overrides.custom.qwen38-27b-uncensored.supports_vision','true'], self.calls())
        self.assertIn(['hermes'], self.calls())

    def test_existing_wrong_model_is_not_reused_or_killed(self):
        (self.root/'ready').write_text('qwen3.8-flash-next')
        r = self.run_script()
        self.assertNotEqual(r.returncode, 0, r.stdout+r.stderr)
        self.assertFalse((self.root/'gpu.json').exists())
        self.assertFalse(any(c[0]=='hermes' for c in self.calls()))

    def test_existing_matching_model_is_reused(self):
        (self.root/'ready').write_text('qwen38-27b-uncensored')
        r = self.run_script()
        self.assertEqual(r.returncode, 0, r.stdout+r.stderr)
        self.assertFalse((self.root/'gpu.json').exists())
        self.assertIn(['hermes'], self.calls())

    def test_require_running_never_starts_gufo_and_only_opens_hermes_for_expected_model(self):
        self.env['GUFO_REQUIRE_RUNNING'] = '1'
        for model, expected_code in (('qwen38-27b-uncensored', 0),
                                     ('qwen3.8-flash-next', 1), (None, 1)):
            with self.subTest(model=model):
                (self.root/'ready').unlink(missing_ok=True)
                (self.root/'calls.jsonl').unlink(missing_ok=True)
                if model is not None:
                    (self.root/'ready').write_text(model)
                r = self.run_script()
                self.assertEqual(r.returncode, expected_code, r.stdout+r.stderr)
                self.assertFalse((self.root/'gpu.json').exists())
                self.assertFalse(any(c[0] == 'toolbox' for c in self.calls()))
                if expected_code == 0:
                    self.assertIn(['hermes'], self.calls())
                else:
                    self.assertEqual(self.calls(), [])
                    if model is None:
                        self.assertIn('GUFO_REQUIRE_RUNNING=1', r.stderr)

    def test_load_failure_is_reported_before_hermes_configuration(self):
        self.env['TEST_SCENARIO'] = 'load-failure'
        r = self.run_script()
        self.assertNotEqual(r.returncode, 0, r.stdout+r.stderr)
        self.assertIn('invalid BF16 Qwen vision tensor', r.stdout+r.stderr)
        self.assertFalse(any(c[0]=='hermes' for c in self.calls()))

    def test_optional_cache_absent_does_not_prevent_serving(self):
        Path(self.env['GUFO_TUNE_CACHE']).unlink()
        self.env['GUFO_HIPBLASLT_PLAN_CACHE'] = '/unexpected/inherited/cache'
        r = self.run_script()
        self.assertEqual(r.returncode, 0, r.stdout+r.stderr)
        g = json.loads((self.root/'gpu.json').read_text())
        self.assertIsNone(g['cache'])

    def test_default_cache_is_opt_in_and_loader_environment_is_clean(self):
        self.env.pop('GUFO_TUNE_CACHE')
        self.env['GUFO_HIPBLASLT_PLAN_CACHE'] = '/unexpected/inherited/cache'
        self.env['LD_LIBRARY_PATH'] = '/unexpected/host/rocm/lib'
        self.env['LD_PRELOAD'] = 'libm.so.6'
        r = self.run_script()
        self.assertEqual(r.returncode, 0, r.stdout+r.stderr)
        g = json.loads((self.root/'gpu.json').read_text())
        self.assertIsNone(g['cache'])
        self.assertIsNone(g['ld_library_path'])
        self.assertIsNone(g['ld_preload'])

    def test_inside_toolbox_fails_before_any_external_action(self):
        # Export a shell function at the filesystem boundary, without creating
        # /run/.toolboxenv or adding a production guard-bypass setting.
        command = r'''test() {
            if [[ "$#" == 2 && "$1" == -e && "$2" == /run/.toolboxenv ]]; then
                return 0
            fi
            builtin test "$@"
        }
        export -f test
        exec bash "$1"
        '''
        r = subprocess.run(['bash', '-c', command, 'toolbox-marker-fixture', str(SCRIPT)],
                           env=self.env, text=True, capture_output=True, timeout=5)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('HOST', r.stderr)
        self.assertIn('exit', r.stderr)
        self.assertEqual(self.calls(), [])

    def test_help_does_not_start_or_configure_services(self):
        r = subprocess.run(['bash', str(SCRIPT), '--help'], env=self.env,
                           text=True, capture_output=True, timeout=5)
        self.assertEqual(r.returncode, 0, r.stdout+r.stderr)
        self.assertIn('GUFO_TUNE_CACHE', r.stdout)
        self.assertEqual(self.calls(), [])

    def test_hermes_errors_remain_visible(self):
        self.make(self.root/'bin/hermes', "import sys; print('Hermes configuration refused', file=sys.stderr); sys.exit(9)\n")
        (self.root/'ready').write_text('qwen38-27b-uncensored')
        r = self.run_script()
        self.assertEqual(r.returncode, 9, r.stdout+r.stderr)
        self.assertIn('Hermes configuration refused', r.stderr)

if __name__ == '__main__':
    unittest.main(verbosity=2)
