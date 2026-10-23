import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / 'toolboxes/gufo-nix/bin/gufo-tune-hipblaslt'


class TunerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'prefix/bin').mkdir(parents=True)
        self.env = {**os.environ, 'GUFO_TOOLS_PREFIX': str(self.root / 'prefix')}
        self.stub('gufo', '#!/usr/bin/env python3\nimport json,sys\nprint("gufo version test" if "--version" in sys.argv else json.dumps({"fingerprintId":"abc"}))\n')
        self.stub('tune_hipblaslt', '#!/usr/bin/env python3\nimport pathlib,sys\npathlib.Path(sys.argv[sys.argv.index("--out")+1]).write_bytes(b"plans")\nprint("tuned actual stub invocation",sys.argv[1:])\n')

    def stub(self, name, text):
        path = self.root / 'prefix/bin' / name
        path.write_text(text)
        path.chmod(0o755)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *args):
        return subprocess.run([str(CLI), '--outdir', str(self.root / 'results'), *args],
                              env=self.env, capture_output=True, text=True)

    def test_unique_results_and_exact_tool_options(self):
        first = self.run_cli('--quick', '--shape', '5120x4608', '--batch', '128,256')
        second = self.run_cli('--quick')
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertNotIn('.pending-', first.stdout)
        dirs = list((self.root / 'results').iterdir())
        self.assertEqual(len(dirs), 2)
        for directory in dirs:
            self.assertEqual((directory / 'plans.bin').read_bytes(), b'plans')
            report = json.loads((directory / 'runtime.json').read_text())
            self.assertIn('gufo version test', report['engine_version'])
            self.assertEqual(report['probe']['fingerprintId'], 'abc')
            self.assertIn('BF16', report['scope'])
        reports = [json.loads((d / 'runtime.json').read_text()) for d in dirs]
        command = next(r['command'] for r in reports if '--shape' in r['command'])
        self.assertEqual(command[command.index('--shape') + 1], '5120x4608')
        self.assertEqual(command[command.index('--batch') + 1], '128,256')

    def test_failure_does_not_publish_partial_cache(self):
        self.stub('tune_hipblaslt', '#!/usr/bin/env python3\nimport pathlib,sys\npathlib.Path(sys.argv[sys.argv.index("--out")+1]).write_bytes(b"partial")\nprint("GPU error",file=sys.stderr)\nsys.exit(7)\n')
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('GPU error', result.stderr)
        self.assertEqual(list((self.root / 'results').glob('*/plans.bin')), [])

    def test_help_explains_scope(self):
        result = subprocess.run([str(CLI), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('synthetic', result.stdout)
        self.assertIn('vision', result.stdout)
