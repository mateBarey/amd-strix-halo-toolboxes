"""Run the real build/smoke shell fragments against isolated Nix fixtures."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / 'toolboxes/gufo-nix/build/build-gufo.sh'
SMOKE = ROOT / 'scripts/gufo-nix/smoke-image.sh'


class DevelopmentShellTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in ('opt/gufo-toolbox', 'roots', 'store', 'etc/nix', 'bin'):
            (self.root / name).mkdir(parents=True)
        self.log = self.root / 'calls.jsonl'
        self.env = {**os.environ, 'NIX_SSL_CERT_FILE': str(self.root / 'ca.pem'),
                    'NIX_BUILD_CORES': '4', 'NIX_MAX_JOBS': '1'}
        for name in ('BASH_ENV', 'ENV', 'SHELLOPTS', 'BASHOPTS'):
            self.env.pop(name, None)
        (self.root / 'ca.pem').write_text('fixture')
        self.write(self.root / 'bin/nix', f'''#!/usr/bin/python3
import json, os, pathlib, sys
root=pathlib.Path({str(self.root)!r})
args=sys.argv[1:]
with open({str(self.log)!r}, 'a') as f: f.write(json.dumps(args)+'\\n')
def output(name):
 p=root/'store'/name; p.mkdir(exist_ok=True); return str(p)
if args[0]=='build' and 'nixpkgs#bashInteractive' in args:
 if os.environ.get('FAIL_BASH')=='1': sys.exit(37)
 print(output('bash-out')); print(output('bash-man'))
elif args[0]=='build' and '--out-link' in args:
 pathlib.Path(args[args.index('--out-link')+1]).symlink_to(output('gufo'))
elif args[0]=='develop':
 pathlib.Path(args[args.index('--profile')+1]).symlink_to(output('shell'))
elif args[0]=='eval':
 print('{{}}' if '--json' in args else output('source-'+args[-1].split('.')[-1]))
''')
        self.write(self.root / 'bin/nix-store', f'''#!/usr/bin/python3
import json, pathlib, sys
root=pathlib.Path({str(self.root)!r})
targets={{p.resolve().name for p in (root/'roots').iterdir()}}
missing={{'bash-out','bash-man'}}-targets
if missing:
 print('GC would delete required Bash outputs: '+str(missing),file=sys.stderr);sys.exit(42)
(root/'gc-finished').write_text('yes')
''')

    def write(self, path, text):
        path.write_text(text)
        path.chmod(0o755)

    def tearDown(self):
        self.tmp.cleanup()

    def run_build(self):
        script = BUILD.read_text().replace('/opt/nix/bin/nix-store', str(self.root / 'bin/nix-store'))
        script = script.replace('/opt/nix/bin/nix', str(self.root / 'bin/nix'))
        script = script.replace('/opt/', str(self.root / 'opt') + '/')
        script = script.replace('/etc/nix/', str(self.root / 'etc/nix') + '/')
        script = script.replace('/nix/var/nix/gcroots/', str(self.root / 'roots') + '/')
        # Production validates output paths before creating GC roots. Keep that
        # check active, scoped to this fixture's synthetic store.
        script = script.replace('/nix/store/', str(self.root / 'store') + '/')
        return subprocess.run(['/bin/bash', '-euo', 'pipefail', '-c', script],
                              env=self.env, capture_output=True, text=True, timeout=10)

    def test_build_retains_every_default_bash_output_before_gc(self):
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        bash = next(call for call in calls if 'nixpkgs#bashInteractive' in call)
        self.assertIn('--print-out-paths', bash)
        self.assertLess(next(i for i,c in enumerate(calls) if c[0]=='registry'), calls.index(bash))
        self.assertTrue((self.root / 'gc-finished').exists())

    def test_bash_build_failure_stops_before_gc(self):
        self.env['FAIL_BASH'] = '1'
        result = self.run_build()
        self.assertEqual(result.returncode, 37, result.stderr)
        self.assertFalse((self.root / 'gc-finished').exists())

    def run_shell_smoke_fragment(self, mode):
        # Execute the actual independent-Bash and develop checks, without
        # running a container engine or touching its filesystem/user setup.
        body = SMOKE.read_text()
        fragment = body[body.index('user_nix store ping\n') + len('user_nix store ping\n'):body.index("expression='let")]
        stub = f'''#!/usr/bin/python3
import json, pathlib, sys
args=sys.argv[1:]
with open({str(self.log)!r}, 'a') as f: f.write(json.dumps(args)+'\\n')
if args[0]=='build' and {mode!r}=='missing': sys.exit(19)
if args[0]=='develop':
 if {mode!r}=='ignored': print('error (ignored): Cannot build bashInteractive',file=sys.stderr)
 if {mode!r}=='build': print("building '/nix/store/unexpected.drv'...",file=sys.stderr)
 print('gufo version fixture')
'''
        self.write(self.root / 'bin/nix', stub)
        preamble = 'set -euo pipefail\nversion="gufo version fixture"\nuser_nix() { "'+str(self.root / 'bin/nix')+'" "$@"; }\n'
        return subprocess.run(['/bin/bash', '-c', preamble + fragment],
                              env=self.env, capture_output=True, text=True, timeout=10)

    def test_offline_smoke_cannot_accept_ignored_fallback(self):
        result = self.run_shell_smoke_fragment('ignored')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ignored', result.stderr)

    def test_offline_smoke_cannot_accept_build_activity(self):
        result = self.run_shell_smoke_fragment('build')
        self.assertNotEqual(result.returncode, 0)

    def test_missing_bash_stops_before_develop(self):
        result = self.run_shell_smoke_fragment('missing')
        self.assertEqual(result.returncode, 19, result.stderr)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertFalse(any(call[0]=='develop' for call in calls))

    def test_offline_smoke_uses_no_build_and_no_substitution(self):
        result = self.run_shell_smoke_fragment('ok')
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual([c[0] for c in calls], ['build', 'develop'])
        for call in calls:
            self.assertIn('--offline', call)
            self.assertEqual(call[call.index('--max-jobs')+1], '0')
            self.assertEqual(call[call.index('--option')+1:call.index('--option')+3], ['substitute', 'false'])


if __name__ == '__main__':
    unittest.main()
