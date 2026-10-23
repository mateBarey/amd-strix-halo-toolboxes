"""Memory diagnostics parse real-shaped data without requiring an AMD GPU."""
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'toolboxes/gufo-nix/gufo_memory.py'


def load_module():
    spec = importlib.util.spec_from_file_location('gufo_memory', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class FakeHip:
    def __init__(self, failure=0):
        self.hipInit = Function(lambda _: 0)
        self.hipGetDeviceCount = Function(lambda p: self.assign(p, ctypes.c_int, 1))
        self.hipSetDevice = Function(lambda _: 0)
        self.hipMemGetInfo = Function(lambda free, total: failure or
            (self.assign(free, ctypes.c_size_t, 9 * 1024**3) or
             self.assign(total, ctypes.c_size_t, 124 * 1024**3)))
        self.hipGetErrorName = Function(lambda _: b'hipErrorFixture')

    @staticmethod
    def assign(pointer, kind, value):
        ctypes.cast(pointer, ctypes.POINTER(kind))[0] = value
        return 0


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def process(self, *, executable='/nix/store/example/bin/.gufo-wrapped',
                uid=None, name='gufo worker'):
        path = self.root / '123'
        path.mkdir()
        (path / 'exe').symlink_to(executable)
        (path / 'cmdline').write_bytes(b'gufo\0serve\0--port\08081\0--model=/models/model.gguf\0--mmproj\0/models/mmproj.gguf\0--api-key\0SECRET\0')
        (path / 'status').write_text(f'Name:\tgufo\nUid:\t{os.getuid() if uid is None else uid}\t0\t0\t0\nVmRSS:\t4096 kB\n')
        # Field22 starts19 fields after the comm field; comm can contain spaces.
        fields = ['S'] + ['0'] * 18 + ['98765'] + ['0'] * 4
        (path / 'stat').write_text(f'123 ({name}) ' + ' '.join(fields))
        return path

    def test_meminfo_preserves_bytes_and_distinguishes_missing_fields(self):
        parsed = self.module.parse_meminfo('MemTotal: 125000 kB\nMemAvailable: 7 kB\nSwapFree: 0 kB\nHugePages_Total: 3\n')
        self.assertEqual(parsed['MemAvailable'], 7168)
        self.assertEqual(parsed['SwapFree'], 0)
        self.assertNotIn('MemFree', parsed)

    def test_hip_query_reports_the_actual_free_and_total_separately(self):
        result = self.module.query_hip('/exact/libamdhip64.so', loader=lambda _: FakeHip())
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['devices'], [{'index': 0, 'free_bytes': 9 * 1024**3, 'total_bytes': 124 * 1024**3}])

    def test_hip_failure_is_not_reported_as_zero_available_or_success(self):
        result = self.module.query_hip('/exact/libamdhip64.so', loader=lambda _: FakeHip(failure=2))
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['devices'], [])
        self.assertIn('hipErrorFixture', result['errors'][0])

    def test_owned_gufo_process_is_identified_without_dumping_secret_arguments(self):
        self.process()
        processes = self.module.scan_gufo_processes(self.root, uid=os.getuid())
        self.assertEqual(len(processes), 1)
        process = processes[0]
        self.assertEqual(process['pid'], 123)
        self.assertEqual(process['model'], '/models/model.gguf')
        self.assertEqual(process['port'], '8081')
        self.assertEqual(process['rss_bytes'], 4194304)
        self.assertEqual(process['start_time_ticks'], 98765)
        self.assertTrue(process['serving'])
        self.assertNotIn('SECRET', str(process))

    def test_process_named_gufo_but_running_another_executable_is_rejected(self):
        self.process(executable='/usr/bin/python3')
        self.assertEqual(self.module.scan_gufo_processes(self.root, uid=os.getuid()), [])

    def test_other_uid_and_disappeared_process_are_rejected(self):
        path = self.process(uid=os.getuid() + 1)
        self.assertEqual(self.module.scan_gufo_processes(self.root, uid=os.getuid()), [])
        (path / 'stat').unlink()
        self.assertIsNone(self.module.inspect_gufo_process(123, self.root, uid=os.getuid()))


class MemoryShellTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'repo with spaces'
        self.script = self.repo / 'scripts/gufo-nix/inspect-memory.sh'
        self.script.parent.mkdir(parents=True)
        (self.repo / 'toolboxes/gufo-nix').mkdir(parents=True)
        self.installed = self.root / 'installed'
        self.installed.mkdir()
        (self.root / 'profile').symlink_to(self.installed)
        for folder in ('clr/lib', 'rocr/lib'):
            (self.root / folder).mkdir(parents=True)
        (self.root / 'clr/lib/libamdhip64.so').touch()
        self.marker = self.root / 'toolboxenv'
        self.marker.touch()
        self.env = dict(os.environ, TEST_ROOT=str(self.root), LD_LIBRARY_PATH='/wrong/rocm')
        self.stub('nix', '''import json, os, pathlib, sys
r=pathlib.Path(os.environ['TEST_ROOT']); a=sys.argv[1:]
with (r/'calls').open('a') as f: f.write(json.dumps(a)+'\\n')
assert '--offline' in a
assert a[0] in ('path-info','eval'), a
if a[0]=='path-info': print(r/('other' if os.environ.get('TEST_MISMATCH') else 'installed'))
elif a[-1].endswith('.clr.outPath'): print(r/'clr')
elif a[-1].endswith('.rocmRuntime.outPath'): print(r/'rocr')
else: sys.exit(19)
''')
        self.stub('python-probe', '''import json, os, pathlib, sys
r=pathlib.Path(os.environ['TEST_ROOT'])
print(json.dumps({'args':sys.argv[1:], 'loader':os.environ.get('LD_LIBRARY_PATH'), 'preload':os.environ.get('LD_PRELOAD')}))
(r/'queried').touch()
''')
        text = (ROOT / 'scripts/gufo-nix/inspect-memory.sh').read_text()
        for original, replacement in (
            ('/run/.toolboxenv', self.marker),
            ('/run/.containerenv', self.root / 'containerenv'),
            ('/usr/local/bin/nix', self.root / 'nix'),
            ('/usr/bin/python3', self.root / 'python-probe'),
            ('/opt/gufo', self.root / 'profile'),
        ):
            text = text.replace(original, str(replacement))
        self.script.write_text(text)

    def stub(self, name, body):
        path = self.root / name
        path.write_text('#!/usr/bin/python3\n' + body)
        path.chmod(0o755)

    def run_probe(self):
        return subprocess.run(['bash', str(self.script)], env=self.env,
                              capture_output=True, text=True, timeout=10)

    def test_inside_toolbox_selects_exact_libraries_without_building_or_nesting(self):
        result = self.run_probe()
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output['loader'], str(self.root / 'rocr/lib'))
        self.assertIsNone(output['preload'])
        self.assertIn(str(self.root / 'clr/lib/libamdhip64.so'), output['args'])
        calls = [json.loads(line) for line in (self.root / 'calls').read_text().splitlines()]
        self.assertEqual([call[0] for call in calls], ['path-info', 'eval', 'eval'])

    def test_different_installed_package_stops_before_hip_initialization(self):
        self.env['TEST_MISMATCH'] = '1'
        result = self.run_probe()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / 'queried').exists())
        self.assertIn('different runtime', result.stderr)


if __name__ == '__main__':
    unittest.main()
