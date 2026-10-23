#!/usr/bin/env python3
"""Host installer checks; uses real Git and isolated temporary fixtures, no GPU."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/gufo-nix/setup-qwen27b.sh'


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), "repository setup command must exist")
        self.temp = tempfile.TemporaryDirectory(prefix='gufo-oneshot-check-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = os.environ.copy()

    def shell(self, body, *, ok=True, env=None, script=SCRIPT):
        result = subprocess.run(['bash', '-c', f'source {shlex.quote(str(script))}\n' + body],
                                env=env or self.env, text=True, capture_output=True)
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def executable(self, name, text):
        bindir = self.root / 'bin'
        bindir.mkdir(exist_ok=True)
        path = bindir / name
        path.write_text(text)
        path.chmod(0o755)
        self.env['PATH'] = str(bindir) + os.pathsep + os.environ['PATH']
        return path

    def test_foreign_directory_untouched(self):
        work = self.root / 'existing'
        work.mkdir()
        precious = work / 'user-file'
        precious.write_text('keep this')
        self.shell(f'SETUP_DIR={shlex.quote(str(work))}\nprepare_directory', ok=False)
        self.assertEqual(precious.read_text(), 'keep this')
        self.assertEqual(sorted(p.name for p in work.iterdir()), ['user-file'])

    def test_container_collision_uses_suffix_and_engine_error_stops(self):
        self.executable('podman', '''#!/usr/bin/env bash
case "$3" in
  gufo-rocm-nix|gufo-rocm-nix-1) exit 0 ;;
  gufo-rocm-nix-2) exit 1 ;;
esac
exit 125
''')
        result = self.shell('choose_container_name\nprintf "%s" "$TOOLBOX_NAME"')
        self.assertEqual(result.stdout, 'gufo-rocm-nix-2')
        self.executable('podman', '#!/usr/bin/env bash\nexit 125\n')
        result = self.shell('choose_container_name', ok=False)
        self.assertIn('exit 125', result.stderr)

    def test_occupied_port_is_skipped(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind(('0.0.0.0', 0))
            occupied = held.getsockname()[1]
            if occupied > 65400:
                self.skipTest('Ephemeral port too close to the maximum for this test')
            result = self.shell(f'FIRST_PORT={occupied}\nchoose_port')
        self.assertGreater(int(result.stdout), occupied)
        self.assertLess(int(result.stdout), occupied + 100)

    def test_toolbox_probes_clear_loader_overrides_before_tool_execution(self):
        self.executable('toolbox', '''#!/usr/bin/env bash
[[ "$1" == run && "$2" == -c && "$3" == gufo-rocm-nix ]] || exit 99
shift 3
exec "$@"
''')
        result = self.shell('''TOOLBOX_NAME=gufo-rocm-nix
export LD_LIBRARY_PATH=/unwanted/old/rocm/lib
export LD_PRELOAD=libm.so.6
in_toolbox python3 -c 'import os; print("LD_LIBRARY_PATH" in os.environ, "LD_PRELOAD" in os.environ)'
''')
        self.assertEqual(result.stdout.strip(), 'False False')

    def test_saved_launcher_preserves_exact_values_with_shell_characters(self):
        work = self.root / "work with space and ' quote"
        repo = work / 'repo'
        helper = repo / 'scripts/gufo-nix/launch-hermes-qwen27b.sh'
        helper.parent.mkdir(parents=True)
        captured = self.root / 'captured.json'
        helper.write_text('#!/usr/bin/env bash\npython3 - <<\'PY\'\nimport json,os\n'
                          f'open({str(captured)!r},"w").write(json.dumps(dict(os.environ)))\nPY\n')
        values = {
            'SETUP_DIR': str(work), 'REPO': str(repo), 'LAUNCHER': str(work / 'start-qwen27b.sh'),
            'TOOLBOX_NAME': 'gufo-rocm-nix-2', 'PORT': '8083', 'CONTEXT': '262144',
            'MODEL': str(work / 'model.gguf'), 'F16': str(work / 'input.gguf'),
            'BF16': str(work / 'converted.gguf'), 'IMAGE': 'localhost/gufo-nix:latest',
            'PROJECT': str(work / 'project $(false) `false` $HOME'),
        }
        assignments = '\n'.join(f'{key}={shlex.quote(value)}' for key, value in values.items())
        self.shell(assignments + '\nsave_launcher\nbash "$LAUNCHER"')
        got = json.loads(captured.read_text())
        for envkey, key in [('GUFO_TOOLBOX_NAME', 'TOOLBOX_NAME'), ('GUFO_PORT', 'PORT'),
                            ('GUFO_MODEL_PATH', 'MODEL'), ('GUFO_VISION_PATH', 'BF16'),
                            ('GUFO_PROJECT_ROOT', 'PROJECT')]:
            self.assertEqual(got[envkey], values[key])
        self.assertEqual(got['GUFO_TUNE_CACHE'], '')

    def test_prior_marker_migrates_under_lock_without_touching_old_repo(self):
        work = self.root / 'setup'
        old_repo = work / 'repo'
        old_repo.mkdir(parents=True)
        precious = old_repo / 'local-work'
        precious.write_text('keep this exact dirty checkout')
        (work / 'setup-id').write_text('mateBarey-gufo-nix-v1:167275137ec25be536ea7bb74c5c612602e44efd6dfb697e84f1ff71c74eaea3\n')
        self.shell(f'SETUP_DIR={shlex.quote(str(work))}\nprepare_directory')
        self.assertEqual((work/'setup-id').read_text().strip(), 'mateBarey-gufo-nix-repository-v2')
        self.assertEqual(precious.read_text(), 'keep this exact dirty checkout')
        self.shell(f'SETUP_DIR={shlex.quote(str(work))}\nprepare_directory')
        self.assertEqual(precious.read_text(), 'keep this exact dirty checkout')

    def test_failed_build_stops_before_toolbox_mutations(self):
        # The copied real setup derives this isolated repository from its path.
        repo = self.root / 'fresh source'
        scripts = repo / 'scripts/gufo-nix'
        scripts.mkdir(parents=True)
        copied = scripts / SCRIPT.name
        copied.write_bytes(SCRIPT.read_bytes())
        build = scripts / 'build-local.sh'
        build.write_text('#!/usr/bin/env bash\nprintf "injected build failure\\n"\nexit 42\n')
        called = self.root / 'toolbox-was-called'
        self.executable('toolbox', f'#!/usr/bin/env bash\ntouch {shlex.quote(str(called))}\n')
        work = self.root / 'setup'
        self.env['GUFO_SETUP_DIR'] = str(work)
        result = self.shell('require_host() { :; }; main', ok=False, script=copied)
        self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
        self.assertFalse(called.exists())
        self.assertIn('injected build failure', (work / 'build.log').read_text())


if __name__ == '__main__':
    unittest.main(verbosity=2)
