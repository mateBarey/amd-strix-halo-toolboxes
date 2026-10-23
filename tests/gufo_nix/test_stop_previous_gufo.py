"""Exercise process selection and pidfd termination without signalling real processes."""
import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/gufo-nix/stop-previous-gufo.py'
SPEC = importlib.util.spec_from_file_location('gufo_stop_previous_test_module', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class StopPreviousGufoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.proc = self.root / 'proc'
        self.proc.mkdir()
        self.uid = 12345
        self.model = self.root / 'models' / 'original Flash.gguf'
        self.model.parent.mkdir()
        self.model.touch()
        self.binary_dir = self.root / 'bin'
        self.binary_dir.mkdir()
        self.poller = mock.Mock()
        self.poller.poll.return_value = [(700, MODULE.select.POLLIN)]
        self.open_fd = mock.patch.object(MODULE.os, 'pidfd_open', return_value=700)
        self.send_signal = mock.patch.object(MODULE.signal, 'pidfd_send_signal')
        self.close_fd = mock.patch.object(MODULE.os, 'close')
        self.poll = mock.patch.object(MODULE.select, 'poll', return_value=self.poller)
        self.fd = self.open_fd.start()
        self.send = self.send_signal.start()
        self.close = self.close_fd.start()
        self.poll.start()
        self.addCleanup(mock.patch.stopall)

    def process(self, pid, *, exe='.gufo-wrapped', argv=None, uid=None,
                start=100, state='S', effective_uid=None):
        directory = self.proc / str(pid)
        directory.mkdir(exist_ok=True)
        binary = self.binary_dir / exe
        binary.write_bytes(b'\x7fELFfixture')
        (directory / 'exe').symlink_to(binary)
        (directory / 'cwd').symlink_to(self.root)
        owner = self.uid if uid is None else uid
        effective = owner if effective_uid is None else effective_uid
        (directory / 'status').write_text(f'Name:\tgufo\nUid:\t{owner}\t{effective}\t{owner}\t{owner}\n')
        self.stat(pid, start, state)
        args = argv or [str(binary), 'serve', '--port', '8080', 'llm', '--model', str(self.model)]
        (directory / 'cmdline').write_bytes(b'\0'.join(os.fsencode(a) for a in args) + b'\0')
        return directory

    def stat(self, pid, start, state='S'):
        # The command may itself contain spaces and parentheses.
        fields = [state] + ['0'] * 18 + [str(start)] + ['0'] * 4
        (self.proc / str(pid) / 'stat').write_text(f'{pid} (gufo worker (old)) ' + ' '.join(fields))

    def run_stop(self):
        return MODULE.stop_matching(self.model, proc_root=self.proc, uid=self.uid)

    def test_only_exact_gufo_serve_model_and_current_uid_are_selected(self):
        self.process(101)
        self.process(102, exe='bash', argv=['bash', '-c', f'gufo serve --model {self.model}'])
        self.process(103, argv=['gufo', 'serve', 'llm', '--model', str(self.model) + '.backup'])
        self.process(104, uid=self.uid + 1)
        self.process(105, argv=['gufo', 'diagnose', '--model', str(self.model)])
        self.process(106, effective_uid=self.uid + 1)
        self.process(107, state='Z')
        self.process(108, argv=['gufo', 'serve', 'llm', '--model', str(self.model), '--model', '/different'])
        result = self.run_stop()
        self.assertEqual(result, {'model': str(self.model), 'stopped_pids': [101]})
        self.fd.assert_called_once_with(101, 0)
        self.send.assert_called_once_with(700, signal.SIGTERM, None, 0)
        self.close.assert_called_once_with(700)
        self.poller.poll.assert_called_once_with(15000)

    def test_equals_option_and_symlink_resolve_to_exact_model(self):
        alias = self.root / 'model alias.gguf'
        alias.symlink_to(self.model)
        self.process(201, exe='gufo-server', argv=['gufo-server', 'serve', 'llm', '--model=' + str(alias)])
        self.assertEqual(self.run_stop()['stopped_pids'], [201])

    def test_pid_reuse_between_scan_and_pidfd_open_cannot_signal(self):
        self.process(301)
        self.poller.poll.return_value = []
        def replace_pid(pid, flags):
            self.stat(pid, 999)
            return 700
        self.fd.side_effect = replace_pid
        with self.assertRaises(MODULE.StopFailure) as caught:
            self.run_stop()
        self.assertIn('identity changed', str(caught.exception))
        self.send.assert_not_called()
        self.close.assert_called_once_with(700)

    def test_command_change_after_pidfd_open_cannot_signal(self):
        directory = self.process(302)
        self.poller.poll.return_value = []
        def change_command(pid, flags):
            (directory / 'cmdline').write_bytes(b'gufo\0serve\0llm\0--model\0/different\0')
            return 700
        self.fd.side_effect = change_command
        with self.assertRaises(MODULE.StopFailure):
            self.run_stop()
        self.send.assert_not_called()

    def test_no_pidfd_support_fails_without_signal_and_no_match_succeeds(self):
        self.assertEqual(self.run_stop()['stopped_pids'], [])
        self.process(401)
        with mock.patch.object(MODULE.os, 'pidfd_open', None):
            with self.assertRaises(MODULE.StopFailure) as caught:
                self.run_stop()
        self.assertIn('pidfd', str(caught.exception))
        self.send.assert_not_called()

    def test_timeout_reports_failure_without_force_kill(self):
        self.process(501)
        self.poller.poll.return_value = []
        with self.assertRaises(MODULE.StopFailure) as caught:
            self.run_stop()
        self.assertIn('15 seconds', str(caught.exception))
        self.send.assert_called_once_with(700, signal.SIGTERM, None, 0)
        self.close.assert_called_once_with(700)

    def test_exited_process_between_scan_and_open_is_ignored(self):
        self.process(601)
        self.fd.side_effect = ProcessLookupError()
        self.assertEqual(self.run_stop()['stopped_pids'], [])
        self.send.assert_not_called()

    def test_exit_after_pidfd_open_is_checked_on_bound_descriptor(self):
        self.process(602)
        def exit_process(pid, flags):
            self.stat(pid, 100, 'Z')
            return 700
        self.fd.side_effect = exit_process
        self.assertEqual(self.run_stop()['stopped_pids'], [])
        self.poller.poll.assert_called_once_with(0)
        self.send.assert_not_called()
        self.close.assert_called_once_with(700)

    def test_status_reports_sorted_exact_matches_without_process_actions(self):
        self.process(809)
        self.process(701)
        self.process(702, exe='bash', argv=['bash', '-c', f'gufo serve --model {self.model}'])
        self.process(703, argv=['gufo', 'serve', '--model', str(self.model) + '.backup'])
        self.process(704, uid=self.uid + 1)
        self.process(705, effective_uid=self.uid + 1)
        self.process(706, state='Z')
        self.process(707, exe='gufo-server')
        (self.binary_dir / 'gufo-server').write_bytes(b'#!/bin/sh\n')
        self.process(708, argv=['gufo', 'serve', '--model', str(self.model), '--model', '/different'])
        result = MODULE.status_matching(self.model, proc_root=self.proc, uid=self.uid)
        self.assertEqual(result, {'model': str(self.model), 'running_pids': [701, 809]})
        self.fd.assert_not_called()
        self.send.assert_not_called()
        self.close.assert_not_called()
        self.poller.poll.assert_not_called()
        MODULE.select.poll.assert_not_called()

    def test_status_rejects_identity_change_during_inspection(self):
        self.process(801)
        read_identity = MODULE._stat_identity
        def change_identity(directory, pid):
            identity = read_identity(directory, pid)
            self.stat(pid, 999)
            return identity
        with mock.patch.object(MODULE, '_stat_identity', side_effect=change_identity):
            result = MODULE.status_matching(self.model, proc_root=self.proc, uid=self.uid)
        self.assertEqual(result['running_pids'], [])
        self.fd.assert_not_called()
        self.send.assert_not_called()

    def test_status_resolves_model_alias_without_pidfd_support(self):
        alias = self.root / 'model alias.gguf'
        alias.symlink_to(self.model)
        self.process(901, argv=['gufo', 'serve', '--model=models/original Flash.gguf'])
        with mock.patch.object(MODULE.os, 'pidfd_open', None), \
                mock.patch.object(MODULE.signal, 'pidfd_send_signal', None):
            result = MODULE.status_matching(alias, proc_root=self.proc, uid=self.uid)
        self.assertEqual(result, {'model': str(self.model), 'running_pids': [901]})
        MODULE.select.poll.assert_not_called()

    def test_status_cli_returns_empty_success_and_reports_errors(self):
        for failure, expected_code in ((None, 0), (OSError('cannot scan proc'), 1)):
            with self.subTest(failure=failure):
                output = io.StringIO()
                with mock.patch.object(sys, 'argv', [str(SCRIPT), '--status', '--model', str(self.model)]), \
                        mock.patch.object(MODULE, 'status_matching',
                                          return_value={'model': str(self.model), 'running_pids': []},
                                          side_effect=failure), \
                        mock.patch.object(MODULE, 'stop_matching') as stop, \
                        contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                    try:
                        code = MODULE.main()
                    except SystemExit as error:
                        code = error.code
                self.assertEqual(code, expected_code)
                result = json.loads(output.getvalue())
                self.assertEqual(result['model'], str(self.model))
                self.assertEqual(result['running_pids'], [])
                self.assertNotIn('stopped_pids', result)
                if failure:
                    self.assertIn('cannot scan proc', result['error'])
                stop.assert_not_called()

    def test_cli_requires_model_and_help_never_scans_processes(self):
        with mock.patch.object(MODULE, 'stop_matching') as stop, \
                mock.patch.object(MODULE, 'status_matching') as status:
            for arguments, code in (([], 2), (['--status'], 2), (['--help'], 0),
                                    (['--status', '--help'], 0)):
                output, errors = io.StringIO(), io.StringIO()
                with mock.patch.object(sys, 'argv', [str(SCRIPT), *arguments]), \
                        contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                    with self.assertRaises(SystemExit) as caught:
                        MODULE.main()
                self.assertEqual(caught.exception.code, code)
                self.assertIn('--model', output.getvalue() + errors.getvalue())
            stop.assert_not_called()
            status.assert_not_called()


if __name__ == '__main__':
    unittest.main()
