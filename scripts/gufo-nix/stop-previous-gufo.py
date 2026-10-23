#!/usr/bin/env python3
"""Inspect or stop only an explicitly selected, current-user Gufo model."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import select
import signal
import stat
import sys


GUFO_EXECUTABLES = frozenset({'gufo', 'gufo-server', '.gufo-wrapped', '.gufo-server-wrapped'})
GRACE_MILLISECONDS = 15000
MAX_CMDLINE_BYTES = 131072


class StopFailure(RuntimeError):
    def __init__(self, message: str, stopped_pids: list[int] | None = None):
        super().__init__(message)
        self.stopped_pids = list(stopped_pids or [])


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_time: int
    executable_device: int
    executable_inode: int
    argv: tuple[str, ...]
    model: str


def _stat_identity(directory: Path, pid: int) -> tuple[int, str]:
    # comm can contain spaces or ')'; fields after its final ')' start at state.
    content = (directory / 'stat').read_text()
    closing = content.rfind(')')
    if closing < 0 or int(content.split(' ', 1)[0]) != pid:
        raise ValueError('invalid process stat')
    fields = content[closing + 1:].split()
    return int(fields[19]), fields[0]


def _model_argument(argv: tuple[str, ...]) -> str | None:
    # The supported launcher starts the real executable with `gufo serve ...`.
    # Shells/wrappers mentioning that text and other Gufo commands do not match.
    if len(argv) < 3 or argv[1] != 'serve':
        return None
    values = []
    index = 2
    while index < len(argv):
        argument = argv[index]
        if argument == '--':
            return None
        if argument == '--model':
            index += 1
            if index >= len(argv) or not argv[index] or argv[index].startswith('--'):
                return None
            values.append(argv[index])
        elif argument.startswith('--model='):
            values.append(argument[len('--model='):])
        index += 1
    # Reject duplicates instead of guessing the CLI parser's winner.
    return values[0] if len(values) == 1 and values[0] else None


def _matching_identity(proc_root: Path, pid: int, uid: int,
                       target: str) -> ProcessIdentity | None:
    directory = proc_root / str(pid)
    try:
        before, state = _stat_identity(directory, pid)
        if state in ('Z', 'X', 'x'):
            return None
        owner_line = next(line for line in (directory / 'status').read_text().splitlines()
                          if line.startswith('Uid:'))
        owners = [int(value) for value in owner_line.split()[1:]]
        if len(owners) != 4 or any(owner != uid for owner in owners):
            return None
        executable = os.readlink(directory / 'exe')
        basename = Path(executable.removesuffix(' (deleted)')).name
        if basename not in GUFO_EXECUTABLES:
            return None
        executable_info = (directory / 'exe').stat()
        if not stat.S_ISREG(executable_info.st_mode):
            return None
        with (directory / 'exe').open('rb') as stream:
            if stream.read(4) != b'\x7fELF':
                return None
        with (directory / 'cmdline').open('rb') as stream:
            raw = stream.read(MAX_CMDLINE_BYTES + 1)
        if not raw.endswith(b'\0') or len(raw) > MAX_CMDLINE_BYTES:
            return None
        argv = tuple(os.fsdecode(value) for value in raw[:-1].split(b'\0'))
        model = _model_argument(argv)
        if model is None:
            return None
        model_path = Path(model)
        if not model_path.is_absolute():
            model_path = Path(os.readlink(directory / 'cwd')) / model_path
        resolved = str(model_path.resolve())
        if resolved != target:
            return None
        after, state = _stat_identity(directory, pid)
        executable_after = (directory / 'exe').stat()
        if before != after or state in ('Z', 'X', 'x'):
            return None
        if (executable_info.st_dev, executable_info.st_ino) != (
                executable_after.st_dev, executable_after.st_ino):
            return None
        return ProcessIdentity(pid, before, executable_info.st_dev,
                               executable_info.st_ino, argv, resolved)
    except (OSError, ValueError, IndexError, StopIteration):
        # An unreadable or changing process never becomes a termination target.
        return None


def _select_matches(model: str | Path, *, proc_root: Path,
                    uid: int | None) -> tuple[str, int, list[ProcessIdentity]]:
    if not str(model):
        raise StopFailure('An explicit model path is required.')
    target = str(Path(model).expanduser().resolve())
    current_uid = os.getuid() if uid is None else uid
    candidates = []
    for directory in proc_root.iterdir():
        if not directory.name.isascii() or not directory.name.isdecimal():
            continue
        pid = int(directory.name)
        if pid == os.getpid():
            continue
        identity = _matching_identity(proc_root, pid, current_uid, target)
        if identity is not None:
            candidates.append(identity)
    candidates.sort(key=lambda identity: identity.pid)
    return target, current_uid, candidates


def status_matching(model: str | Path, *, proc_root: Path = Path('/proc'),
                    uid: int | None = None) -> dict:
    """Report a read-only snapshot of verified matches without opening pidfds."""
    target, _, candidates = _select_matches(model, proc_root=proc_root, uid=uid)
    return {'model': target, 'running_pids': [identity.pid for identity in candidates]}


def stop_matching(model: str | Path, *, proc_root: Path = Path('/proc'),
                  uid: int | None = None) -> dict:
    """Stop verified matches. proc_root/uid are injectable only for unit tests."""
    target, current_uid, candidates = _select_matches(model, proc_root=proc_root, uid=uid)
    stopped = []
    if candidates and (not callable(getattr(os, 'pidfd_open', None)) or
                       not callable(getattr(signal, 'pidfd_send_signal', None))):
        raise StopFailure('Linux pidfd support is required; no process was signalled.')

    for identity in candidates:
        try:
            descriptor = os.pidfd_open(identity.pid, 0)
        except ProcessLookupError:
            continue
        except OSError as error:
            raise StopFailure(f'Cannot pin Gufo PID {identity.pid}: {error.strerror}', stopped) from error
        try:
            # A pidfd pins the task receiving the signal. Check the /proc
            # identity again after opening it to reject PID reuse or exec changes.
            poller = select.poll()
            poller.register(descriptor, select.POLLIN | select.POLLHUP)
            current = _matching_identity(proc_root, identity.pid, current_uid, target)
            if current != identity:
                # A task which exited while being inspected is already gone.
                # Readiness is checked on the pinned descriptor, never by PID.
                if any(fd == descriptor and event & (select.POLLIN | select.POLLHUP)
                       for fd, event in poller.poll(0)):
                    continue
                raise StopFailure(f'Gufo PID {identity.pid} identity changed; no signal sent.', stopped)
            try:
                signal.pidfd_send_signal(descriptor, signal.SIGTERM, None, 0)
            except ProcessLookupError:
                continue
            events = poller.poll(GRACE_MILLISECONDS)
            if not any(fd == descriptor and event & (select.POLLIN | select.POLLHUP)
                       for fd, event in events):
                raise StopFailure(
                    f'Gufo PID {identity.pid} did not exit within 15 seconds after SIGTERM; '
                    'no force-kill was sent.', stopped)
            stopped.append(identity.pid)
        except OSError as error:
            raise StopFailure(f'Could not stop Gufo PID {identity.pid}: {error.strerror}', stopped) from error
        finally:
            os.close(descriptor)
    return {'model': target, 'stopped_pids': stopped}


def main() -> int:
    parser = argparse.ArgumentParser(description=(
        'Send SIGTERM only to your Gufo serve processes using the exact selected model. '
        'Wait up to 15 seconds; never force-kill or stop another model. '
        'Use --status to report matching PIDs without stopping them.'))
    parser.add_argument('--model', required=True, help='Exact model path selected for replacement')
    parser.add_argument('--status', action='store_true',
                        help='Report matching PIDs without opening pidfds, signalling, or waiting')
    arguments = parser.parse_args()
    try:
        result = (status_matching(arguments.model) if arguments.status
                  else stop_matching(arguments.model))
    except (StopFailure, OSError) as error:
        pids_key = 'running_pids' if arguments.status else 'stopped_pids'
        pids = [] if arguments.status else getattr(error, 'stopped_pids', [])
        print(json.dumps({'model': str(Path(arguments.model).expanduser().resolve()),
                          pids_key: pids,
                          'error': str(error)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())
