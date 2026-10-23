#!/usr/bin/env python3
"""Read memory accounting using the installed Gufo HIP/ROCr libraries.

HIP initialization creates runtime resources; no model or benchmark buffers
are allocated. Process arguments are restricted to non-secret model settings.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform


ENVIRONMENT_KEYS = (
    'HIP_HIDDEN_FREE_MEM', 'GPU_MAX_HEAP_SIZE', 'GPU_SINGLE_ALLOC_PERCENT',
    'HSA_LOCAL_MEMORY_ENABLE', 'HSA_XNACK', 'HIP_VISIBLE_DEVICES',
    'ROCR_VISIBLE_DEVICES',
)
GUFO_EXECUTABLES = {'gufo', 'gufo-server', '.gufo-wrapped', '.gufo-server-wrapped'}


def read_text(path):
    try:
        return Path(path).read_text(errors='replace').strip()
    except OSError:
        return None


def parse_meminfo(text):
    result = {}
    wanted = {'MemTotal', 'MemFree', 'MemAvailable', 'SwapTotal', 'SwapFree',
              'AnonPages', 'Shmem', 'Mlocked', 'Unevictable'}
    for line in text.splitlines():
        name, separator, value = line.partition(':')
        words = value.split()
        if separator and name in wanted and words:
            try:
                result[name] = int(words[0]) * (1024 if words[1:] == ['kB'] else 1)
            except ValueError:
                pass
    return result


def process_start_ticks(text):
    # /proc/PID/stat comm may contain spaces and parentheses; field22 follows
    # the final closing parenthesis by19 zero-based fields starting with state.
    return int(text.rsplit(')', 1)[1].split()[19])


def selected_arguments(arguments):
    selected = {name: None for name in ('model', 'mmproj', 'port', 'served_model_name')}
    flags = {'--model': 'model', '--mmproj': 'mmproj', '--port': 'port',
             '--served-model-name': 'served_model_name'}
    for index, argument in enumerate(arguments):
        flag, equal, value = argument.partition('=')
        if flag in flags:
            if not equal:
                value = arguments[index + 1] if index + 1 < len(arguments) else None
            selected[flags[flag]] = value
    return selected


def inspect_gufo_process(pid, proc_root=Path('/proc'), *, uid=None):
    """Return a stable, same-UID Gufo identity; never signal the process."""
    uid = os.getuid() if uid is None else uid
    process = Path(proc_root) / str(pid)
    try:
        if process.stat().st_uid != uid:
            return None
        before = process_start_ticks((process / 'stat').read_text())
        executable = os.readlink(process / 'exe')
        if Path(executable.removesuffix(' (deleted)')).name not in GUFO_EXECUTABLES:
            return None
        fields = dict(line.split(':', 1) for line in (process / 'status').read_text().splitlines() if ':' in line)
        if int(fields['Uid'].split()[0]) != uid:
            return None
        arguments = [item.decode(errors='replace') for item in (process / 'cmdline').read_bytes().split(b'\0') if item]
        after = process_start_ticks((process / 'stat').read_text())
        if before != after or os.readlink(process / 'exe') != executable:
            return None
        return {
            'pid': int(pid), 'uid': uid, 'executable': executable,
            'start_time_ticks': before,
            'rss_bytes': int(fields.get('VmRSS', '0').split()[0]) * 1024,
            'serving': len(arguments) > 1 and arguments[1] == 'serve',
            **selected_arguments(arguments[1:]),
        }
    except (OSError, ValueError, KeyError, IndexError):
        return None


def scan_gufo_processes(proc_root=Path('/proc'), *, uid=None):
    processes = []
    try:
        entries = list(Path(proc_root).iterdir())
    except OSError:
        return processes
    for entry in entries:
        if entry.name.isdecimal():
            process = inspect_gufo_process(int(entry.name), proc_root, uid=uid)
            if process is not None:
                processes.append(process)
    return sorted(processes, key=lambda item: item['pid'])


def query_hip(library, *, loader=ctypes.CDLL):
    result = {'status': 'error', 'devices': [], 'errors': []}
    try:
        hip = loader(str(library))
        for name, arguments in (
            ('hipInit', [ctypes.c_uint]),
            ('hipGetDeviceCount', [ctypes.POINTER(ctypes.c_int)]),
            ('hipSetDevice', [ctypes.c_int]),
            ('hipMemGetInfo', [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]),
        ):
            function = getattr(hip, name)
            function.argtypes = arguments
            function.restype = ctypes.c_int
        hip.hipGetErrorName.argtypes = [ctypes.c_int]
        hip.hipGetErrorName.restype = ctypes.c_char_p

        def checked(name, *arguments):
            code = getattr(hip, name)(*arguments)
            if code:
                detail = hip.hipGetErrorName(code)
                raise RuntimeError(f'{name}: {detail.decode(errors="replace") if detail else code}')

        checked('hipInit', 0)
        count = ctypes.c_int()
        checked('hipGetDeviceCount', ctypes.byref(count))
        if not 1 <= count.value <= 64:
            raise RuntimeError(f'No usable HIP device count: {count.value}')
        for index in range(count.value):
            checked('hipSetDevice', index)
            free, total = ctypes.c_size_t(), ctypes.c_size_t()
            checked('hipMemGetInfo', ctypes.byref(free), ctypes.byref(total))
            result['devices'].append({'index': index, 'free_bytes': free.value, 'total_bytes': total.value})
        result['status'] = 'ok'
    except (OSError, AttributeError, RuntimeError) as error:
        result['errors'].append(str(error))
    return result


def gpu_memory():
    result = []
    for device in sorted(Path('/sys/class/drm').glob('card[0-9]*/device')):
        if read_text(device / 'vendor') != '0x1002':
            continue
        values = {'device': str(device.resolve())}
        for name in ('mem_info_gtt_total', 'mem_info_gtt_used',
                     'mem_info_vram_total', 'mem_info_vram_used'):
            value = read_text(device / name)
            if value is not None and value.isdecimal():
                values[name] = int(value)
        result.append(values)
    return result


def kfd_clients():
    clients = []
    for path in sorted(Path('/sys/devices/virtual/kfd/kfd/proc').glob('[0-9]*')):
        allocations = {}
        for allocation in path.glob('vram_*'):
            value = read_text(allocation)
            if value is not None and value.isdecimal():
                allocations[allocation.name[5:]] = int(value)
        pid = int(path.name)
        try:
            uid = Path(f'/proc/{pid}').stat().st_uid
        except OSError:
            uid = None
        clients.append({'pid': pid, 'uid': uid, 'comm': read_text(f'/proc/{pid}/comm'),
                        'accounted_vram_bytes_by_gpu': allocations})
    return clients


def kernel_settings():
    keys = ('amd_iommu', 'ttm.pages_limit', 'ttm.page_pool_size',
            'amdgpu.gttsize', 'amdgpu.vm_fragment_size', 'amdgpu.lockup_timeout')
    parameters = {}
    for name in keys[1:]:
        module, parameter = name.split('.', 1)
        parameters[name] = read_text(f'/sys/module/{module}/parameters/{parameter}')
    # Do not dump unrelated kernel command-line values.
    command_line = read_text('/proc/cmdline') or ''
    return {'release': platform.release(), 'parameters': parameters,
            'relevant_boot_arguments': [token for token in command_line.split()
                                        if token.partition('=')[0] in keys]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hip-library', required=True, type=Path)
    parser.add_argument('--rocr-library-dir', required=True, type=Path)
    args = parser.parse_args()
    hip = query_hip(args.hip_library)
    mapped = set()
    for line in (read_text('/proc/self/maps') or '').splitlines():
        if 'libhsa-runtime64.so' in line or 'libamdhip64.so' in line:
            mapped.add(line.split()[-1])
    result = {
        'schema': 'gufo-memory.v1',
        'captured_at': datetime.now(timezone.utc).isoformat(),
        'runtime': {'hip_library': str(args.hip_library.resolve()),
                    'rocr_library_dir': str(args.rocr_library_dir.resolve()),
                    'mapped_libraries': sorted(mapped)},
        'hip': hip,
        'host_memory': parse_meminfo(read_text('/proc/meminfo') or ''),
        'environment': {name: os.environ.get(name) for name in ENVIRONMENT_KEYS},
        'gufo_processes': scan_gufo_processes(),
        'gpu_memory': gpu_memory(),
        'kfd_clients': kfd_clients(),
        'kernel': kernel_settings(),
    }
    print(json.dumps(result, indent=2))
    return 0 if hip['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
