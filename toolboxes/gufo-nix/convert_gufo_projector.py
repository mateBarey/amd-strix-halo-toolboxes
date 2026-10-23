#!/usr/bin/env python3
"""Convert the Qwen3.8-27B F16 vision projector to Gufo's BF16/F32 contract.

Requires numpy and gguf (tested with llama.cpp gguf-py 0.19.0, commit
36cbf415f1c72c199e956c00fed0418f22ca53f2). No GPU required.

The input is mapped read-only. All 334 tensor names/shapes, metadata and dtypes
are checked before creating output. Only 110 matrix weights may change: F16
values are numerically converted to BF16 using gguf's nearest-even/quiet-NaN
codec. The 224 F32 tensors stay byte-identical. Existing BF16 matrix weights
are copied. All metadata and its original scalar/array types are preserved,
except general.file_type (if present), which is corrected to MOSTLY_BF16=32.
Little-endian input is required by the x86 Gufo deployment.

Output is streamed one tensor at a time to a temporary file, fully reopened
and checked, then published with an atomic no-overwrite hard link. Conversion
uses chunks of 1,048,576 values and at most one converted tensor buffer.
"""
from __future__ import annotations

import argparse
import mmap
from collections import Counter
import os
from pathlib import Path
import sys
import tempfile

try:
    import numpy as np
    import gguf
except ImportError as exc:
    raise SystemExit('Install dependencies first: python3 -m pip install "gguf==0.19.0"') from exc

Q = gguf.GGMLQuantizationType


def tensor_contract():
    """File-order dimensions and required dtype, matching Gufo encoder.hip."""
    result = {}
    for i in range(27):
        p = f'v.blk.{i}.'
        for name in ('ln1', 'ln2'):
            for suffix in ('weight', 'bias'):
                result[p + name + '.' + suffix] = ((1152,), Q.F32)
        for name, cols, rows in (
            ('attn_qkv', 1152, 3456), ('attn_out', 1152, 1152),
            ('ffn_up', 1152, 4304), ('ffn_down', 4304, 1152),
        ):
            result[p + name + '.weight'] = ((cols, rows), Q.BF16)
            result[p + name + '.bias'] = ((rows,), Q.F32)
    result.update({
        'v.patch_embd.weight': ((16, 16, 3, 1152), Q.F32),
        'v.patch_embd.weight.1': ((16, 16, 3, 1152), Q.F32),
        'v.patch_embd.bias': ((1152,), Q.F32),
        'v.position_embd.weight': ((1152, 2304), Q.F32),
        'v.post_ln.weight': ((1152,), Q.F32),
        'v.post_ln.bias': ((1152,), Q.F32),
        'mm.0.weight': ((4608, 4608), Q.BF16),
        'mm.0.bias': ((4608,), Q.F32),
        'mm.2.weight': ((4608, 5120), Q.BF16),
        'mm.2.bias': ((5120,), Q.F32),
    })
    return result


def validate(reader, *, allow_f16):
    """Reject files outside the exact 27B projector contract before writing."""
    if reader.endianess != gguf.GGUFEndian.LITTLE:
        raise ValueError('Gufo on x86 requires a little-endian projector')
    required = {
        'general.architecture': 'clip', 'general.type': 'mmproj',
        'clip.projector_type': 'qwen3vl_merger',
        'clip.vision.projection_dim': 5120, 'clip.vision.patch_size': 16,
        'clip.vision.embedding_length': 1152,
        'clip.vision.feed_forward_length': 4304,
        'clip.vision.block_count': 27, 'clip.vision.attention.head_count': 16,
        'clip.vision.spatial_merge_size': 2, 'clip.vision.image_size': 768,
        'clip.has_vision_encoder': True, 'clip.use_gelu': True,
        'clip.vision.image_mean': [0.5, 0.5, 0.5],
        'clip.vision.image_std': [0.5, 0.5, 0.5],
        'clip.vision.is_deepstack_layers': [0] * 27,
    }
    integer_types = {gguf.GGUFValueType.UINT8, gguf.GGUFValueType.INT8,
                     gguf.GGUFValueType.UINT16, gguf.GGUFValueType.INT16,
                     gguf.GGUFValueType.UINT32, gguf.GGUFValueType.INT32,
                     gguf.GGUFValueType.UINT64, gguf.GGUFValueType.INT64}
    for key, value in required.items():
        field = reader.fields.get(key)
        if field is None or field.contents() != value:
            raise ValueError(f'incompatible projector metadata: {key}')
        if isinstance(value, str):
            type_ok = field.types == [gguf.GGUFValueType.STRING]
        elif isinstance(value, bool):
            type_ok = field.types == [gguf.GGUFValueType.BOOL]
        elif isinstance(value, int):
            type_ok = len(field.types) == 1 and field.types[0] in integer_types
        elif key.endswith('is_deepstack_layers'):
            type_ok = len(field.types) == 2 and field.types[0] == gguf.GGUFValueType.ARRAY and field.types[1] in {gguf.GGUFValueType.UINT8, gguf.GGUFValueType.UINT16, gguf.GGUFValueType.UINT32, gguf.GGUFValueType.UINT64, gguf.GGUFValueType.BOOL}
        else:
            type_ok = len(field.types) == 2 and field.types[0] == gguf.GGUFValueType.ARRAY and field.types[1] in {gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.FLOAT64}
        if not type_ok:
            raise ValueError(f'incompatible projector metadata type: {key}')
    key = 'clip.vision.attention.layer_norm_epsilon'
    field = reader.fields.get(key)
    if field is None or field.types not in ([gguf.GGUFValueType.FLOAT32], [gguf.GGUFValueType.FLOAT64]) or np.float32(field.contents()) != np.float32(1e-6):
        raise ValueError(f'incompatible projector metadata: {key}')
    specs = tensor_contract()
    found = {t.name for t in reader.tensors}
    if len(reader.tensors) != 334 or found != set(specs):
        missing = sorted(set(specs) - found)
        extra = sorted(found - set(specs))
        raise ValueError(f'expected exactly 334 named tensors; missing={missing}, extra={extra}')
    for t in reader.tensors:
        shape, dtype = specs[t.name]
        accepted = (Q.F16, Q.BF16) if allow_f16 and dtype == Q.BF16 else (dtype,)
        if tuple(t.shape) != shape or t.tensor_type not in accepted:
            raise ValueError(f'incompatible tensor {t.name}: shape={tuple(t.shape)}, '
                             f'dtype={t.tensor_type.name}; expected shape={shape}, '
                             f'dtype={"F16/BF16" if len(accepted) == 2 else dtype.name}')
        if t.n_bytes != int(np.prod(shape)) * (4 if dtype == Q.F32 else 2):
            raise ValueError(f'invalid encoded tensor size: {t.name}')
    return Counter(t.tensor_type.name for t in reader.tensors)


def metadata_fields(reader):
    return {k: f for k, f in reader.fields.items() if not k.startswith('GGUF.')}


def discard_mapped_pages(*readers):
    # Linux can drop clean mapping pages after each tensor without changing the file.
    # This bounds residency as well as conversion buffers for the 931 MB sidecar.
    for reader in readers:
        mapping = getattr(reader.data, '_mmap', None)
        if mapping is not None and hasattr(mapping, 'madvise'):
            mapping.madvise(mmap.MADV_DONTNEED)


def converted_matrix(tensor):
    source = tensor.data.reshape(-1)
    output = np.empty(source.size * 2, dtype=np.uint8)
    for start in range(0, source.size, 1 << 20):
        end = min(source.size, start + (1 << 20))
        encoded = gguf.quantize(source[start:end].astype(np.float32), Q.BF16)
        output[start * 2:end * 2] = encoded.reshape(-1)
    return output


def verify_copy(source, output):
    """Reopen the result and verify every metadata field and every tensor byte."""
    counts = validate(output, allow_f16=False)
    a, b = metadata_fields(source), metadata_fields(output)
    if list(a) != list(b):
        raise ValueError('metadata names/order changed')
    for key, field in a.items():
        expected = 32 if key == 'general.file_type' else field.contents()
        if field.types != b[key].types or expected != b[key].contents():
            raise ValueError(f'metadata changed unexpectedly: {key}')
    if source.alignment != output.alignment:
        raise ValueError('alignment changed')
    for a, b in zip(source.tensors, output.tensors):
        if a.name != b.name or tuple(a.shape) != tuple(b.shape):
            raise ValueError(f'tensor name/order/shape changed: {a.name}')
        if a.tensor_type == Q.F16:
            x = a.data.reshape(-1)
            y = b.data.reshape(-1)
            for start in range(0, x.size, 1 << 20):
                end = min(x.size, start + (1 << 20))
                expected = gguf.quantize(x[start:end].astype(np.float32), Q.BF16)
                if not np.array_equal(expected.reshape(-1), y[start * 2:end * 2]):
                    raise ValueError(f'BF16 numerical verification failed: {a.name}')
        elif not np.array_equal(a.data.view(np.uint8), b.data.view(np.uint8)):
            raise ValueError(f'unchanged tensor byte verification failed: {a.name}')
        discard_mapped_pages(source, output)
    return counts


def convert(source_path, output_path):
    source_path, output_path = Path(source_path), Path(output_path)
    if source_path.resolve() == output_path.resolve():
        raise ValueError('input and output must be different paths')
    if os.path.lexists(output_path):
        raise FileExistsError(f'output already exists: {output_path}')
    reader = gguf.GGUFReader(source_path, 'r')
    counts = validate(reader, allow_f16=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + output_path.name + '.',
                                      suffix='.partial.gguf', dir=output_path.parent)
    os.close(fd)
    writer = None
    try:
        writer = gguf.GGUFWriter(temporary, 'clip', endianess=reader.endianess)
        writer.data_alignment = reader.alignment
        writer.kv_data = [{}]  # Re-add architecture too, preserving original field order/type.
        for key, field in metadata_fields(reader).items():
            value = 32 if key == 'general.file_type' else field.contents()
            writer.add_key_value(key, value, field.types[0],
                                 sub_type=field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
        for t in reader.tensors:
            target_type = Q.BF16 if t.tensor_type == Q.F16 else t.tensor_type
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype,
                                   t.n_bytes, raw_dtype=target_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for t in reader.tensors:
            data = converted_matrix(t) if t.tensor_type == Q.F16 else t.data
            writer.write_tensor_data(data, tensor_endianess=reader.endianess)
            del data
            discard_mapped_pages(reader)
        writer.close()
        writer = None
        verified = gguf.GGUFReader(temporary, 'r')
        final_counts = verify_copy(reader, verified)
        # link() is atomic and refuses any existing output, including symlinks.
        os.link(temporary, output_path)
        print(f'Converted {counts.get("F16", 0)} F16 matrices to BF16; '
              f'verified {final_counts["BF16"]} BF16 and {final_counts["F32"]} '
              f'byte-preserved F32 tensors (334 total).')
        print(f'Saved: {output_path}')
    finally:
        if writer is not None:
            writer.close()
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', type=Path, help='existing F16/BF16 Qwen3.8-27B mmproj GGUF')
    parser.add_argument('output', type=Path, nargs='?', help='new BF16 GGUF path (must not exist)')
    parser.add_argument('--check-only', action='store_true', help='require the input already satisfies Gufo BF16/F32 vision contract')
    args = parser.parse_args()
    try:
        if args.check_only:
            if args.output is not None:
                parser.error('--check-only does not take an output path')
            counts = validate(gguf.GGUFReader(args.input, 'r'), allow_f16=False)
            print(f'Gufo projector check passed: {counts["BF16"]} BF16 matrices, '
                  f'{counts["F32"]} F32 tensors, 334 exact names/shapes, output width 5120.')
        else:
            if args.output is None:
                parser.error('conversion requires an output path')
            convert(args.input, args.output)
    except (ValueError, OSError, KeyError, AssertionError) as exc:
        print(f'Projector conversion failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
