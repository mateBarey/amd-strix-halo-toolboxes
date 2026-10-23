#!/usr/bin/env python3
"""Require real text and image completions from the selected local Gufo server."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import urllib.request
import zlib


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('gufo_profile', ROOT / 'toolboxes/gufo-nix/gufo_profile.py')
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


def synthetic_image():
    """A valid 96x96 RGB PNG; generated locally without image dependencies."""
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    pixels = (b'\x00' + b'\xff\x00\x00' * 96) * 96
    png = (b'\x89PNG\r\n\x1a\n'
           + chunk(b'IHDR', struct.pack('>IIBBBBB', 96, 96, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))
    messages = [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png).decode()}},
        {'type': 'text', 'text': 'Describe the visible contents and dominant color of this image.'},
    ]}]
    return messages, hashlib.sha256(png).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8081/v1')
    parser.add_argument('--model', default='qwen38-27b-uncensored')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', type=Path, help='Actual PNG/JPEG; otherwise send a generated solid-red PNG')
    parser.add_argument('--timeout', type=profile.positive, default=300)
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f'Existing report preserved: {args.output}', file=sys.stderr)
        return 1
    report = {'schema': 'gufo-inference-check.v1', 'status': 'failed',
              'created_at': datetime.now(timezone.utc).isoformat(), 'model': args.model,
              'checks': [], 'scope': 'Short text and image inference; not a throughput or full-context benchmark.'}
    workload = 'models'
    try:
        base_url = profile.endpoint(args.base_url, False)
        report['base_url'] = base_url
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), profile.NoRedirect())
        headers = {}
        if os.environ.get('OPENAI_API_KEY'):
            headers['Authorization'] = 'Bearer ' + os.environ['OPENAI_API_KEY']
        with profile.open_request(opener, urllib.request.Request(base_url + '/models', headers=headers), args.timeout) as response:
            catalog = json.load(response)
        if not any(item.get('id') == args.model for item in catalog.get('data', [])):
            raise profile.ProfileError(f'Expected model is absent from /models: {args.model}')
        image_messages, digest = profile.image_message(args.image) if args.image else synthetic_image()
        workloads = [
            ('text', [{'role': 'user', 'content': 'Respond with a short greeting.'}], None),
            ('image', image_messages, digest),
        ]
        for workload, messages, image_digest in workloads:
            sample = profile.measure(opener, base_url, headers, args.model, messages, False, 64, args.timeout)
            tokens = sample['usage'].get('completion_tokens')
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
                raise profile.ProfileError(f'{workload} request did not report generated completion tokens')
            sample.update(workload=workload, image_sha256=image_digest)
            report['checks'].append(sample)
        report['status'] = 'passed'
    except (profile.ProfileError, OSError, http.client.HTTPException, ValueError, TypeError, KeyError, AttributeError) as error:
        report.update(failed_workload=workload, error=str(error))
        print(f'Gufo {workload} inference check failed: {error}', file=sys.stderr)
    try:
        profile.publish(args.output, report)
    except OSError as error:
        print(f'Could not save inference report: {error}', file=sys.stderr)
        return 1
    if report['status'] == 'passed':
        print(f'Text and image inference checks passed. Report: {args.output}')
        return 0
    print(f'Failure report: {args.output}', file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
