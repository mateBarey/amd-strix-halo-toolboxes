#!/usr/bin/python3
"""Measure real requests to a running Gufo server without changing settings."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


class ProfileError(Exception):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProfileError('HTTP redirects are not followed; specify the final server URL')


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def endpoint(value, allow_remote):
    parsed = urllib.parse.urlsplit(value)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or
            parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ProfileError('base URL must be http(s), without credentials, query or fragment')
    local = parsed.hostname.lower() == 'localhost'
    try:
        local = local or ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        pass
    if not local and not allow_remote:
        raise ProfileError('non-loopback endpoints require explicit --allow-remote')
    return value.rstrip('/')


def open_request(opener, request, timeout):
    try:
        return opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        # Response bodies can echo prompts or credentials; do not print them.
        raise ProfileError(f'HTTP {error.code} {error.reason} from '
                           f'{urllib.parse.urlsplit(request.full_url).path}') from error
    except urllib.error.URLError as error:
        raise ProfileError(f'connection failed: {error.reason}') from error


def stream_events(response):
    data = []
    for raw in response:
        line = raw.decode('utf-8').rstrip('\r\n')
        if not line:
            if data:
                yield '\n'.join(data)
                data = []
        elif line.startswith('data:'):
            data.append(line[5:].lstrip(' '))
    if data:
        yield '\n'.join(data)


def measure(opener, base_url, headers, model, messages, cache_prompt, max_tokens, timeout):
    body = {'model': model, 'messages': messages, 'temperature': 0,
            'max_tokens': max_tokens, 'stream': True,
            'stream_options': {'include_usage': True}, 'cache_prompt': cache_prompt}
    request = urllib.request.Request(base_url + '/chat/completions',
        data=json.dumps(body).encode(), headers={**headers, 'Content-Type': 'application/json'})
    started = time.perf_counter()
    first = None
    completion = hashlib.sha256()
    usage = None
    finish_reason = None
    fingerprint = None
    done = False
    with open_request(opener, request, timeout) as response:
        if response.headers.get_content_type() != 'text/event-stream':
            raise ProfileError('server did not return text/event-stream')
        for event in stream_events(response):
            if event == '[DONE]':
                done = True
                break
            try:
                chunk = json.loads(event)
            except ValueError as error:
                raise ProfileError('malformed JSON in SSE response') from error
            if not isinstance(chunk, dict):
                raise ProfileError('SSE event is not a JSON object')
            if chunk.get('error'):
                raise ProfileError('server reported an SSE error (inspect the server log)')
            fingerprint = chunk.get('system_fingerprint', fingerprint)
            if chunk.get('usage') is not None:
                usage = chunk['usage']
            for choice in chunk.get('choices', []):
                delta = choice.get('delta') or {}
                content = delta.get('content')
                if isinstance(content, str) and content:
                    if first is None:
                        first = time.perf_counter()
                    completion.update(content.encode('utf-8'))
                if choice.get('finish_reason') is not None:
                    finish_reason = choice['finish_reason']
    elapsed = (time.perf_counter() - started) * 1000
    if not done or finish_reason is None:
        raise ProfileError('incomplete SSE response: missing [DONE] or finish_reason')
    if first is None:
        raise ProfileError('response had no text content; text TTFT cannot be measured')
    if not isinstance(usage, dict):
        raise ProfileError('response omitted requested usage metrics')
    return {'wall_ms': elapsed, 'client_ttft_ms': (first - started) * 1000,
            'completion_sha256': completion.hexdigest(), 'finish_reason': finish_reason,
            'usage': usage, 'system_fingerprint': fingerprint}


def image_message(path):
    data = path.read_bytes()
    if not data or len(data) > 15 * 1024 * 1024:
        raise ProfileError('image must be nonempty and at most 15 MiB before base64 encoding')
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        mime = 'image/png'
    elif data.startswith(b'\xff\xd8\xff'):
        mime = 'image/jpeg'
    else:
        raise ProfileError('image must be PNG or JPEG (checked by file signature)')
    url = f'data:{mime};base64,' + base64.b64encode(data).decode('ascii')
    return [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': url}},
        {'type': 'text', 'text': 'Describe the visible contents of this image precisely.'}]}], hashlib.sha256(data).hexdigest()


def publish(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.gufo-profile-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as output:
            json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        # Exclusive atomic publication, including races with another writer.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Profile actual text/image requests to an already running Gufo server. Measures a baseline; does not tune Q4 kernels or the vision encoder.')
    parser.add_argument('--base-url', default='http://127.0.0.1:8080/v1')
    parser.add_argument('--model', required=True, help='Exact served API model ID')
    parser.add_argument('--image', type=Path, help='Optional actual PNG/JPEG sent to the server')
    parser.add_argument('--repetitions', type=positive, default=3)
    parser.add_argument('--max-tokens', type=positive, default=128)
    parser.add_argument('--timeout', type=positive, default=600, help='HTTP socket timeout in seconds')
    parser.add_argument('--allow-remote', action='store_true', help='Explicitly permit sending workloads/image to a non-loopback endpoint')
    parser.add_argument('--output', required=True, type=Path, help='New JSON report; existing files are never overwritten')
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ProfileError(f'output already exists: {args.output}')
        base_url = endpoint(args.base_url, args.allow_remote)
        workloads = [('text', [{'role': 'user', 'content': 'Explain how a bounded request queue handles concurrent work. Give concrete examples.'}], None)]
        if args.image:
            messages, digest = image_message(args.image)
            workloads.append(('image', messages, digest))
        opener = urllib.request.build_opener(NoRedirect())
        headers = {}
        if os.environ.get('OPENAI_API_KEY'):
            headers['Authorization'] = 'Bearer ' + os.environ['OPENAI_API_KEY']
        with open_request(opener, urllib.request.Request(base_url + '/models', headers=headers), args.timeout) as response:
            catalog = json.load(response)
        models = catalog.get('data', [])
        selected = next((item for item in models if item.get('id') == args.model), None)
        if selected is None:
            raise ProfileError(f'requested model ID is not advertised by /models: {args.model}')
        samples = []
        summaries = []
        for workload, messages, digest in workloads:
            for mode, cache in [('uncached_prompt', False), ('warmed_repeat', True)]:
                if cache:
                    measure(opener, base_url, headers, args.model, messages, True, args.max_tokens, args.timeout)
                group = []
                for repetition in range(args.repetitions):
                    sample = measure(opener, base_url, headers, args.model, messages, cache, args.max_tokens, args.timeout)
                    sample.update(workload=workload, cache_mode=mode, repetition=repetition + 1, cache_prompt=cache)
                    samples.append(sample)
                    group.append(sample)
                summaries.append({'workload': workload, 'cache_mode': mode,
                    'image_sha256': digest, 'wall_ms_median': statistics.median(s['wall_ms'] for s in group),
                    'client_ttft_ms_median': statistics.median(s['client_ttft_ms'] for s in group)})
        report = {'schema': 'gufo-toolbox-profile.v1',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'server': {'model_ids': [m.get('id') for m in models], 'selected_model': selected,
                       'engine_version': 'not advertised; system_fingerprint retained per response'},
            'settings': {'temperature': 0, 'max_tokens': args.max_tokens, 'repetitions': args.repetitions},
            'samples': samples, 'summaries': summaries,
            'recommendation': {'status': 'baseline_only', 'settings': {}, 'reason': 'No candidate settings were changed or ranked.'},
            'limitations': ['Image TTFT includes image processing and text prefill; it is not encoder-only timing.',
                'Uncached prompt requests bypass prompt reuse, not OS/GPU caches or model loading.',
                'Warmed repeats include one unmeasured request first; cached token counts remain in usage.',
                'No hardware or GPU performance was inferred without measurement.']}
        publish(args.output, report)
        print(f'Profile saved: {args.output}')
        return 0
    except (ProfileError, OSError, http.client.HTTPException, ValueError, TypeError, KeyError, AttributeError) as error:
        print(f'gufo-profile: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
