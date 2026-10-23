import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / 'toolboxes/gufo-nix/gufo_profile.py'


class Handler(BaseHTTPRequestHandler):
    requests_seen = []
    mode = 'ok'

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({'data': [{'id': 'wrong' if self.mode == 'wrong_model' else 'test-model', 'context_length': 262144}]}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.requests_seen.append(payload)
        if self.mode == 'http_error':
            self.send_error(503, 'GPU unavailable')
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Transfer-Encoding', 'chunked')
        self.end_headers()
        parts = [b': keepalive\r\n\r\n', b'data: {"choices":[{"delta":{"content":"caf',
                 b'\xc3\xa9"},"finish_reason":null}]}\r\n\r\n',
                 b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
                 b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":1,"gufo":{"prefill_ms":12}}}\n\n']
        if self.mode == 'sse_error':
            parts = [b'data: {"error":{"message":"device lost"}}\n\n']
        if self.mode == 'malformed':
            parts = [b'data: {bad json}\n\n']
        if self.mode == 'no_usage':
            parts = parts[:-1]
        if self.mode == 'broken_transfer':
            self.wfile.write(b'FF\r\ndata: incomplete')
            self.wfile.flush()
            self.close_connection = True
            return
        if self.mode != 'truncated':
            parts.append(b'data: [DONE]\n\n')
        for part in parts:
            self.wfile.write(f'{len(part):X}\r\n'.encode() + part + b'\r\n')
            self.wfile.flush()
        self.wfile.write(b'0\r\n\r\n')


class ProfileTests(unittest.TestCase):
    def setUp(self):
        Handler.mode = 'ok'
        Handler.requests_seen = []
        self.tmp = tempfile.TemporaryDirectory()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.output = Path(self.tmp.name) / 'report.json'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(CLI), '--base-url',
            f'http://127.0.0.1:{self.server.server_port}/v1', '--model', 'test-model',
            '--repetitions', '1', '--output', str(self.output), *args], capture_output=True, text=True)

    def test_streaming_image_usage_metadata_and_cache_labels(self):
        image = Path(self.tmp.name) / 'image.png'
        image.write_bytes(b'\x89PNG\r\n\x1a\nfixture')
        result = self.run_cli('--image', str(image))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(self.output.read_text())
        self.assertEqual(report['recommendation']['status'], 'baseline_only')
        self.assertEqual(report['server']['model_ids'], ['test-model'])
        self.assertIn('created_at', report)
        self.assertEqual(len(report['samples']), 4)
        self.assertEqual({s['workload'] for s in report['samples']}, {'text', 'image'})
        self.assertEqual({s['cache_mode'] for s in report['samples']}, {'uncached_prompt', 'warmed_repeat'})
        for sample in report['samples']:
            self.assertEqual(sample['completion_sha256'], hashlib.sha256('café'.encode()).hexdigest())
            self.assertEqual(sample['usage']['completion_tokens'], 1)
            self.assertGreaterEqual(sample['wall_ms'], sample['client_ttft_ms'])
        self.assertEqual([p['cache_prompt'] for p in Handler.requests_seen], [False, True, True] * 2)
        image_payload = Handler.requests_seen[3]['messages'][0]['content'][0]['image_url']['url']
        self.assertEqual(image_payload, 'data:image/png;base64,' + base64.b64encode(image.read_bytes()).decode())
        self.assertNotIn('café', self.output.read_text())

    def test_http_and_stream_errors_leave_no_success_report(self):
        for mode in ('http_error', 'sse_error', 'truncated', 'malformed', 'no_usage', 'broken_transfer', 'wrong_model'):
            with self.subTest(mode=mode):
                Handler.mode = mode
                result = self.run_cli()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('gufo-profile:', result.stderr)
                self.assertFalse(self.output.exists())

    def test_existing_report_is_never_overwritten_or_sent(self):
        self.output.write_text('keep me')
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.output.read_text(), 'keep me')
        self.assertEqual(Handler.requests_seen, [])

    def test_remote_endpoint_requires_explicit_opt_in(self):
        result = subprocess.run([sys.executable, str(CLI), '--base-url', 'https://example.com/v1',
            '--model', 'x', '--output', str(self.output)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--allow-remote', result.stderr)
