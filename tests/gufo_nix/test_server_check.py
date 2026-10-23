"""Exercise actual text and image HTTP/SSE requests made by the startup check."""
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/gufo-nix/check-server.py'
MODEL = 'qwen38-27b-uncensored'


class ServerCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.report = Path(self.temp.name) / 'check.json'
        self.requests = []
        self.mode = 'ok'
        case = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'data': [{'id': MODEL}]}).encode())

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                case.requests.append(body)
                is_image = isinstance(body['messages'][0]['content'], list)
                if is_image and case.mode == 'image_error':
                    self.send_response(500)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                usage = {'prompt_tokens': 12, 'completion_tokens': 0 if case.mode == 'zero_tokens' else 4}
                events = [
                    {'choices': [{'delta': {'content': 'A red square.' if is_image else 'Hello.'}, 'finish_reason': None}]},
                    {'choices': [{'delta': {}, 'finish_reason': 'stop'}], 'usage': usage},
                ]
                for event in events:
                    self.wfile.write(('data: ' + json.dumps(event) + '\n\n').encode())
                if case.mode != 'truncated':
                    self.wfile.write(b'data: [DONE]\n\n')

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def run_check(self):
        return subprocess.run(['python3', str(SCRIPT), '--base-url',
                               f'http://127.0.0.1:{self.server.server_port}/v1',
                               '--model', MODEL, '--output', str(self.report)],
                              capture_output=True, text=True, timeout=10)

    def test_success_requires_text_and_actual_png_request(self):
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(self.report.read_text())
        self.assertEqual(report['status'], 'passed')
        self.assertEqual([x['workload'] for x in report['checks']], ['text', 'image'])
        self.assertEqual(len(self.requests), 2)
        for request in self.requests:
            self.assertEqual(request['model'], MODEL)
            self.assertFalse(request['cache_prompt'])
        image_url = self.requests[1]['messages'][0]['content'][0]['image_url']['url']
        self.assertTrue(image_url.startswith('data:image/png;base64,'))
        png = base64.b64decode(image_url.split(',', 1)[1])
        self.assertTrue(png.startswith(b'\x89PNG\r\n\x1a\n'))
        self.assertGreater(len(png), 60)

    def test_image_failure_is_not_reported_as_text_and_image_success(self):
        self.mode = 'image_error'
        result = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        report = json.loads(self.report.read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['failed_workload'], 'image')
        self.assertEqual(len(report['checks']), 1)

    def test_truncated_stream_and_zero_tokens_fail(self):
        for mode in ('truncated', 'zero_tokens'):
            with self.subTest(mode=mode):
                self.mode = mode
                self.report = Path(self.temp.name) / (mode + '.json')
                result = self.run_check()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(json.loads(self.report.read_text())['status'], 'failed')

    def test_existing_report_is_preserved_without_sending_requests(self):
        self.report.write_text('existing result')
        result = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.report.read_text(), 'existing result')
        self.assertEqual(self.requests, [])


if __name__ == '__main__':
    unittest.main()
