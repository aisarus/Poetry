"""Tests of actual HTTP JSON exchange, using a local test server not a real model."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import poetry_lab as lab


class OllamaStub(BaseHTTPRequestHandler):
    seen = []
    fail = False

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        payload = json.loads(self.rfile.read(n).decode('utf-8'))
        type(self).seen.append((self.path, payload))
        if type(self).fail:
            self.send_response(503)
            data = b'{"error":"model not loaded"}'
        else:
            self.send_response(200)
            data = json.dumps({'message': {'role': 'assistant', 'content': 'Привет, мир!'}},
                              ensure_ascii=False).encode('utf-8')
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class TestOllamaHttp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), OllamaStub)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=2)
        cls.server.server_close()

    def setUp(self):
        OllamaStub.seen.clear()
        OllamaStub.fail = False

    def test_unicode_roundtrip(self):
        url = f'http://127.0.0.1:{self.server.server_address[1]}'
        messages = [{'role': 'user', 'content': 'Сочини про счастье'}]
        answer = lab.ollama_chat(messages, 'qwen2.5:7b', url, 0.4)
        self.assertEqual(answer, 'Привет, мир!')
        path, payload = OllamaStub.seen[0]
        self.assertEqual(path, '/api/chat')
        self.assertEqual(payload['messages'], messages)
        self.assertEqual(payload['model'], 'qwen2.5:7b')
        self.assertFalse(payload['stream'])
        self.assertEqual(payload['options']['temperature'], 0.4)

    def test_http_error_is_useful(self):
        OllamaStub.fail = True
        url = f'http://127.0.0.1:{self.server.server_address[1]}'
        with self.assertRaisesRegex(RuntimeError, 'Ollama HTTP 503'):
            lab.ollama_chat([{'role': 'user', 'content': 'тест'}], 'dummy', url)


if __name__ == '__main__':
    unittest.main()
