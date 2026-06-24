"""Run the GradLab backend locally as a tiny HTTP server (no AWS needed).

    python scripts/run_local.py        # serves on http://localhost:8000

Point the frontend at it by setting API_BASE = "http://localhost:8000" in
frontend/config.js.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import handler as backend  # noqa: E402


class H(BaseHTTPRequestHandler):
    def _send(self, resp):
        self.send_response(resp["statusCode"])
        for k, v in resp["headers"].items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(resp["body"].encode())

    def do_OPTIONS(self):
        self._send(backend.handler({"httpMethod": "OPTIONS", "path": self.path}))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode() if length else "{}"
        self._send(backend.handler({"httpMethod": "POST", "path": self.path, "body": body}))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"GradLab backend on http://localhost:{port}  (Ctrl-C to stop)")
    HTTPServer(("0.0.0.0", port), H).serve_forever()
