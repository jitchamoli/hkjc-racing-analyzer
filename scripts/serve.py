#!/usr/bin/env python3
"""Local dashboard server with refresh API.

Serves results/racing_dashboard.html at localhost:8080 and exposes:
  GET  /              → serves racing_dashboard.html
  GET  /api/status    → {'refreshing': bool, 'last_refresh': iso_ts, 'race_date': 'YYYY/MM/DD'}
  POST /api/refresh   → kicks off generate_predictions.py in background
                        Header: X-Race-Date: YYYY/MM/DD  (defaults to today)
  GET  /api/refresh/log → tail of last refresh's stdout

The dashboard's refresh button hits /api/refresh then polls /api/status.

Usage:
    python3 -m scripts.serve                    # Port 8080
    python3 -m scripts.serve --port 9000
    python3 -m scripts.serve --open             # Open browser on start
"""

import argparse
import http.server
import json
import os
import re
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'


class State:
    """Mutable server-wide state for refresh tracking."""
    refreshing = False
    lock = threading.Lock()
    last_refresh_ts = None
    last_refresh_date = None
    last_refresh_log = ""
    last_refresh_error = None


def run_refresh(race_date: str) -> None:
    """Execute generate_predictions.py for the given date."""
    State.last_refresh_log = ""
    State.last_refresh_error = None
    try:
        proc = subprocess.run(
            ['python3', 'generate_predictions.py', race_date],
            cwd=str(ROOT), capture_output=True, text=True, timeout=600,
        )
        State.last_refresh_log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            State.last_refresh_error = f"exit code {proc.returncode}"
    except Exception as e:
        State.last_refresh_error = str(e)
        State.last_refresh_log = str(e)
    finally:
        State.last_refresh_ts = datetime.now().isoformat(timespec='seconds')
        State.last_refresh_date = race_date
        State.refreshing = False


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(RESULTS), **kwargs)

    def log_message(self, fmt, *args):
        # Quiet default access log except for API and errors
        msg = fmt % args
        if '/api/' in msg or ' 4' in msg[:30] or ' 5' in msg[:30]:
            sys.stderr.write(f"[{self.log_date_time_string()}] {msg}\n")

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/' or self.path == '/dashboard':
            self.path = '/racing_dashboard.html'
        if self.path == '/api/status':
            self._send_json({
                'refreshing': State.refreshing,
                'last_refresh': State.last_refresh_ts,
                'race_date': State.last_refresh_date,
                'error': State.last_refresh_error,
            })
            return
        if self.path == '/api/refresh/log':
            self._send_json({'log': State.last_refresh_log[-4000:]})
            return
        # Disable caching for the dashboard so refreshes appear instantly
        if self.path.endswith('.html') or self.path.endswith('.json'):
            try:
                fp = Path(str(RESULTS)) / self.path.lstrip('/')
                data = fp.read_bytes()
                self.send_response(200)
                ct = 'text/html; charset=utf-8' if self.path.endswith('.html') else 'application/json'
                self.send_header('Content-Type', ct)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            except FileNotFoundError:
                pass
        return super().do_GET()

    def do_POST(self):
        if self.path == '/api/refresh':
            race_date = (self.headers.get('X-Race-Date')
                         or datetime.now().strftime('%Y/%m/%d'))
            # Validate before passing to subprocess — even with shell=False,
            # an unvalidated string lets any loopback caller spawn unlimited
            # background python processes.
            if not re.fullmatch(r'\d{4}/\d{2}/\d{2}', race_date):
                self._send_json({'error': 'invalid race_date, expected YYYY/MM/DD'}, status=400)
                return
            with State.lock:
                if State.refreshing:
                    self._send_json({'error': 'already refreshing'}, status=409)
                    return
                State.refreshing = True
            threading.Thread(target=run_refresh, args=(race_date,), daemon=True).start()
            self._send_json({'status': 'started', 'race_date': race_date})
            return
        self.send_response(404)
        self.end_headers()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--open', action='store_true', help='Open browser on start')
    args = parser.parse_args()

    server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Dashboard server running at {url}")
    print("Endpoints:")
    print(f"  Dashboard:  {url}")
    print(f"  Status:     {url}api/status")
    print(f"  Refresh:    POST {url}api/refresh (header X-Race-Date: YYYY/MM/DD)")
    print("Press Ctrl+C to stop.\n")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == '__main__':
    main()
