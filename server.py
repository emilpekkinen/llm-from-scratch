"""Tiny web server for chatting with the NumPy GPT in a browser (stdlib only).

    python3 server.py            # then open http://localhost:8765
"""
import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from model import GPT
from tokenizer import BPETokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


class ModelHolder:
    """Loads the checkpoint and reloads it whenever train.py writes a new one."""

    def __init__(self, out_dir):
        self.ckpt = os.path.join(out_dir, "ckpt.npz")
        self.tok_path = os.path.join(out_dir, "tokenizer.json")
        self.lock = threading.Lock()  # forward() caches activations, so one request at a time
        self.mtime = None
        self.model = self.tok = None
        self.extra = {}

    def get(self):
        mtime = os.path.getmtime(self.ckpt)
        if mtime != self.mtime:
            self.model, self.extra = GPT.load(self.ckpt)
            self.tok = BPETokenizer.load(self.tok_path)
            self.mtime = mtime
            print(f"loaded {self.ckpt} (step {self.extra.get('step')}, val loss {self.extra.get('val_loss'):.4f})")
        return self.model, self.tok

    def info(self):
        model, _ = self.get()
        return {**model.cfg.__dict__, "params": model.num_params(),
                "step": int(self.extra.get("step", -1)), "val_loss": float(self.extra.get("val_loss", float("nan")))}


class Handler(BaseHTTPRequestHandler):
    holder: ModelHolder = None

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "static", "index.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/info":
            try:
                self._json(self.holder.info())
            except FileNotFoundError:
                self._json({"error": "no checkpoint yet - run train.py first"}, 503)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/generate":
            return self.send_error(404)
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        prompt = req.get("prompt") or "\n"
        max_new = max(1, min(int(req.get("max_new_tokens", 200)), 1000))
        temperature = float(req.get("temperature", 0.8))
        top_k = int(req.get("top_k", 40)) or None
        stop_at_speech = bool(req.get("stop_at_speech", True))

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()  # HTTP/1.0 with no Content-Length: the body ends when the connection closes

        def send(text):
            self.wfile.write(text.encode())
            self.wfile.flush()

        try:
            with self.holder.lock:
                model, tok = self.holder.get()
                ids, sent = [], ""
                for nxt in model.generate_stream(tok.encode(prompt), max_new, temperature, top_k):
                    ids.append(nxt)
                    text = tok.decode(ids)
                    if text.endswith("�"):  # partial UTF-8 character, wait for more bytes
                        continue
                    # A blank line ends a speech; stop there for a chat-like turn
                    if stop_at_speech and text.strip() and "\n\n" in text.lstrip("\n"):
                        text = text[:len(text) - len(text.lstrip("\n"))] + text.lstrip("\n").split("\n\n")[0]
                        send(text[len(sent):])
                        break
                    send(text[len(sent):])
                    sent = text
        except (BrokenPipeError, ConnectionResetError):
            pass  # client pressed "stop"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="out")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    Handler.holder = ModelHolder(args.out_dir)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as e:
        if e.errno != 48 and e.errno != 98:  # EADDRINUSE on macOS / Linux
            raise
        raise SystemExit(f"Port {args.port} is already in use - is the server already running?\n"
                         f"Stop it with:  kill $(lsof -t -iTCP:{args.port} -sTCP:LISTEN)\n"
                         f"or pick another port:  python3 server.py --port {args.port + 1}")
    print(f"NumPyGPT running at http://localhost:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
