#!/usr/bin/env python
"""배치 추론 HTTP 서버 (stdlib http.server 기반).

사용 예:
    python scripts/serve.py --ckpt runs/10b/ckpt_final_full.pt \
        --tokenizer tokenizer/ --host 0.0.0.0 --port 8000

요청:
    curl -X POST http://localhost:8000/generate \
        -H 'Content-Type: application/json' \
        -d '{"prompt": "옛날 옛적에", "max_new_tokens": 128, "temperature": 0.8, "top_k": 50}'

동시에 들어온 요청은 DynamicBatcher가 짧은 시간창 동안 모아 배치로 처리합니다.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

from src.serve import load_engine, DynamicBatcher  # noqa: E402


def make_handler(batcher: DynamicBatcher):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # 조용히
            pass

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"status": "ok"})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/generate":
                self._json(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(n) or b"{}")
                prompt = payload.get("prompt", "")
                text = batcher.submit(
                    prompt,
                    max_new_tokens=int(payload.get("max_new_tokens", 128)),
                    temperature=float(payload.get("temperature", 0.8)),
                    top_k=payload.get("top_k", 50),
                )
                self._json(200, {"text": text})
            except Exception as e:  # noqa
                self._json(500, {"error": str(e)})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="consolidated .pt (ckpt_final_full.pt)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--max-wait-ms", type=int, default=20)
    args = ap.parse_args()

    print(f"모델 로드 중: {args.ckpt}")
    engine = load_engine(args.ckpt, args.tokenizer)
    batcher = DynamicBatcher(engine, max_batch=args.max_batch, max_wait_ms=args.max_wait_ms)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(batcher))
    print(f"서빙 시작: http://{args.host}:{args.port}  (POST /generate, GET /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        batcher.stop()
        server.server_close()


if __name__ == "__main__":
    main()
