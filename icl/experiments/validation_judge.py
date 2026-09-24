"""Local HTTP adapter for a real Qwen3-8B judge during GPU integration tests.

The production client protocol and prompts are unchanged. This small serial
server avoids needing a second serving stack just for the integration test.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from threading import Thread


@contextmanager
def judge_server(model, tokenizer):
    import torch
    from icl.experiments.singlepass import render_chat
    from icl.experiments.telemetry import emit

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                text = render_chat(tokenizer, request["messages"], True, enable_thinking=False)
                ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(model.device)
                with torch.inference_mode():
                    output = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=request["max_tokens"],
                                            do_sample=False, pad_token_id=tokenizer.eos_token_id)
                answer = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
                emit("judge_request", request=request, response=answer)
                response = {"choices": [{"message": {"content": answer}}]}
                self.send_response(200)
            except Exception as exc:
                emit("judge_error", error=repr(exc))
                response = {"error": repr(exc)}
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        worker.join()
        server.server_close()
