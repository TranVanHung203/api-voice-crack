"""Verify fixed-voice POST requests, recovery, and trustworthy warm-up receipts."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from tools import warm_zerotts_cache as warm
from tools import warm_gwen_cache as runner


@pytest.fixture
def api():
    state = {"voice": warm.VOICE_ID, "calls": [], "fail_once": False, "invalid": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, data, content_type):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            assert self.path == "/zerotts/voices"
            self.send(200, json.dumps({"voices": [{"voice_id": state["voice"]}]}).encode(), "application/json")

        def do_POST(self):
            assert self.path == "/zerotts/synthesize"
            assert self.headers["Content-Type"] == "application/json"
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["calls"].append(payload)
            if state["fail_once"]:
                state["fail_once"] = False
                self.send(503, b'{"detail":"warming up"}', "application/json")
            elif state["invalid"]:
                self.send(200, b"<html>Space is starting</html>", "text/html")
            else:
                self.send(200, b"ID3" + b"audio-data" * 10, "audio/mpeg")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", state
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def arguments(tmp_path, base):
    texts = tmp_path / "texts.json"
    texts.write_text(json.dumps(["năm", "Xin chào.\nCùng đếm nhé!", "năm"], ensure_ascii=False), encoding="utf-8")
    return ["--texts", str(texts), "--output", str(tmp_path / "output"), "--base-url", base, "--delay", "0"]


def test_fixed_voice_utf8_post_and_resume(api, tmp_path):
    base, state = api
    args = arguments(tmp_path, base)
    assert warm.main(args) == 0
    assert state["calls"] == [
        {"voice_id": warm.VOICE_ID, "text": "năm", "format": "mp3"},
        {"voice_id": warm.VOICE_ID, "text": "Xin chào.\nCùng đếm nhé!", "format": "mp3"},
    ]
    assert len(list((tmp_path / "output/audio").glob("*.mp3"))) == 2
    assert warm.main(args) == 0
    assert len(state["calls"]) == 2
    assert warm.main(args + ["--rewarm", "--limit", "1"]) == 0
    assert len(state["calls"]) == 3
    assert warm.main(args) == 0
    assert len(state["calls"]) == 4


def test_wrong_voice_fails_before_synthesis(api, tmp_path):
    base, state = api
    state["voice"] = "another-voice"
    assert warm.main(arguments(tmp_path, base)) == 2
    assert not state["calls"]


def test_retry_then_save_once(api, tmp_path, monkeypatch):
    base, state = api
    state["fail_once"] = True
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    assert warm.main(arguments(tmp_path, base) + ["--limit", "1"]) == 0
    assert len(state["calls"]) == 2
    journal = tmp_path / "output/progress.jsonl"
    receipts = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert len(receipts) == 1
    assert receipts[0]["attempt"] == 2 and receipts[0]["status"] == "ok"


def test_invalid_response_is_not_a_success(api, tmp_path):
    base, state = api
    state["invalid"] = True
    assert warm.main(arguments(tmp_path, base) + ["--limit", "1"]) == 1
    assert not runner.previous_successes(tmp_path / "output/progress.jsonl")
    assert not list((tmp_path / "output/audio").iterdir())


def test_complete_dry_run_does_not_call_network(tmp_path, monkeypatch):
    monkeypatch.setattr(warm.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Dry run called the API"))
    output = tmp_path / "dry-run"
    assert warm.main(["--dry-run", "--output", str(output)]) == 0
    jobs = [json.loads(line) for line in (output / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(jobs) == 1070
    assert all(job["url"] == warm.DEFAULT_URL + "/zerotts/synthesize" for job in jobs)
    assert all(job["payload"]["voice_id"] == warm.VOICE_ID and "seed" not in job["payload"] for job in jobs)


def test_request_identity_changes_when_url_or_text_changes():
    job = warm.build_job(warm.DEFAULT_URL, "năm")
    assert job == warm.build_job(warm.DEFAULT_URL + "/", "năm")
    assert job["id"] != warm.build_job(warm.DEFAULT_URL, "sáu")["id"]
    assert job["id"] != warm.build_job("https://another.hf.space", "năm")["id"]
