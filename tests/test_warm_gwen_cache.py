"""Batch extraction and HTTP warm-up must never speak templates or fake successes."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit

import pytest

from tools.extract_textvoice_for_gwen import extract
from tools import warm_gwen_cache as warm


def test_extracts_only_concrete_voice_rows_and_preserves_text():
    document = """
### KD-01. Bài 1
#### helper.dart
##### TextVoice, câu hỏi và phản hồi
| Dòng | Biến / ngữ cảnh | Loại | Nội dung nguyên văn / mẫu |
| --- | --- | --- | --- |
| 5 | prompt | Câu cố định | Xin chào bạn. |
| 6, 9 | repeat | Câu cố định | Xin chào bạn. |
| 10 | title | Câu cố định | Ngày 15 tháng 5<br>Lôi sẽ kiểm tra môn toán |
| 11 | number | Mẫu động | Có ${value} quả táo |
| 12 | dynamic | Câu cố định | Xin chào $name |
| 13 | label | Từ / nhãn | Ba |
| 14 | hintVoice | Từ / nhãn | hintVoice |
<details><summary>Nhãn và dữ liệu bổ trợ (1 mục)</summary>
| Dòng | Biến / ngữ cảnh | Loại | Nội dung nguyên văn / mẫu |
| --- | --- | --- | --- |
| 20 | ui | Câu cố định | Không tự đọc nút này |
</details>
"""
    result = extract(document)
    assert result["texts"] == ["Xin chào bạn.", "Ngày 15 tháng 5\nLôi sẽ kiểm tra môn toán", "Ba"]
    assert len(result["sources"]["Xin chào bạn."]) == 2
    assert result["counts"]["unresolved_template"] == 2
    assert result["counts"]["supplementary_labels_or_data"] == 1
    assert result["counts"]["technical_or_debug_text"] == 1


@pytest.fixture
def api():
    state = {"supported": True, "calls": [], "retry_once": False, "invalid_audio": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path == "/gwen/voices":
                if not state["supported"]:
                    self.send_error(404)
                    return
                data = json.dumps({"voices": [{"voice_id": "yen_nhi"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            else:
                params = parse_qs(parsed.query)
                state["calls"].append(params)
                if state["retry_once"]:
                    state["retry_once"] = False
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    data = b'{"detail":"warming up"}'
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                data = b"ID3" + b"test-audio" * 10
                self.send_response(200)
                self.send_header("Content-Type", "text/html" if state["invalid_audio"] else "audio/mpeg")
                self.send_header("ETag", '"cached-audio"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", state
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def arguments(tmp_path, base):
    texts = tmp_path / "texts.json"
    texts.write_text(json.dumps(["Xin chào.", "Có 5 quả táo.", "Xin chào."], ensure_ascii=False), encoding="utf-8")
    return ["--texts", str(texts), "--output", str(tmp_path / "output"), "--base-url", base, "--delay", "0"]


def test_utf8_calls_are_deduplicated_saved_and_resumable(api, tmp_path):
    base, state = api
    args = arguments(tmp_path, base)
    assert warm.main(args) == 0
    assert len(state["calls"]) == 2
    assert state["calls"][0] == {"provider": ["gwen"], "voice": ["yen_nhi"], "text": ["Xin chào."], "format": ["mp3"]}
    assert len(list((tmp_path / "output/audio").glob("*.mp3"))) == 2
    assert warm.main(args) == 0 and len(state["calls"]) == 2
    assert warm.main(args + ["--rewarm", "--limit", "1"]) == 0
    assert len(state["calls"]) == 3
    assert warm.main(args) == 0 and len(state["calls"]) == 4


def test_missing_gwen_stops_before_any_tts_request(api, tmp_path):
    base, state = api
    state["supported"] = False
    assert warm.main(arguments(tmp_path, base)) == 2
    assert state["calls"] == []
    report = json.loads((tmp_path / "output/preflight.json").read_text())
    assert report["status"] == "blocked"
    assert not (tmp_path / "output/progress.jsonl").exists()


def test_transient_failure_retries_then_records_success(api, tmp_path, monkeypatch):
    base, state = api
    state["retry_once"] = True
    monkeypatch.setattr(warm.time, "sleep", lambda _: None)
    assert warm.main(arguments(tmp_path, base) + ["--limit", "1"]) == 0
    assert len(state["calls"]) == 2
    receipts = [json.loads(line) for line in (tmp_path / "output/progress.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(receipts) == 1 and receipts[0]["status"] == "ok" and receipts[0]["attempt"] == 2


def test_dry_run_does_not_use_network(tmp_path, monkeypatch):
    monkeypatch.setattr(warm.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("Dry run called network"))
    assert warm.main(arguments(tmp_path, "https://example.test") + ["--dry-run"]) == 0


def test_wrong_content_is_never_saved_as_success(api, tmp_path):
    base, state = api
    state["invalid_audio"] = True
    assert warm.main(arguments(tmp_path, base) + ["--limit", "1"]) == 1
    assert list((tmp_path / "output/audio").iterdir()) == []
    assert warm.previous_successes(tmp_path / "output/progress.jsonl") == set()


def test_dynamic_input_is_rejected_and_partial_receipts_do_not_break_resume(tmp_path):
    text_file = tmp_path / "texts.json"
    text_file.write_text('["Hello ${name}"]', encoding="utf-8")
    with pytest.raises(warm.WarmupError, match="templates"):
        warm.load_texts(text_file)
    progress = tmp_path / "progress.jsonl"
    progress.write_text('{"broken":', encoding="utf-8")
    warm.append_receipt(progress, {"id": "good", "status": "ok"})
    assert warm.previous_successes(progress) == {"good"}


def test_generated_text_list_is_concrete_and_unique():
    texts = warm.load_texts(warm.DEFAULT_TEXTS)
    assert len(texts) == 1070
    assert texts[0].startswith("Xin chào các bạn")
