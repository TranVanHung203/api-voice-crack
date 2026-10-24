"""Make an offline listening comparison from one original Gwen/Yen Nhi WAV."""

from __future__ import annotations

import argparse
import hashlib
import html
from io import BytesIO
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tts_delivery import change_wav_tempo, encode_mp3

DEFAULT_TEXT = "Xin chào các bạn, một bàn tay có năm ngón tay đúng không nhỉ, cùng đếm với mình nha."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-wav", type=Path, help="Use a recorded original instead of calling the Space")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--base-url", default="https://tranvanhung203-edge-tts-api.hf.space")
    parser.add_argument("--output", type=Path, default=ROOT / "samples/gwen-rate")
    args = parser.parse_args()
    if args.source_wav:
        source = args.source_wav.read_bytes()
    else:
        url = args.base_url.rstrip("/") + "/tts?" + urllib.parse.urlencode({
            "provider": "gwen", "voice": "yen_nhi", "text": args.text, "format": "wav",
        })
        print("Downloading one original Gwen/Yen Nhi WAV", flush=True)
        with urllib.request.urlopen(url, timeout=1200) as response:
            source = response.read()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "text.txt").write_text(args.text, encoding="utf-8")
    rows, cards = [], []
    for name, label, tempo in [("original", "Bản gốc — không chỉnh tốc độ", 1.0),
                                ("minus-10", "Chậm hơn 10%", .9), ("minus-20", "Chậm hơn 20%", .8)]:
        audio = change_wav_tempo(source, tempo, ROOT / "models/gwen/rate-demo")
        basename = "yen-nhi-" + name
        (args.output / (basename + ".wav")).write_bytes(audio)
        (args.output / (basename + ".mp3")).write_bytes(encode_mp3(audio, ROOT / "models/gwen/rate-demo"))
        with wave.open(BytesIO(audio), "rb") as wav:
            seconds = wav.getnframes() / wav.getframerate()
            sample_rate = wav.getframerate()
        rows.append({"name": name, "tempo": tempo, "seconds": round(seconds, 3),
                     "sample_rate": sample_rate, "file": basename + ".mp3"})
        cards.append(f'<section><h2>{html.escape(label)} · {seconds:.2f} giây</h2>'
                     f'<audio controls preload="metadata" src="{basename}.mp3"></audio>'
                     f'<p><a href="{basename}.wav" download>Tải WAV</a> · '
                     f'<a href="{basename}.mp3" download>Tải MP3</a></p></section>')
    report = {"text": args.text, "source_sha256": hashlib.sha256(source).hexdigest(), "variants": rows}
    (args.output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    page = ('<!doctype html><html lang="vi"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Nghe thử Yến Nhi: tốc độ gốc và chậm hơn</title>'
            '<style>body{font:17px/1.6 system-ui;max-width:760px;margin:40px auto;padding:0 20px;'
            'color:#182338;background:#f4f6fa}h1{font-size:28px}h2{font-size:19px}'
            'section{background:white;border:1px solid #dce2ec;border-radius:12px;padding:14px 22px;'
            'margin:18px 0}audio{width:100%}a{color:#164da3}</style>'
            '<h1>Yến Nhi — so sánh tốc độ đọc</h1>'
            '<p>Cả ba mẫu dùng cùng một bản thu gốc. Hai bản chậm được xử lý bằng FFmpeg atempo.</p>'
            f'<blockquote>{html.escape(args.text)}</blockquote>' + "".join(cards) +
            '<p>Nghe cùng âm lượng để so sánh âm sắc, độ mượt và độ tự nhiên. '
            'Giữ cao độ không đảm bảo chất giọng hoàn toàn giống bản gốc.</p></html>')
    (args.output / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=True, indent=2))
    print("Open:", args.output.resolve() / "index.html")


if __name__ == "__main__":
    main()
