# Bundled ZeroTTS voices

ZeroTTS voice-pack directories in `zerotts/` are loaded automatically by the
API. Each pack must contain `voice.npz`; `meta.json`, `voice.bin`, and
`preview.wav` are optional companion files.

Runtime-installed packs are stored separately under `models/zerotts/voices`.
The `models/` directory is intentionally excluded from Git and Docker builds.

`VpYLlCQEgcoHgycJstNP_child` copies the chi tham speaker files and is exposed
as a separate processed voice by `ZeroTTSBackend`. It reuses synthesis/cache
from `VpYLlCQEgcoHgycJstNP`, then raises pitch by three semitones with
resampling and tempo compensation. Vocal resonance follows the pitch shift;
the source speaking tempo is retained. This avoids the independent formant
estimation used in the first preview, which could add a rough/metallic texture.
It is a childlike delivery effect, not a recording or trained child speaker.
The original speaker's files and cache keys are unchanged. The effect settings
in `meta.json` describe the backend constants in `zerotts_backend.py`.

ZeroTTS 0.1.5 reads pack metadata with the platform's default text encoding.
Keep non-ASCII characters JSON-escaped (`json.dumps(..., ensure_ascii=True)`)
in bundled `meta.json` files so they also load under Windows ANSI code pages.
Escaping preserves the displayed Vietnamese names. `start_server.ps1` enables
Python UTF-8 mode for metadata loaded by other SDK components. For manual
startup on Windows, use `.venv-rvc\Scripts\python.exe -X utf8 -m uvicorn server:app`.

The registered male derivatives `quangminh_child` and `tiendat_child` use
`quangminh` (+5 semitones) and `tiendat` (+4 semitones), respectively. Their
`voice.npz` files copy the corresponding speaker from the pinned standard
model, and the backend applies the effect after source synthesis/cache lookup.
Their metadata stores the source revision and embedding checksum. These are
processed childlike versions of the original male speakers.

`quangvinh_child` is the Southern alternative, using public community speaker
`htSzITlVXGL3s4XwNqOv` (Quang Vinh), tagged `male`, `young`, `southern` in
the ZeroWeight Audio Studio public catalogue. Its original source embedding
is bundled too. The childlike effect raises pitch by four semitones and keeps
the source tempo, using the same processing as the accepted `tiendat_child`.
Metadata records catalogue provenance and content checksums. The original
Tiến Đạt and Quang Minh registrations and caches remain usable.
