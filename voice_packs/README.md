# Bundled ZeroTTS voices

ZeroTTS voice-pack directories in `zerotts/` are loaded automatically by the
API. Each pack must contain `voice.npz`; `meta.json`, `voice.bin`, and
`preview.wav` are optional companion files.

Runtime-installed packs are stored separately under `models/zerotts/voices`.
The `models/` directory is intentionally excluded from Git and Docker builds.
