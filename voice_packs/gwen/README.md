# Gwen-TTS bundled reference speakers

Source: https://github.com/ggroup-ai-lab/gwen-tts

Upstream commit: `f0f24d8636596dc635323e02f41bee42bb64385e`.
`data/ref_info.json` and the recordings in `data/ref_audio/` were copied from
that commit. The model/repository declares the MIT license; see the upstream
README. The recordings are reference inputs, not model weights. Only speakers
listed in `ref_info.json` are exposed by the API.

Model: https://huggingface.co/g-group-ai-lab/gwen-tts-0.6B

Default model revision: `a83e1f08656d48217a83f0ad422d1610d5b5c963`.
Model weights download separately on the first synthesis request. The default
speaker is `yen_nhi` (Yến Nhi); both IDs and display names work.
