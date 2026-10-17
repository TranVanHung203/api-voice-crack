# Deploy to a free Hugging Face Space

The default deployment uses **Gradio SDK + ZeroGPU hardware**. Hugging Face
allows up to two free ZeroGPU Spaces for personal accounts in good standing,
with verified email and an account older than 30 days. A newer account can
be rejected even when its token has write permission. No subscription or
paid hardware upgrade is selected by the deployment script.

The application still serves the existing FastAPI routes: `/tts`,
`/zerotts/synthesize`, `/voices`, `/health` and `/docs`. `/` contains a small
Gradio landing page. Edge and ZeroTTS inference remain CPU workloads;
`hf_app.py` registers ZeroGPU compatibility following Hugging Face's official
CPU/API-proxy hosting recipe, without dispatching normal API calls to GPU
workers or using their GPU quota.

## Upload from PowerShell

```powershell
cd F:\edge-tts
py -3.11 -c "from huggingface_hub import login; login()"
py -3.11 tools/deploy_huggingface.py --sdk gradio
```

Enter credentials locally, never into chat or source files. The script defaults
to `YOUR_ACCOUNT/edge-tts-api`, creates a free ZeroGPU Gradio Space if eligible,
and uploads only application files and bundled voice packs. Local models,
audio caches, recordings, environment files and virtual environments are excluded.

To use a specific locally authenticated account:

```powershell
py -3.11 tools/deploy_huggingface.py --sdk gradio --account tranvanhung203 --repo-id tranvanhung203/edge-tts-api
```

For a ZIP, run `py -3.11 tools/deploy_huggingface.py --prepare-only`.
Extract `build/huggingface-space.zip` and upload its contents to the Space
repository root, preserving the `src` and `voice_packs` directories.
Do not upload the ZIP itself as application source.

The remote `requirements.txt` is generated from `requirements-gradio.txt`;
local requirements and the optional Docker deployment remain separate.
Both profiles explicitly install `aiohttp`, `certifi`, `tabulate`, and
`typing-extensions` for the vendored Edge client/API, plus
the dependencies in `requirements-hidden-suffix.txt` for recognition and alignment.
The Gradio upload expands that include into a self-contained `requirements.txt`,
because the Hugging Face dependency-install stage cannot read sibling files.
Docker retains the local include. The deployment
package includes that requirements file, `zerotts_suffix.py`, and `tts_delivery.py`; local virtual
environments are not uploaded.
Both dependency profiles include PyAV for MP3 delivery. `packages.txt` and the
Docker image also install FFmpeg as a fallback. Add `format=mp3` to a ZeroTTS
GET URL for the smaller response. GET/HEAD support ETag validation, byte ranges,
and private device caching; POST audio remains `no-store`.
The Space uses Python 3.10.13, Gradio 5.49.1 and PyTorch 2.8.0. ZeroTTS weights
are preloaded at build time into the default Hugging Face cache; the first TTS
request still loads ONNX sessions into RAM.

## Verify the running API

Use the **App URL** (`https://?hf.space`), not the outer Space repository page.
`/health` returns `{"status":"ok"}` without loading models or calling an
external service. `/docs` provides the API form. A liveness check does not
prove that every TTS backend can generate audio; test synthesis as well.

Send this JSON to `POST /tts` for the bundled chi tham voice:

```json
{"provider":"zerotts","voice_id":"VpYLlCQEgcoHgycJstNP","text":"H?y c?ng kh?m ph? nh?","eoa_extra_frames":1}
```

For Edge:

```json
{"provider":"edge","voice":"vi-VN-HoaiMyNeural","text":"Xin ch?o."}
```

The Space may sleep while idle. After wake-up, allow extra time for model
initialization. Generated cache can be lost on container restart; both hot RAM
caches are bounded to 64 MiB. Run a single worker to share model memory.

## Optional Docker profile

The vendored Edge TTS handshake implementation (`drm.py` and `constants.py`)
is updated from upstream commit
[`360fa8e`](https://github.com/rany2/edge-tts/commit/360fa8e1de354e1dda74c09bd695887d6a44df74)
to support Microsoft's v2 handshake. The API still depends on Microsoft
accepting requests; retry handling cannot guarantee external service uptime.

`py -3.11 tools/deploy_huggingface.py --sdk docker --repo-id YOUR_ACCOUNT/YOUR_SPACE`
prepares/uploads the Docker profile. New Docker Spaces currently require a
paid plan; this profile is not the selected free deployment. The Docker image
uses CPU-only PyTorch and includes pinned ZeroTTS weights.

Official references:
[Free ZeroGPU eligibility](https://huggingface.co/docs/hub/en/spaces-zerogpu),
[CPU/API-proxy recipe](https://github.com/huggingface/skills/blob/main/skills/huggingface-spaces/references/inference-providers.md),
[Space configuration](https://huggingface.co/docs/hub/en/spaces-config-reference).
