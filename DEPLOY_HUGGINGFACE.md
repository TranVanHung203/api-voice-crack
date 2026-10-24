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
and uploads only application files and bundled voice packs to the Space. Local
generated audio is backed up separately to a private Dataset (see below).
Local models, recordings, environment files and virtual environments are excluded.

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

Gwen is provisioned automatically in `.venv-gwen` on Gradio Space startup.
This installs CPU PyTorch and Gwen's own transformers version in isolation,
checks SDK imports, and downloads the pinned model before launching the API.
Allow several minutes on the first start and follow the runtime logs for
`Gwen: isolated runtime and model files ready`. Docker does this at build time.
`/gwen/voices` only lists bundled recordings; confirm synthesis with a real
`/tts?provider=gwen&voice=yen_nhi&text=Xin%20chao&format=mp3` request before
starting `warm_gwen_cache.ps1`.

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
initialization on cache misses. Hot RAM caches are bounded; run a single worker
to share model memory.

## Audio cache that survives redeployment

The deployment command now automatically creates a **private Dataset** named
`YOUR_ACCOUNT/edge-tts-api-audio-cache`, backs up existing local audio, and
configures these settings on the Space before uploading application code:

- Variable `HF_AUDIO_CACHE_REPO`: the private Dataset ID.
- Secret `HF_AUDIO_CACHE_TOKEN`: the selected account's locally saved write token.

Use `--cache-repo YOUR_ACCOUNT/tts-audio-cache` to reuse a particular private
Dataset. The script refuses to upload audio to a public Dataset. The token needs
write access to the Dataset and permission to configure the Space. Credentials
are stored as Space secrets, never in the source archive or logs. `--prepare-only`
only builds the source ZIP; it does not create or configure remote storage.
For manual ZIP deployment, create the private Dataset and add the variable and
secret above in Space Settings yourself.

Edge, Gwen, ZeroTTS, VieNeu and V-TTS check local disk and then the Dataset before
generating new audio; RAM caches stay in front where supported. A fresh container downloads only the requested audio file.
Generated WAVs, encoded MP3s and Gwen tempo variants are uploaded before the
first response returns. This adds upload time to a new cache entry; subsequent
RAM/disk hits do not contact the Hub. The Dataset is separate from the Space,
so rebuilding or replacing application code does not delete the backed-up audio.
Local runs keep their existing disk cache unless `HF_AUDIO_CACHE_REPO` is set.

The deployment script also migrates existing `.wav`/`.mp3` cache files from
`models/cache`, `models/gwen/audio_cache`, `models/zerotts/audio_cache` and
`models/vieneu/audio_cache` and `models/vtts/audio_cache`, respecting the corresponding `*_AUDIO_CACHE_DIR`
overrides and `ZEROTTS_DATA_DIR`. Model weights, logs, reference clips and
temporary files are not copied into the Dataset. Cache filenames contain hashes;
the audio itself stays private. Dataset storage quotas and Hub API limits apply.

Keep the same Dataset ID on later deploys. ZeroTTS now uses portable content
fingerprints instead of absolute paths and modification times. Its older cache
keys are invalidated once; new entries survive relocation. Keep ZeroTTS pinned
to the same Hub commit to reuse cache before downloading weights. Changing the
text, voice assets or synthesis settings can correctly require new audio.
V-TTS cache lookup uses the reference clip or registered voice profile content;
keep those assets separately so they can be supplied again after a rebuild.
Set `VTTS_AUDIO_CACHE_VERSION` to a new value if updating V-TTS's default model.

Hub failures are logged and uploads retry three times. Speech delivery continues
using the local copy if backup fails; that entry is **not protected against a
container rebuild** until successfully backed up. Audio that existed only inside
an older Space container cannot be recovered after that container is deleted;
enable this feature before generating audio that needs to survive future deploys.

### Recover downloaded ZeroTTS warmup audio

If warmup downloaded audio successfully but the Space logged backup failures,
the MP3s under `models/zerotts_cache_warmup/VpYLlCQEgcoHgycJstNP/audio` can
restore the cache without synthesizing again:

```powershell
.\.venv-rvc\Scripts\python.exe tools\backup_zerotts_cache.py
```

Use `--dry-run` to validate downloads and prepare recovery files without uploading.
The script reads the deployed code, voice files and public ZeroTTS settings to
reproduce the Space's cache keys. It preserves existing WAVs, decodes missing WAVs
from downloaded MP3s, and uploads the original MP3 bytes under the encoding cache
keys. A recovered WAV contains decoded MP3 audio; it does not restore the original
uncompressed waveform. Uploads use batches of at most 100 files. All audio checksums
are checked against Hub metadata before `recovery/backup-report.json` records
`complete: true`. The private Dataset also receives a recovery manifest.

This command does not deploy, restart the Space, delete remote audio, or call TTS.
It can be rerun after an interruption and skips files already on the Hub. Normal
warmup's `progress.jsonl` records successful downloads, not durable backup status;
the recovery report is the backup confirmation. No application update is needed
when the deployed backend already supports the Dataset cache.

Official references: [persisting Space data in a Dataset](https://huggingface.co/docs/huggingface_hub/en/guides/upload),
[Space secrets and variables](https://huggingface.co/docs/huggingface_hub/en/guides/manage-spaces).

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
