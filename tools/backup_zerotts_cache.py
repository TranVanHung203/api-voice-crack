"""Recover downloaded ZeroTTS audio into the live Space's private Dataset cache.

No synthesis, Space deployment, or deletion is performed. Existing WAVs win;
missing WAVs are decoded from the downloaded MP3. Original MP3 bytes are retained.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import inspect
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from audio_cache_store import _failure_details, atomic_write, remote_name
from tools.warm_zerotts_cache import DEFAULT_OUTPUT, VOICE_ID
from zerotts_backend import ZeroTTSBackend, _effective_sampling_seed, _effective_suffix_config
from zerotts_suffix import HiddenSuffixConfig


def successful_records(output: Path) -> list[dict]:
    records = {}
    for line in (output / "progress.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("status") == "reset":
            records.clear()
        elif item.get("status") == "error":
            records.pop(item["id"], None)
        elif item.get("status") == "ok":
            records[item["id"]] = item
    if not records:
        raise ValueError("No successful audio downloads found")
    for item in records.values():
        if not re.fullmatch(r"[a-f0-9]{64}", item["id"]):
            raise ValueError("Invalid download ID")
        if item.get("payload") != {"voice_id": VOICE_ID, "text": item["text"], "format": "mp3"}:
            raise ValueError("Only this voice's default-parameter MP3 receipts are supported")
        path = output / "audio" / (item["id"] + ".mp3")
        data = path.read_bytes()
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"Downloaded audio checksum mismatch: {path.name}")
        if not (data.startswith(b"ID3") or (len(data) > 1 and data[0] == 255 and data[1] & 224 == 224)):
            raise ValueError(f"Invalid MP3: {path.name}")
    return list(records.values())


def entry_defaults(source: str) -> dict[str, str]:
    """Read literal environment defaults, never execute code from the Hub."""
    result = {}
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and ast.unparse(node.func) == "os.environ.setdefault" and len(node.args) == 2
                and all(isinstance(arg, ast.Constant) and isinstance(arg.value, str) for arg in node.args)):
            name, value = (arg.value for arg in node.args)
            if name.startswith("ZEROTTS_"):
                result[name] = value
    return result


@contextmanager
def deployment_environment(values: dict):
    saved = {name: value for name, value in os.environ.items() if name.startswith("ZEROTTS_")}
    try:
        for name in saved:
            del os.environ[name]
        os.environ.update({name: value for name, value in values.items() if name.startswith("ZEROTTS_")})
        yield
    finally:
        for name in list(os.environ):
            if name.startswith("ZEROTTS_"):
                del os.environ[name]
        os.environ.update(saved)


def cache_key(backend: ZeroTTSBackend, text: str) -> str:
    text = text.strip()
    bound = inspect.signature(backend.synthesize_wav).bind(text, VOICE_ID)
    bound.apply_defaults()
    options = dict(bound.arguments)
    options.pop("text")
    options.pop("voice_id")
    seed = _effective_sampling_seed(text, VOICE_ID, options.pop("seed"))
    if seed is not None:
        options["seed"] = seed
    config = _effective_suffix_config(text, VOICE_ID, HiddenSuffixConfig.from_env())
    if config.enabled:
        options["hidden_suffix_config"] = config.cache_options()
    return backend._audio_cache_key(text, VOICE_ID, options)


def decode_mp3(data: bytes) -> bytes:
    import soundfile as sf

    samples, rate = sf.read(BytesIO(data), dtype="int16", always_2d=True)
    if not samples.size or samples.shape[1] != 1:
        raise ValueError("Expected a nonempty mono ZeroTTS MP3")
    output = BytesIO()
    sf.write(output, samples, rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def mp3_name(wav: bytes) -> str:
    key = hashlib.sha256(b"mp3-mono-64k-v2\0" + wav).hexdigest()
    return remote_name("mp3", Path(key + ".mp3"))


def file_matches(info, data: bytes) -> bool:
    if info.size != len(data):
        return False
    if info.lfs is not None:
        return info.lfs.sha256 == hashlib.sha256(data).hexdigest()
    blob = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    return info.blob_id == hashlib.sha1(blob).hexdigest()


def upload_missing(api, repo: str, entries: dict[str, Path], existing: set[str], batch_size: int) -> int:
    from huggingface_hub import CommitOperationAdd

    pending = [(name, path) for name, path in entries.items() if name not in existing]
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        for attempt in range(5):
            try:
                api.create_commit(
                    repo_id=repo, repo_type="dataset", commit_message="Recover downloaded ZeroTTS audio cache",
                    operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in batch],
                )
                break
            except Exception as exc:
                if attempt == 4:
                    raise RuntimeError(_failure_details(exc)) from None
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
                if status in {400, 401, 403, 404, 413, 422}:
                    raise RuntimeError(_failure_details(exc)) from None
                delay = 5 * (attempt + 1)
                retry_after = getattr(response, "headers", {}).get("Retry-After", "")
                if retry_after.isdecimal():
                    delay = min(60, max(delay, int(retry_after)))
                print(f"Upload retry: {_failure_details(exc)}; wait {delay}s", flush=True)
                time.sleep(delay)
        print(f"Uploaded {min(offset + batch_size, len(pending))}/{len(pending)} missing files", flush=True)
    return len(pending)


def verify_entries(api, repo: str, entries: dict[str, Path]) -> None:
    names = list(entries)
    for offset in range(0, len(names), 100):
        batch = names[offset:offset + 100]
        infos = {item.path: item for item in api.get_paths_info(repo, batch, repo_type="dataset")}
        for name in batch:
            if name not in infos or not file_matches(infos[name], entries[name].read_bytes()):
                raise ValueError(f"Remote checksum verification failed: {name}")
        print(f"Verified {min(offset + 100, len(names))}/{len(names)} remote files", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space-id", default="tranvanhung203/edge-tts-api")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 100:
        parser.error("--batch-size must be 1..100")
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    records = successful_records(args.output)
    print(f"Validated {len(records)} local MP3 files for {VOICE_ID}", flush=True)
    variables = api.get_space_variables(args.space_id)
    repo = variables.get("HF_AUDIO_CACHE_REPO")
    if repo is None or not api.repo_info(repo.value, repo_type="dataset").private:
        raise ValueError("Space must use an existing private HF_AUDIO_CACHE_REPO")
    repo = repo.value
    download = lambda name: Path(hf_hub_download(args.space_id, name, repo_type="space"))
    for name in ("zerotts_backend.py", "zerotts_suffix.py", "tts_delivery.py"):
        if download(name).read_text(encoding="utf-8") != (ROOT / name).read_text(encoding="utf-8"):
            raise ValueError(f"Local {name} differs from deployed code; refuse to guess cache keys")
    settings = entry_defaults(download("hf_app.py").read_text(encoding="utf-8"))
    settings.update({name: value.value for name, value in variables.items() if name.startswith("ZEROTTS_")})
    if settings.get("ZEROTTS_MODEL_DIR") or not re.fullmatch(r"[a-fA-F0-9]{40}", settings.get("ZEROTTS_REVISION", "")):
        raise ValueError("Recovery requires the Space's pinned Hub model revision")
    recovery = args.output / "recovery"
    voice_root = recovery / "deployed-voices"
    # Isolate only the files actually deployed. Workstation preview WAVs and
    # separately installed packs must not change a Space's cache identity.
    filenames = api.list_repo_files(args.space_id, repo_type="space")
    prefix = f"voice_packs/zerotts/{VOICE_ID}/"
    deployed_names = []
    for name in filenames:
        if name.startswith(prefix):
            target = voice_root / VOICE_ID / Path(name).name
            atomic_write(target, download(name).read_bytes())
            deployed_names.append(target.name)
    if "voice.npz" not in deployed_names:
        raise ValueError("Voice pack is missing from deployed Space")
    local_names = {path.name for path in (voice_root / VOICE_ID).iterdir() if path.is_file()}
    if local_names != set(deployed_names):
        raise ValueError("Recovery voice folder contains extra files; use a clean output directory")

    class DeployedBackend(ZeroTTSBackend):
        @classmethod
        def _voice_roots(cls):
            return (("bundled", voice_root),)

    backend = DeployedBackend()
    with deployment_environment(settings):
        # This delivery change would otherwise be applied a second time to a
        # recovered MP3 that already includes it.
        import zerotts_backend as module
        if module._HANDSPAN_TEMPO != 1.0:
            raise ValueError("Non-default handspan delivery tempo requires a separate recovery mapping")
        jobs = [(item, cache_key(backend, item["text"])) for item in records]
    existing = set(api.list_repo_files(repo, repo_type="dataset"))
    matches = sum(remote_name("zerotts", Path(key + ".wav")) in existing for _, key in jobs)
    print(f"Dataset: {repo}; existing WAVs: {matches}; WAVs to restore: {len(jobs) - matches}", flush=True)

    def prepare(job):
        item, key = job
        wav_name = remote_name("zerotts", Path(key + ".wav"))
        source = args.output / "audio" / (item["id"] + ".mp3")
        if wav_name in existing:
            from huggingface_hub.errors import LocalEntryNotFoundError

            try:
                wav_path = Path(hf_hub_download(repo, wav_name, repo_type="dataset", local_files_only=True))
            except LocalEntryNotFoundError:
                wav_path = Path(hf_hub_download(repo, wav_name, repo_type="dataset"))
            wav = wav_path.read_bytes()
        else:
            wav = decode_mp3(source.read_bytes())
            wav_path = recovery / "wav" / (key + ".wav")
            atomic_write(wav_path, wav)
        return {"text": item["text"], "voice_id": VOICE_ID, "wav": wav_name,
                "mp3": mp3_name(wav), "original_mp3_sha256": item["sha256"],
                "wav_restored_from_mp3": wav_name not in existing}, wav_path, source

    entries = {}
    manifest = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (record, wav_path, mp3_path) in enumerate(pool.map(prepare, jobs), 1):
            entries[record["wav"]] = wav_path
            entries[record["mp3"]] = mp3_path
            manifest.append(record)
            if index % 100 == 0 or index == len(jobs):
                print(f"Prepared {index}/{len(jobs)} texts", flush=True)
    manifest_path = recovery / "manifest.json"
    atomic_write(manifest_path, json.dumps({"voice_id": VOICE_ID, "space": args.space_id,
                 "settings": settings, "texts": manifest}, ensure_ascii=False, indent=2).encode("utf-8"))
    if args.dry_run:
        print(f"Dry run: {len(set(entries) - existing)} files need upload; manifest: {manifest_path}")
        return 0
    uploaded = upload_missing(api, repo, entries, existing, args.batch_size)
    verify_entries(api, repo, entries)
    # Separate inventories (for example runtime variants) must not overwrite
    # the recovery manifest for the original 1,070 texts.
    inventory = json.dumps(sorted(record["wav"] for record in manifest)).encode("utf-8")
    manifest_id = hashlib.sha256(inventory).hexdigest()
    manifest_name = f"recovery/zerotts/{VOICE_ID}/manifests/{manifest_id}.json"
    upload_missing(api, repo, {manifest_name: manifest_path}, set(), 1)
    verify_entries(api, repo, {manifest_name: manifest_path})
    report = {"complete": True, "voice_id": VOICE_ID, "texts": len(records), "uploaded_files": uploaded,
              "verified_audio_files": len(entries), "restored_wavs": len(jobs) - matches, "dataset": repo}
    atomic_write(recovery / "backup-report.json", json.dumps(report, indent=2).encode("utf-8"))
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Hub exceptions can contain tokens/response bodies. Keep terminal logs safe.
        print(f"Backup incomplete: {_failure_details(exc) if hasattr(exc, 'response') else type(exc).__name__ + ': ' + str(exc)}", file=sys.stderr)
        raise SystemExit(1)
