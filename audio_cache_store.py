"""Optional durable audio cache in a private Hugging Face Dataset.

RAM and local files stay the hot cache. Hub writes finish before returning new
audio; a rebuild only needs to download the requested files, not the dataset.
"""

from __future__ import annotations

import logging
import hashlib
from functools import lru_cache
import os
from pathlib import Path
import tempfile
import time


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=256)
def _file_digest(path: str, size: int, modified: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_digest(path: Path) -> str:
    """Portable asset identity, memoized until the local asset changes."""
    stat = path.stat()
    return _file_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def remote_name(namespace: str, path: Path) -> str:
    # Hash filenames avoid storing input text and shard large caches on the Hub.
    return f"audio/{namespace}/{path.name[:2]}/{path.name}"


def _token():
    return os.getenv("HF_AUDIO_CACHE_TOKEN") or os.getenv("HF_TOKEN") or None


def _failure_details(exc: Exception) -> str:
    """Expose actionable HTTP metadata without logging bodies or credentials."""
    details = [type(exc).__name__]
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    # requests.Response is false for 4xx/5xx; do not use its truth value here.
    if isinstance(status, int):
        details.append(f"HTTP {status}")
        hint = {
            401: "check cache token authentication",
            403: "check cache token write permission",
            404: "check cache repository and token access",
            409: "concurrent commit conflict",
            429: "Hub rate limit reached",
        }.get(status)
        if hint:
            details.append(hint)
        elif 500 <= status <= 599:
            details.append("Hub server error")
        headers = getattr(response, "headers", {})
        retry_after = headers.get("Retry-After", "")
        if isinstance(retry_after, str) and retry_after.isascii() and retry_after.isdecimal():
            details.append(f"retry after {retry_after[:10]} seconds")
    return "; ".join(details)


def atomic_write(path: Path, data: bytes) -> None:
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_audio(path: Path, namespace: str) -> bytes:
    """Read local storage first, then restore one remote entry on a miss."""
    try:
        return path.read_bytes()
    except OSError:
        pass
    repo = os.getenv("HF_AUDIO_CACHE_REPO")
    if repo:
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import EntryNotFoundError

            try:
                downloaded = hf_hub_download(
                    repo_id=repo, repo_type="dataset", filename=remote_name(namespace, path),
                    token=_token(), etag_timeout=10,
                )
            except EntryNotFoundError:
                raise FileNotFoundError(path) from None
            data = Path(downloaded).read_bytes()
            try:
                atomic_write(path, data)
            except OSError:
                LOGGER.warning("Could not restore local %s audio cache", namespace)
            return data
        except FileNotFoundError:
            raise
        except Exception as exc:
            # Never include response bodies/credentials in application logs.
            LOGGER.warning("Remote %s cache read failed (%s); generating audio", namespace, _failure_details(exc))
    raise FileNotFoundError(path)


def backup_audio(path: Path, namespace: str, data: bytes) -> bool:
    """Write through to the Hub, retrying transient errors before returning.

    Outages must not prevent speech delivery. A failed backup is logged and the
    local file remains available in this container. Failed backups must be
    recovered before rebuilding a Space with ephemeral storage.
    """
    repo = os.getenv("HF_AUDIO_CACHE_REPO")
    if not repo:
        return False
    for attempt in range(3):
        try:
            from huggingface_hub import HfApi

            HfApi(token=_token()).upload_file(
                repo_id=repo, repo_type="dataset", path_in_repo=remote_name(namespace, path),
                path_or_fileobj=data, commit_message=f"Cache {namespace} audio",
            )
            return True
        except Exception as exc:
            failure = _failure_details(exc)
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    LOGGER.warning("Audio backup failed for %s (%s); only the local copy is available", namespace, failure)
    return False


def local_cache_files():
    """Yield only generated WAV/MP3 files for migration during deployment."""
    zero_root = Path(os.getenv("ZEROTTS_DATA_DIR", str(ROOT / "models/zerotts")))
    roots = {
        "edge": Path(os.getenv("EDGE_AUDIO_CACHE_DIR", str(ROOT / "models/cache"))),
        "zerotts": Path(os.getenv("ZEROTTS_AUDIO_CACHE_DIR", str(zero_root / "audio_cache"))),
        "gwen": Path(os.getenv("GWEN_AUDIO_CACHE_DIR", str(ROOT / "models/gwen/audio_cache"))),
        "vieneu": Path(os.getenv("VIENEU_AUDIO_CACHE_DIR", str(ROOT / "models/vieneu/audio_cache"))),
        "vtts": Path(os.getenv("VTTS_AUDIO_CACHE_DIR", str(
            Path(os.getenv("VTTS_MODEL_DIR", str(ROOT / "models/vtts"))) / "audio_cache"
        ))),
    }
    for provider, directory in roots.items():
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.is_symlink() or path.suffix not in {".wav", ".mp3"}:
                continue
            relative = path.relative_to(directory)
            if len(relative.parts) == 1:
                namespace = provider
            elif len(relative.parts) == 2 and relative.parts[0] in {"encoded", "tempo", "voice_color"}:
                namespace = "mp3" if relative.parts[0] == "encoded" else relative.parts[0]
            else:
                continue
            yield remote_name(namespace, path), path
