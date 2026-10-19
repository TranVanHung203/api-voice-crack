"""Prepare or upload this API to a Gradio/ZeroGPU or Docker Space."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    "Dockerfile", ".dockerignore", "README.md", "requirements.txt", "LICENSE", "gpl-3.0.txt",
    "server.py", "zerotts_backend.py", "zerotts_suffix.py", "tts_delivery.py", "vtts_backend.py",
    "DEPLOY_HUGGINGFACE.md", "vieneu_backend.py", "gwen_backend.py", "gwen_worker.py", "requirements-gwen.txt", "setup_gwen.ps1",
    "hf_app.py", "requirements-gradio.txt", "requirements-hidden-suffix.txt", "packages.txt",
)


def deployment_files() -> list[Path]:
    files = [ROOT / name for name in ROOT_FILES]
    files.extend(p for p in (ROOT / "src").rglob("*") if p.is_file() and p.suffix in {".py", ".typed"})
    files.extend(p for p in (ROOT / "voice_packs").rglob("*") if p.is_file() and p.suffix in {".npz", ".bin", ".json", ".md"})
    files.extend((ROOT / "voice_packs" / "gwen" / "data" / "ref_audio").glob("*.wav"))
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing deployment files: {missing}")
    return sorted(files)


def deployment_payload(files: list[Path], sdk: str) -> dict[str, bytes]:
    payload = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in files}
    if sdk == "gradio":
        payload.pop("Dockerfile")
        # The Gradio builder mounts only this file at /tmp/requirements.txt
        # during dependency installation; sibling -r includes are unavailable.
        requirements = (ROOT / "requirements-gradio.txt").read_text(encoding="utf-8")
        suffix_requirements = (ROOT / "requirements-hidden-suffix.txt").read_text(encoding="utf-8")
        requirements = requirements.replace("-r requirements-hidden-suffix.txt", suffix_requirements.strip())
        if any(line.strip().startswith(("-r ", "--requirement ")) for line in requirements.splitlines()):
            raise ValueError("Gradio deployment requirements must not reference sibling files")
        payload["requirements.txt"] = requirements.encode("utf-8")
    else:
        readme = payload["README.md"].decode("utf-8")
        body = readme.split("---", 2)[2]
        payload["README.md"] = ("---\ntitle: Vietnamese TTS API\nemoji: 🎙️\ncolorFrom: yellow\ncolorTo: blue\nsdk: docker\napp_port: 7860\n---" + body).encode("utf-8")
    return payload


def prepare_archive(payload: dict[str, bytes]) -> Path:
    archive = ROOT / "build" / "huggingface-space.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive, "w", ZIP_DEFLATED) as zip_file:
        for name, data in payload.items():
            zip_file.writestr(name, data)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", help="owner/space-name; defaults to your-account/edge-tts-api")
    parser.add_argument("--private", action="store_true", help="Create a private Space")
    parser.add_argument("--sdk", choices=["gradio", "docker"], default="gradio")
    parser.add_argument("--account", help="Select an account from locally saved credentials")
    parser.add_argument("--prepare-only", action="store_true", help="Build the ZIP without logging in or uploading")
    args = parser.parse_args()
    files = deployment_files()
    payload = deployment_payload(files, args.sdk)
    archive = prepare_archive(payload)
    print(f"Prepared {len(payload)} files ({args.sdk}): {archive}")
    if args.prepare_only:
        return

    from huggingface_hub import HfApi, CommitOperationAdd, get_token
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    if not get_token():
        raise SystemExit("Log in locally first: hf auth login. Do not put your token in chat or source files.")
    api = HfApi()
    if args.account:
        from huggingface_hub.utils._auth import get_stored_tokens
        for saved_token in get_stored_tokens().values():
            candidate = HfApi(token=saved_token)
            try:
                if candidate.whoami()["name"] == args.account:
                    api = candidate
                    break
            except HfHubHTTPError:
                continue
        else:
            raise SystemExit(f"No valid locally saved credential for {args.account}")
    account = api.whoami()["name"]
    repo_id = args.repo_id or f"{account}/edge-tts-api"
    if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repo_id):
        raise SystemExit("repo-id must have the form owner/space-name")
    try:
        info = api.space_info(repo_id)
    except RepositoryNotFoundError:
        try:
            api.create_repo(
                repo_id, repo_type="space", space_sdk=args.sdk, private=args.private,
                space_hardware="zero-a10g" if args.sdk == "gradio" else "cpu-basic",
            )
        except HfHubHTTPError as exc:
            if exc.response is not None and exc.response.status_code == 402:
                raise SystemExit(
                    "Hugging Face rejected Space creation under this account's current plan/eligibility. "
                    "No code was uploaded. Free ZeroGPU hosting requires a verified account older than 30 days."
                ) from None
            raise
        info = api.space_info(repo_id)
    if info.sdk != args.sdk:
        raise SystemExit(f"{repo_id} uses {info.sdk}, not {args.sdk}; choose a different Space")
    api.create_commit(
        repo_id=repo_id,
        repo_type="space",
        commit_message=f"Deploy Vietnamese TTS API using {args.sdk}",
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=data) for name, data in payload.items()],
    )
    print(f"Uploaded. Build/runtime status: https://huggingface.co/spaces/{repo_id}")
    print("When the Space is Running, open its App URL and check /health and /docs.")


if __name__ == "__main__":
    main()
