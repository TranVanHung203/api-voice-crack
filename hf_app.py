"""Gradio-hosted entry point retaining the existing FastAPI routes."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("EDGE_WARMUP", "0")
os.environ.setdefault("PORT", "7860")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("ZEROTTS_THREADS", "2")
os.environ.setdefault("ZEROTTS_REVISION", "c2bfbd67dc648cac455077333f7cf5c18a2e3bb4")
os.environ.setdefault("ZEROTTS_CACHE_DIR", str(Path.home() / ".cache" / "huggingface" / "hub"))

# Hugging Face's CPU/proxy recipe for free ZeroGPU-hosted Gradio apps:
# https://github.com/huggingface/skills/blob/main/skills/huggingface-spaces/references/inference-providers.md
# Register compatibility without dispatching CPU inference to a GPU worker.
if os.getenv("SPACE_ID"):
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_compatibility():
        return None

import gradio as gr
import server

with gr.Blocks(title="Vietnamese TTS API") as demo:
    gr.Markdown("# Vietnamese TTS API\n\n[Thử API và xem tài liệu](/docs) · [Kiểm tra trạng thái](/health)")
    gr.Markdown("Dùng `/tts` để tạo giọng Edge hoặc ZeroTTS. Space có thể ngủ khi không được sử dụng.")

def api_app_kwargs():
    # Let Gradio own its root/config routes; retain the application's API routes.
    builtin_paths = {"/", "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}
    return {
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "routes": [r for r in server.app.router.routes if getattr(r, "path", None) not in builtin_paths],
        "lifespan": server.lifespan,
        "middleware": server.app.user_middleware,
    }


app = gr.routes.App.create_app(demo, app_kwargs=api_app_kwargs())

# ZeroGPU may reserve PORT for its internal service; the public app uses 7860.
os.environ["PORT"] = "7860"

if __name__ == "__main__":
    # launch() runs the ZeroGPU startup hook; direct uvicorn.run() skips it.
    demo.launch(
        server_name="0.0.0.0", server_port=7860,
        app_kwargs=api_app_kwargs(), ssr_mode=False,
    )
