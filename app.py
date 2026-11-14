"""
GCAEMMCD: AI-Powered Multimodal Cancer Diagnosis Workstation.
Production Application Entrypoint for Hugging Face Spaces & Cloud Deployment.
"""

import os
import sys

# Patch gradio_client bug where boolean in json schema breaks get_type
try:
    import gradio_client.utils
    _orig_get_type = gradio_client.utils.get_type
    def _safe_get_type(schema):
        if isinstance(schema, bool):
            return "boolean"
        return _orig_get_type(schema)
    gradio_client.utils.get_type = _safe_get_type
except Exception:
    pass

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import spaces
    @spaces.GPU
    def _zero_gpu_guard():
        pass
except Exception:
    pass

from src.ui.aisight_app import build_aisight_ui

demo = build_aisight_ui()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    share = os.environ.get("GRADIO_SHARE", "false").lower() in ("true", "1", "yes")
    demo.launch(server_name="0.0.0.0", server_port=port, share=share, show_api=False)

