"""
GCAEMMCD: AI-Powered Multimodal Cancer Diagnosis Workstation.
Production Application Entrypoint for Hugging Face Spaces & Cloud Deployment.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ui.aisight_app import build_aisight_ui

demo = build_aisight_ui()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    share = os.environ.get("GRADIO_SHARE", "false").lower() in ("true", "1", "yes")
    demo.launch(server_name="0.0.0.0", server_port=port, share=share)
