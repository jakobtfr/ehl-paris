"""HuggingFace Spaces entry point. Launches the floorgen demo.

Space config: SDK = gradio, app_file = app.py. Add src to path so the package
imports without an editable install.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from floorgen.demo.app import build_demo  # noqa: E402

demo = build_demo()

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860"))),
    )
