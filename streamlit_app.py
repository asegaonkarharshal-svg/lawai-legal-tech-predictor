"""
streamlit_app.py
----------------
Root entrypoint for Streamlit Community Cloud deployment.
Redirects execution into frontend/app.py with proper sys.path resolution.
"""

from pathlib import Path
import runpy
import sys

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Launch main frontend application
app_path = ROOT_DIR / "frontend" / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
