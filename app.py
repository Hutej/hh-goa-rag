"""Hugging Face Spaces entrypoint (also runnable locally).

Spaces expects the app object to be importable as ``app`` from the file named
in the Space's metadata (or ``app.py`` by default). We expose the FastAPI app
built in ``backend.app`` so the deployment and local runs share ONE code path:

    local :  venv/bin/python -m uvicorn backend.app:app --port 7860
    Space  :  uvicorn backend.app:app --host 0.0.0.0 --port 7860  (via this file)

No logic is duplicated here — this module only re-exports the real app.
"""

from backend.app import app  # noqa: F401  (the Spaces server imports `app`)
