# Vercel Python entrypoint for API. We re-export FastAPI app from backend.
from backend.app import app  # noqa: F401


