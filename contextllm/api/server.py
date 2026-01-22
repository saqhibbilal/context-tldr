"""FastAPI server application."""

import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextllm.api.routes import router
from contextllm.utils.logging_setup import setup_logging
from contextllm.utils.config import get_config

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Get config
config = get_config()

# Create FastAPI app
app = FastAPI(
    title="Context Budget Optimizer",
    description="Intelligently selects document chunks to maximize answer quality within token budget",
    version="0.1.0"
)

# Include API routes
app.include_router(router)

# Get frontend directory
frontend_dir = Path(__file__).parent.parent.parent / "frontend"


@app.get("/")
async def root():
    """Serve the main frontend page."""
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Context Budget Optimizer API", "frontend": "not found"}


# Mount static files if frontend directory exists
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    logger.info(f"Serving static files from {frontend_dir}")


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Context Budget Optimizer API starting up...")
    logger.info(f"Default budget: {config.get('optimization.default_budget', 2000)} tokens")
    logger.info(f"Embedding model: {config.get('embedding.model_name', 'unknown')}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Context Budget Optimizer API shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
