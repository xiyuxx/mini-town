"""Mini-Town backend entry point."""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from backend.town.engine import SimulationEngine
from backend.api.routes import router, set_engine

engine = SimulationEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.init()
    set_engine(engine)
    yield
    await engine.shutdown()


app = FastAPI(title="Mini-Town", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve the production frontend from the same origin as REST and WebSocket.
# This keeps Cohub Works deployments single-port and avoids cross-origin proxying.
_frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=_frontend_dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def frontend_app(full_path: str):
        requested = (_frontend_dist / full_path).resolve()
        if requested.is_file() and _frontend_dist.resolve() in requested.parents:
            return FileResponse(requested)
        return FileResponse(_frontend_dist / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
