"""FastAPI application entry point."""
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..agents.register import register_all
from ..agents.adversarial_register import register_adversarial
from ..core.config import settings
from .routes import run, feedback, ingest, graph, providers, sessions, auth_openrouter

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    register_all()           # standard: analyst, critic, synthesizer
    register_adversarial()   # adversarial: adv:planner, adv:actor, adv:critic, adv:validator, adv:refiner, adv:judge
    log.info("fable_started")
    yield
    log.info("fable_stopped")


app = FastAPI(
    title="F.A.B.L.E. API",
    description="Federated Agent Bus & Lifecycle Engine",
    version="0.1.0",
    lifespan=lifespan,
)

# In multi-user mode, lock CORS to the app origin and allow credentialed requests;
# legacy single-user mode stays permissive for local development.
_cors_kwargs = (
    dict(allow_origins=[settings.app_url], allow_credentials=True)
    if settings.use_supabase
    else dict(allow_origins=["*"])
)
app.add_middleware(
    CORSMiddleware,
    allow_methods=["*"],
    allow_headers=["*"],
    **_cors_kwargs,
)

app.include_router(run.router, tags=["Orchestration"])
app.include_router(feedback.router, tags=["Feedback"])
app.include_router(ingest.router, tags=["RAG"])
app.include_router(graph.router, tags=["Knowledge Graph"])
app.include_router(providers.router, tags=["Providers"])
app.include_router(auth_openrouter.router, tags=["Auth"])
app.include_router(sessions.router, tags=["Sessions & Memory"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
