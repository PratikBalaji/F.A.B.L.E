"""
Supabase access for the backend (service-role).

IMPORTANT: the service-role key BYPASSES Row-Level Security. Every function here
takes an explicit user_id and filters on it. RLS is the backstop for direct/anon
client access; this layer is the primary per-user guard. The service-role key must
never be sent to the browser.
"""
from __future__ import annotations

import structlog

from .config import settings

log = structlog.get_logger()

_client = None  # lazy singleton


def get_db():
    """Return a cached service-role Supabase client; raise if not configured."""
    global _client
    if _client is None:
        if not settings.supabase_url or not settings.supabase_service_role_key:
            raise RuntimeError(
                "Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
                "(and USE_SUPABASE=true) to enable multi-user mode."
            )
        from supabase import create_client  # imported lazily so legacy mode needs no dep

        _client = create_client(settings.supabase_url, settings.supabase_service_role_key)
        log.info("supabase_client_initialized", url=settings.supabase_url)
    return _client


def vector_literal(embedding: list[float]) -> str:
    """Format an embedding as a pgvector text literal: [0.1,0.2,...]."""
    return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


def match_memory_chunks(
    user_id: str, embedding: list[float], match_count: int = 8
) -> list[dict]:
    """Call the user-scoped cosine-search SQL function. Returns rows with `similarity`."""
    db = get_db()
    resp = db.rpc(
        "match_memory_chunks",
        {
            "p_user_id": user_id,
            "query_embedding": vector_literal(embedding),
            "match_count": match_count,
        },
    ).execute()
    return resp.data or []
