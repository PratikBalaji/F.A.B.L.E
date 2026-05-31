"""Chat sessions + cross-session memory search. All endpoints require an authenticated user."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..schemas import (
    ChatMessageOut,
    MemoryHitOut,
    SessionCreateRequest,
    SessionOut,
)
from ...core.auth import AuthedUser, get_current_user
from ...core.db import get_db
from ...core.memory_service import memory_service

router = APIRouter()

_SESSION_COLUMNS = "id,title,domain,created_at,updated_at"


def ensure_session(user_id: str, domain: str = "general", title: str | None = None) -> str:
    """Create a chat session and return its id. Used by /run when no session is supplied."""
    db = get_db()
    res = (
        db.table("chat_sessions")
        .insert({"user_id": user_id, "domain": domain, "title": title})
        .execute()
    )
    return res.data[0]["id"]


@router.post("/sessions", response_model=SessionOut)
async def create_session(
    req: SessionCreateRequest, user: AuthedUser = Depends(get_current_user)
) -> SessionOut:
    db = get_db()
    res = (
        db.table("chat_sessions")
        .insert({"user_id": user.id, "domain": req.domain, "title": req.title})
        .execute()
    )
    return SessionOut(**{k: res.data[0].get(k) for k in ("id", "title", "domain", "created_at", "updated_at")})


@router.get("/sessions", response_model=list[SessionOut])
async def list_sessions(user: AuthedUser = Depends(get_current_user)) -> list[SessionOut]:
    db = get_db()
    res = (
        db.table("chat_sessions")
        .select(_SESSION_COLUMNS)
        .eq("user_id", user.id)
        .order("updated_at", desc=True)
        .execute()
    )
    return [SessionOut(**r) for r in (res.data or [])]


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageOut])
async def session_messages(
    session_id: str, user: AuthedUser = Depends(get_current_user)
) -> list[ChatMessageOut]:
    db = get_db()
    owned = (
        db.table("chat_sessions")
        .select("id")
        .eq("id", session_id)
        .eq("user_id", user.id)
        .limit(1)
        .execute()
        .data
        or []
    )
    if not owned:
        raise HTTPException(404, "Session not found")
    res = (
        db.table("chat_messages")
        .select("id,role,content,model_used,scores,adversarial_run_id,created_at")
        .eq("session_id", session_id)
        .eq("user_id", user.id)
        .order("created_at")
        .execute()
    )
    return [ChatMessageOut(**r) for r in (res.data or [])]


@router.get("/memory", response_model=list[MemoryHitOut])
async def memory_search(
    q: str, top_k: int = 8, user: AuthedUser = Depends(get_current_user)
) -> list[MemoryHitOut]:
    hits = await memory_service.retrieve(user.id, q, top_k=top_k)
    return [
        MemoryHitOut(
            id=h["id"],
            source_type=h["source_type"],
            session_id=h.get("session_id"),
            domain=h.get("domain"),
            content=h.get("content", ""),
            similarity=float(h.get("similarity") or 0.0),
            created_at=h.get("created_at"),
        )
        for h in hits
    ]
