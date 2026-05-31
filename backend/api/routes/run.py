from fastapi import APIRouter, Depends, HTTPException

from ..schemas import RunRequest, RunResponse, AgentMessageOut, GraphState, AdversarialMeta
from .sessions import ensure_session
from ...core.adversarial_lifecycle import run_adversarial_task
from ...core.auth import AuthedUser, get_optional_user
from ...core.config import settings
from ...core.lifecycle import run_task
from ...router.model_router import make_router_for_user

router = APIRouter()


@router.post("/run", response_model=RunResponse)
async def run_collaboration(
    req: RunRequest, user: "AuthedUser | None" = Depends(get_optional_user)
) -> RunResponse:
    user_id = user.id if user else None
    session_id = req.session_id
    per_user_router = None

    # Multi-user mode: resolve the caller's own provider credential and ensure a session.
    if settings.use_supabase and user_id:
        per_user_router = await make_router_for_user(user_id)
        if per_user_router is None:
            raise HTTPException(
                status_code=400,
                detail="No LLM provider connected. Connect one via /auth/openrouter/start or POST /providers.",
            )
        if not session_id:
            session_id = ensure_session(user_id, req.domain)

    if req.mode == "adversarial":
        result = await run_adversarial_task(
            input_text=req.input,
            domain=req.domain,
            max_rounds=req.max_rounds,
            user_id=user_id,
            session_id=session_id,
            router=per_user_router,
        )
        adv_meta = AdversarialMeta(**result["adversarial_meta"])
    else:
        result = await run_task(
            input_text=req.input,
            domain=req.domain,
            pipeline=req.pipeline,
            user_id=user_id,
            session_id=session_id,
            router=per_user_router,
        )
        adv_meta = None

    return RunResponse(
        task_id=result["task_id"],
        domain=result["domain"],
        pipeline=result["pipeline"],
        messages=[AgentMessageOut(**m) for m in result["messages"]],
        scores=result.get("scores", {}),
        model_used=result.get("model_used", ""),
        knowledge_graph=GraphState(**result["knowledge_graph"]),
        adversarial_meta=adv_meta,
    )
