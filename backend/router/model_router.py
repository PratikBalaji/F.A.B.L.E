"""
Learned Model Router — routes queries to the best-performing model
based on historical performance data from the knowledge engine.

Starts with 2 models, learns which excels at what over time.
"""
from __future__ import annotations

from dataclasses import dataclass

import openai
import structlog

from ..core.config import settings

log = structlog.get_logger()


@dataclass
class ModelResponse:
    content: str
    model: str
    usage: dict[str, int]


# Available models on OpenRouter
AVAILABLE_MODELS = {
    "primary": settings.primary_model,       # Claude Sonnet
    "secondary": settings.secondary_model,   # GPT-4o-mini
}

# Adversarial pipeline: maps adv: role keys to specific model IDs
ROLE_MODEL_MAP: dict[str, str] = {
    "adv:planner":   settings.planner_model,
    "adv:actor":     settings.actor_model,
    "adv:critic":    settings.adv_critic_model,
    "adv:validator": settings.validator_model,
    "adv:refiner":   settings.refiner_model,
    "adv:judge":     settings.judge_model,
}


class ModelRouter:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str = "openrouter",
    ) -> None:
        # Defaults preserve the legacy single-key behaviour (ModelRouter()).
        self.provider = provider
        self._client = openai.AsyncOpenAI(
            api_key=api_key or settings.openrouter_api_key,
            base_url=base_url or settings.openrouter_base_url,
            default_headers={
                "HTTP-Referer": settings.app_url,
                "X-Title": settings.app_name,
            },
        )

    @classmethod
    def from_credential(cls, cred) -> "ModelRouter":
        """Build a per-user router from a ResolvedCredential."""
        return cls(api_key=cred.api_key, base_url=cred.base_url, provider=cred.provider)

    def _resolve_model(self, requested: str | None) -> str:
        """
        OpenRouter understands vendor-prefixed slugs (anthropic/claude-...), so we
        pass `requested` through. Direct BYOK providers (openai/anthropic/google)
        can't use OpenRouter slugs or per-role multi-vendor routing, so they fall
        back to that provider's single default model.
        """
        if self.provider == "openrouter":
            return requested or settings.primary_model
        from ..core.credentials import PROVIDER_DEFAULT_MODEL

        return PROVIDER_DEFAULT_MODEL.get(self.provider, settings.primary_model)

    async def complete(
        self,
        system: str,
        user: str,
        role_hint: str = "",
        force_model: str | None = None,
    ) -> ModelResponse:
        model = self._resolve_model(force_model)

        resp = await self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=2048,
        )
        msg = resp.choices[0].message
        return ModelResponse(
            content=msg.content or "",
            model=resp.model,
            usage={
                "input": resp.usage.prompt_tokens if resp.usage else 0,
                "output": resp.usage.completion_tokens if resp.usage else 0,
            },
        )

    async def complete_for_role(
        self,
        role: str,
        system: str,
        user: str,
        max_tokens: int = 2048,
    ) -> ModelResponse:
        """Route to the LLM assigned to this adversarial role."""
        model = self._resolve_model(ROLE_MODEL_MAP.get(role))
        log.info("adversarial_model_routed", role=role, model=model, provider=self.provider)
        resp = await self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
        )
        msg = resp.choices[0].message
        return ModelResponse(
            content=msg.content or "",
            model=resp.model,
            usage={
                "input": resp.usage.prompt_tokens if resp.usage else 0,
                "output": resp.usage.completion_tokens if resp.usage else 0,
            },
        )

    async def complete_with_routing(
        self,
        system: str,
        user: str,
        role_hint: str = "",
        preferred_model: str | None = None,
    ) -> ModelResponse:
        """
        Route to a model based on learned performance data.
        preferred_model comes from the knowledge engine's recommendation.
        Falls back to primary model if no recommendation exists.
        """
        model = preferred_model or settings.primary_model
        log.info("model_routed", model=model, role=role_hint, learned=preferred_model is not None)
        return await self.complete(system, user, role_hint, force_model=model)

    async def compare_models(
        self,
        system: str,
        user: str,
    ) -> dict[str, ModelResponse]:
        """Run the same prompt through both models for comparison."""
        import asyncio
        results = {}
        tasks = []
        for label, model_id in AVAILABLE_MODELS.items():
            tasks.append(self._run_model(label, model_id, system, user))
        completed = await asyncio.gather(*tasks)
        for label, response in completed:
            results[label] = response
        return results

    async def _run_model(
        self, label: str, model_id: str, system: str, user: str
    ) -> tuple[str, ModelResponse]:
        response = await self.complete(system, user, force_model=model_id)
        return (label, response)


# Legacy singleton — used as a fallback in single-user mode and at agent
# registration time. In multi-user mode a per-user router is injected per request.
router = ModelRouter()


async def make_router_for_user(
    user_id: str, preferred: str | None = None
) -> "ModelRouter | None":
    """Resolve the user's active provider credential into a per-user router.

    Returns None if the user has no connected provider (caller decides fallback).
    """
    from ..core.credentials import resolve_credential

    cred = await resolve_credential(user_id, preferred)
    if cred is None:
        return None
    return ModelRouter.from_credential(cred)
