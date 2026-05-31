"""Rubric-based scorer: rates collaboration output on 5 dimensions.

Uses the caller-supplied per-user router (multi-user mode) so scoring spends the
same user's credentials as the run; falls back to the singleton router otherwise.
"""
from __future__ import annotations

import json
import re

from ..router.model_router import ModelRouter, router as default_router

_DIMENSIONS = ["accuracy", "depth", "clarity", "actionability", "coverage"]

_SYSTEM = """You are an impartial evaluator of AI-generated analysis.
Score the provided output on each dimension from 0.0 to 1.0.
Return ONLY a JSON object with these exact keys: accuracy, depth, clarity, actionability, coverage.
No explanation, no markdown — raw JSON only."""

_PROMPT_TEMPLATE = """## Task
{task}

## Agent Output to Evaluate
{output}

Score the output."""


async def score(
    task: str, output: str, router: ModelRouter | None = None
) -> dict[str, float]:
    r = router or default_router
    resp = await r.complete(
        system=_SYSTEM,
        user=_PROMPT_TEMPLATE.format(task=task, output=output),
        role_hint="rubric",
    )
    raw = (resp.content or "").strip()
    raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
    scores = json.loads(raw)
    return {k: float(scores.get(k, 0.0)) for k in _DIMENSIONS}
