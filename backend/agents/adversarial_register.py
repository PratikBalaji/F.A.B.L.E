"""Register adversarial pipeline agents on the global AgentBus.

Uses adv: prefixed role names so these agents coexist with the standard
analyst/critic/synthesizer pipeline without overwriting any existing registrations.
"""
from ..core.bus import bus
from ..router.model_router import router
from .adversarial import (
    PlannerAgent,
    ActorAgent,
    AdversarialCriticAgent,
    ValidatorAgent,
    RefinerAgent,
    JudgeAgent,
)

ADVERSARIAL_PIPELINE = [
    "adv:planner",
    "adv:actor",
    "adv:critic",
    "adv:validator",
    "adv:refiner",
    "adv:judge",
]


def register_adversarial() -> None:
    """Register all six adversarial agents on the global bus."""
    bus.register("adv:planner",   PlannerAgent(router))
    bus.register("adv:actor",     ActorAgent(router))
    bus.register("adv:critic",    AdversarialCriticAgent(router))
    bus.register("adv:validator", ValidatorAgent(router))
    bus.register("adv:refiner",   RefinerAgent(router))
    bus.register("adv:judge",     JudgeAgent(router))
