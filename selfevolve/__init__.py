"""SelfEvolve — self-evolving agent architecture with AGI-oriented meta-cognition.

Layers (bottom-up):
  1. MetaCognition       — agent observes own reasoning, identifies blind spots
  2. ExperienceMemory    — episodic replay with consolidation and forgetting curves
  3. SkillLibrary        — composable skills with auto-extraction from reflection
  4. WorldModel          — probabilistic beliefs about which strategies generalize
  5. CuriosityDrive      — intrinsic motivation from epistemic uncertainty
  6. RecursiveSelfMod    — agent reads/writes own code, tests, promotes
  7. MultiAgentSociety   — debate, teaching, specialization, voting
  8. AGIMetrics          — receptive-field, compositionality, improvement-rate
"""

from .core import (
    TrainablePEP,
    MetaCognition,
    WorldModel,
    SkillLibrary,
    Skill,
    ExperienceMemory,
    SelfEvolvingAgent,
)

__all__ = [
    "TrainablePEP",
    "MetaCognition",
    "WorldModel",
    "SkillLibrary",
    "Skill",
    "ExperienceMemory",
    "SelfEvolvingAgent",
]
