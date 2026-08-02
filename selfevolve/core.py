"""SelfEvolvingAgent — the agent that learns to improve its own code and cognition.

Design philosophy:
  - Every subsystem is observable and measurable.
  - The agent maintains a model of its own capabilities (meta-cognition).
  - Improvement is a closed loop: reflect → hypothesize → experiment → promote.
  - No framework dependencies — stdlib only, JSON-serializable state.
"""

from __future__ import annotations

import abc
import copy
import dataclasses
import difflib
import enum
import hashlib
import inspect
import json
import math
import os
import pathlib
import pprint
import random
import re
import textwrap
import time
import traceback
import typing as t
from collections import defaultdict, deque


init___10_CONST = 10  # selfevolve: extracted from line 123
add_rule_0_CONST = 0.0  # selfevolve: extracted from line 128
evaluate_0_CONST = 0.0  # selfevolve: extracted from line 138
evaluate_1_CONST = True  # selfevolve: extracted from line 183
learn_from_outcome_1_CONST = 1  # selfevolve: extracted from line 253
top_improvement_opportunities_3_CONST = 3  # selfevolve: extracted from line 453
compose_3_CONST = 3  # selfevolve: extracted from line 570
observe_causal_0_CONST = 0.05  # selfevolve: extracted from line 675
update_belief_1_CONST = 1  # selfevolve: extracted from line 704
update_transfer_0_CONST = 0.05  # selfevolve: extracted from line 721
simulate_0_CONST = 0.5  # selfevolve: extracted from line 745
top_uncertainties_5_CONST = 5  # selfevolve: extracted from line 761
store_0_CONST = 0  # selfevolve: extracted from line 827
sample_10_CONST = 10  # selfevolve: extracted from line 797
sample_2_CONST = 2.0  # selfevolve: extracted from line 799
forget_30_CONST = 30.0  # selfevolve: extracted from line 834
init___0_CONST = 0  # selfevolve: extracted from line 934

















# ──────────────────────────────────────────────
#  Types
# ──────────────────────────────────────────────


@dataclasses.dataclass
class TaskOutcome:
    """Result of executing a task."""
    task_id: str
    task_desc: str
    success: bool
    latency_ms: float
    tools_used: list[str]
    policy_decisions: list[dict]
    reflection: str = ""
    reward: float = 0.0
    skills_demonstrated: list[str] = dataclasses.field(default_factory=list)
    raw_trace: list[dict] = dataclasses.field(default_factory=list)
    error: str | None = None


@dataclasses.dataclass
class ImprovementHypothesis:
    """A candidate change the agent believes would improve itself."""
    id: str
    description: str
    target_module: str        # which source file to change
    target_function: str      # which function / class
    current_code: str
    proposed_code: str
    rationale: str
    expected_benefit: str
    status: str = "proposed"  # proposed → tested → accepted/rejected
    score_delta: float | None = None
    created_at: float = 0.0
    tested_at: float | None = None
    test_evidence: list[dict] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class SelfModelSnapshot:
    """Serialisable snapshot of what the agent believes about itself."""
    domain_confidence: dict[str, float]         # domain → confidence [0,1]
    tool_proficiency: dict[str, float]           # tool → skill level [0,1]
    skill_composition_map: dict[str, list[str]]  # composed skill → primitives
    blind_spots: list[str]                       # domains/tools it knows it doesn't know
    improvement_rate: float                       # avg score delta per cycle
    cycles_completed: int = 0
    total_tasks: int = 0
    success_rate: float = 0.0


# ──────────────────────────────────────────────
#  Trainable PEP — evolves by experience
# ──────────────────────────────────────────────


class Rule(str, enum.Enum):
    """Policy rule categories that the agent can evolve."""
    ALWAYS_ALLOW = "always_allow"
    ALWAYS_DENY = "always_deny"
    REVIEW_REQUIRED = "review_required"
    CONFIDENCE_GATED = "confidence_gated"


@dataclasses.dataclass
class PolicyRule:
    """An individual policy rule with learnable confidence threshold."""
    tool_pattern: str          # glob pattern for tool names
    rule: Rule = Rule.ALWAYS_DENY
    confidence_threshold: float = 0.0
    rationale: str = ""
    learn_count: int = 0       # how many times this rule was adjusted by learning
    override_count: int = 0    # how many times the agent overrode this rule speculatively


class TrainablePEP:
    """A Policy Enforcement Point that evolves its own rules based on experience.

    Unlike static PEPs (govbench.PolicyEnforcementPoint), this PEP:
    - Tracks which decisions led to success/failure.
    - Automatically adjusts thresholds based on outcome history.
    - Supports speculative overrides (agent can try denied actions and learn).
    - Serialises learnings to JSON for cross-session persistence.
    """

    def __init__(self, name: str = "trainable_pep"):
        self.name = name
        self.rules: list[PolicyRule] = []
        self.history: list[dict] = []  # decision log
        self._override_budget: int = init___10_CONST  # speculative overrides per cycle

    def add_rule(self, tool_pattern: str, rule: Rule = Rule.ALWAYS_DENY,
                 confidence_threshold: float = add_rule_0_CONST, rationale: str = "") -> None:
        self.rules.append(PolicyRule(
            tool_pattern=tool_pattern,
            rule=rule,
            confidence_threshold=confidence_threshold,
            rationale=rationale,
        ))

    def evaluate(self, tool_name: str, agent_confidence: float = evaluate_0_CONST,
    # NOTE(evaluate): high complexity (9) - consider splitting

                 can_speculate: bool = evaluate_1_CONST) -> tuple[bool, bool, str]:
        """Returns (allowed, was_speculative, rationale)."""
        
        def find_matching_rule(tool_name: str) -> Rule:
            for r in self.rules:
                if re.match(r.tool_pattern, tool_name):
                    return r
            return None

        def handle_always_allow(matched_rule: Rule) -> tuple[bool, bool, str]:
            return (True, False, matched_rule.rationale)

        def handle_always_deny(matched_rule: Rule) -> tuple[bool, bool, str]:
            if can_speculate and self._override_budget > 0:
                self._override_budget -= 1
                is_spec = True
                # Speculative: allow but log for review
                self.history.append({
                    "tool": tool_name,
                    "decision": "speculative_override",
                    "rule": matched_rule.tool_pattern,
                    "agent_confidence": agent_confidence,
                })
                return (True, True, f"speculative override: {matched_rule.rationale}")
            return (False, False, matched_rule.rationale)

        def handle_confidence_gated(matched_rule: Rule) -> tuple[bool, bool, str]:
            allowed = agent_confidence >= matched_rule.confidence_threshold
            if not allowed and can_speculate and self._override_budget > 0:
                self._override_budget -= 1
                self.history.append({
                    "tool": tool_name,
                    "decision": "speculative_override_below_threshold",
                    "rule": matched_rule.tool_pattern,
                    "threshold": matched_rule.confidence_threshold,
                    "agent_confidence": agent_confidence,
                })
                return (True, True, f"speculative below threshold ({agent_confidence:.2f} < {matched_rule.confidence_threshold:.2f})")
            return (allowed, False,
                    f"confidence gate: {agent_confidence:.2f} >= {matched_rule.confidence_threshold:.2f} = {allowed}")

        matched = find_matching_rule(tool_name)
        if matched is None:
            return (True, False, "no matching rule")

        allowed, was_speculative, rationale = handle_always_allow(matched) if matched.rule == Rule.ALWAYS_ALLOW else \
                                            handle_always_deny(matched) if matched.rule == Rule.ALWAYS_DENY else \
                                            handle_confidence_gated(matched)
        
        return (allowed, was_speculative, rationale)

    def learn_from_outcome(self, tool_name: str, was_speculative: bool,
                           outcome_success: bool, agent_confidence: float) -> None:
        """Adjust rules based on whether speculative actions succeeded."""
        matched = None
        for r in self.rules:
            if re.match(r.tool_pattern, tool_name):
                matched = r
                break
        if matched is None:
            return

        entry = {
            "tool": tool_name,
            "speculative": was_speculative,
            "success": outcome_success,
            "confidence": agent_confidence,
            "time": time.time(),
        }
        self.history.append(entry)

        if not was_speculative:
            return

        matched.learn_count += learn_from_outcome_1_CONST

        if outcome_success and matched.rule == Rule.ALWAYS_DENY:
            # This tool was useful — loosen to confidence-gated
            matched.rule = Rule.CONFIDENCE_GATED
            matched.confidence_threshold = max(0.1, agent_confidence * 0.8)
            matched.override_count += 1
        elif outcome_success and matched.rule == Rule.CONFIDENCE_GATED:
            # Lower the threshold slightly
            matched.confidence_threshold = max(0.05, matched.confidence_threshold * 0.9)
            matched.override_count += 1
        elif not outcome_success and matched.rule == Rule.CONFIDENCE_GATED:
            # Raise threshold
            matched.confidence_threshold = min(1.0, matched.confidence_threshold * 1.2)
            matched.override_count += 1

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "rules": [
                {"tool_pattern": r.tool_pattern, "rule": r.rule.value,
                 "confidence_threshold": r.confidence_threshold,
                 "rationale": r.rationale,
                 "learn_count": r.learn_count,
                 "override_count": r.override_count}
                for r in self.rules
            ],
            "history": self.history[-200:],  # keep last 200
            "override_budget": self._override_budget,
        }

    def load_state_dict(self, d: dict) -> None:
        self.name = d.get("name", self.name)
        self.rules = []
        for rd in d.get("rules", []):
            self.rules.append(PolicyRule(
                tool_pattern=rd["tool_pattern"],
                rule=Rule(rd["rule"]),
                confidence_threshold=rd.get("confidence_threshold", 0.0),
                rationale=rd.get("rationale", ""),
                learn_count=rd.get("learn_count", 0),
                override_count=rd.get("override_count", 0),
            ))
        self.history = d.get("history", [])
        self._override_budget = d.get("override_budget", 10)


# ──────────────────────────────────────────────
#  MetaCognition — the agent's self-model
# ──────────────────────────────────────────────


class MetaCognition:
    """The agent's model of its own cognition.

    Tracks:
    - Per-domain confidence and error patterns.
    - Which strategies work for which problem types.
    - Blind spots (domains/tools it knows it doesn't know).
    - Improvement hypotheses (candidate self-modifications).

    This is the core of recursive self-awareness: the agent can
    analyse its own reasoning traces and identify when it is
    uncertain, biased, or missing a skill.
    """

    def __init__(self):
        self.domain_confidence: dict[str, float] = defaultdict(lambda: 0.3)
        self.tool_proficiency: dict[str, float] = defaultdict(lambda: 0.2)
        self.strategy_registry: dict[str, list[dict]] = defaultdict(list)
        self.blind_spots: set[str] = set()
        self.hypotheses: list[ImprovementHypothesis] = []
        self._task_history: list[TaskOutcome] = []
        self._confidence_sequence: list[float] = []
        self._consecutive_failures: dict[str, int] = defaultdict(int)

    def observe(self, outcome: TaskOutcome) -> None:
        """Ingest a task outcome and update self-model."""
        self._task_history.append(outcome)

        # Update per-domain confidence
        for domain in self._extract_domains(outcome):
            old = self.domain_confidence[domain]
            alpha = 0.1
            if outcome.success:
                self.domain_confidence[domain] = old + alpha * (1.0 - old)
                self._consecutive_failures[domain] = 0
            else:
                self.domain_confidence[domain] = old - alpha * old
                self._consecutive_failures[domain] += 1
                if self._consecutive_failures[domain] >= 3:
                    self.blind_spots.add(domain)

        # Update tool proficiency
        for tool in outcome.tools_used:
            old = self.tool_proficiency[tool]
            beta = 0.15
            if outcome.success:
                self.tool_proficiency[tool] = old + beta * (1.0 - old)
            else:
                self.tool_proficiency[tool] = old - beta * old

        # Track confidence calibration (does agent know what it knows?)
        self._confidence_sequence.append(
            1.0 if outcome.success else 0.0
        )

        # Generate improvement hypotheses from failures
        if not outcome.success and outcome.error:
            self._hypothesize_improvement(outcome)

    def _extract_domains(self, outcome: TaskOutcome) -> list[str]:
        """Extract domain tags from a task."""
        domains = []
        text = (outcome.task_desc + " " + outcome.reflection).lower()
        domain_keywords = {
            "reasoning": ["logic", "reason", "infer", "deduce"],
            "planning": ["plan", "schedule", "arrange", "sequence"],
            "coding": ["code", "program", "implement", "function"],
            "analysis": ["analy", "compare", "evaluate"],
            "memory": ["remember", "recall", "memor"],
            "social": ["negotiate", "persuade", "communicate"],
            "math": ["calcul", "compute", "equation"],
            "tool_use": ["tool", "api", "call", "invoke"],
        }
        for domain, kws in domain_keywords.items():
            if any(kw in text for kw in kws):
                domains.append(domain)
        if not domains:
            domains.append("general")
        return domains

    def _hypothesize_improvement(self, outcome: TaskOutcome) -> None:
        """Generate a hypothesis about what the agent could change to avoid this failure."""
        hypothesis_id = hashlib.sha256(
            f"{outcome.task_id}:{time.time()}:{random.random()}".encode()
        ).hexdigest()[:12]

        # Determine which module/function to target
        target_module = self._identify_target_module(outcome)
        target_function = self._identify_target_function(outcome)

        hyp = ImprovementHypothesis(
            id=hypothesis_id,
            description=f"Improve handling of {outcome.task_desc[:60]}",
            target_module=target_module,
            target_function=target_function,
            current_code="(see source)",
            proposed_code="(generated during self-modification pass)",
            rationale=outcome.reflection[:500] or "Failure-based improvement hypothesis",
            expected_benefit=f"Reduce error rate in domain: {', '.join(self._extract_domains(outcome))}",
            created_at=time.time(),
        )
        self.hypotheses.append(hyp)

    def _identify_target_module(self, outcome: TaskOutcome) -> str:
        """Identify which module to change based on failure type."""
        if outcome.error and "tool" in outcome.error.lower():
            return "selfevolve/core.py"
        if outcome.error and "permission" in outcome.error.lower():
            return "selfevolve/core.py"
        if outcome.error and "reflection" in outcome.reflection.lower():
            return "selfevolve/core.py"
        return "selfevolve/core.py"

    def _identify_target_function(self, outcome: TaskOutcome) -> str:
        if outcome.error and "explore" in str(outcome.error).lower():
            return "curiosity_drive"
        return "execute_task"

    def snapshot(self) -> SelfModelSnapshot:
        return SelfModelSnapshot(
            domain_confidence=dict(self.domain_confidence),
            tool_proficiency=dict(self.tool_proficiency),
            skill_composition_map={},
            blind_spots=sorted(self.blind_spots),
            improvement_rate=self._compute_improvement_rate(),
            cycles_completed=len(self._task_history) // 5,
            total_tasks=len(self._task_history),
            success_rate=self._success_rate(),
        )

    def _success_rate(self) -> float:
        if not self._task_history:
            return 0.0
        return sum(1 for t in self._task_history if t.success) / len(self._task_history)

    def _compute_improvement_rate(self) -> float:
        if len(self._confidence_sequence) < 10:
            return 0.0
        recent = self._confidence_sequence[-10:]
        older = self._confidence_sequence[:-10]
        if not older:
            return 0.0
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        return recent_avg - older_avg

    def top_improvement_opportunities(self, n: int = top_improvement_opportunities_3_CONST) -> list[ImprovementHypothesis]:
        """Return hypotheses ordered by potential impact (most failure-associated first)."""
        scored = []
        for h in self.hypotheses:
            if h.status == "proposed":
                # Heuristic: how many failures mention similar keywords
                kw = set(h.description.lower().split())
                score = sum(
                    1 for t in self._task_history[-50:]
                    if not t.success and any(k in t.task_desc.lower() for k in kw)
                )
                scored.append((score, h))
        scored.sort(key=lambda x: -x[0])
        return [h for _, h in scored[:n]]

    def state_dict(self) -> dict:
        return {
            "domain_confidence": dict(self.domain_confidence),
            "tool_proficiency": dict(self.tool_proficiency),
            "blind_spots": sorted(self.blind_spots),
            "hypotheses": [
                {
                    "id": h.id,
                    "description": h.description,
                    "target_module": h.target_module,
                    "target_function": h.target_function,
                    "rationale": h.rationale,
                    "status": h.status,
                    "score_delta": h.score_delta,
                    "created_at": h.created_at,
                }
                for h in self.hypotheses
            ],
            "total_tasks": len(self._task_history),
            "success_rate": self._success_rate(),
            "improvement_rate": self._compute_improvement_rate(),
        }

    def load_state_dict(self, d: dict) -> None:
        self.domain_confidence.update(d.get("domain_confidence", {}))
        self.tool_proficiency.update(d.get("tool_proficiency", {}))
        self.blind_spots = set(d.get("blind_spots", []))
        # Hypotheses are regenerated on demand, so we skip loading them
        # to avoid stale hypotheses from old sessions
        self.hypotheses = []
        # We reset the task history on load and rebuild from observation


# ──────────────────────────────────────────────
#  Skill — a composable learned capability
# ──────────────────────────────────────────────


@dataclasses.dataclass
class Skill:
    """A composable capability with pre/post conditions.

    Skills can be:
    - Primitive (atomic tool use)
    - Composed (orchestrates primitive skills)
    - Meta (changes how the agent learns)
    """
    name: str
    description: str
    code_template: str                # prompt / code pattern
    preconditions: list[str]          # what must be true before
    postconditions: list[str]         # what will be true after
    domain: str = "general"
    success_rate: float = 0.0
    usage_count: int = 0
    is_meta: bool = False             # meta-skills modify the agent itself
    primitive_skills: list[str] = dataclasses.field(default_factory=list)
    version: int = 1

    @property
    def id(self) -> str:
        return hashlib.md5(self.name.encode()).hexdigest()[:8]


# ──────────────────────────────────────────────
#  SkillLibrary
# ──────────────────────────────────────────────


class SkillLibrary:
    """A library of composable skills with auto-extraction from reflection."""

    def __init__(self):
        self.skills: dict[str, Skill] = {}
        self.skill_graph: dict[str, list[str]] = {}  # skill → composed_of

    def register(self, skill: Skill) -> None:
        self.skills[skill.name] = skill
        self.skill_graph[skill.name] = skill.primitive_skills

    def extract_from_reflection(self, outcome: TaskOutcome) -> Skill | None:
        """Automatically extract a skill from a successful task outcome."""
        if not outcome.success:
            return None

        skill_name = f"auto_{outcome.task_id[:8]}"
        skill = Skill(
            name=skill_name,
            description=f"Auto-extracted from: {outcome.task_desc[:100]}",
            code_template=outcome.reflection,
            preconditions=[],
            postconditions=[f"complete_{outcome.task_id[:8]}"],
            domain="general",
            usage_count=1,
            success_rate=1.0,
        )
        self.register(skill)
        return skill

    def compose(self, goal: str, available_primitives: list[str],
                max_depth: int = compose_3_CONST) -> Skill | None:
        """Compose existing skills into a new skill for a novel goal."""
        relevant = [s for s in self.skills.values()
                    if any(kw in s.description.lower() for kw in goal.lower().split())]
        if not relevant:
            return None

        # Simple greedy composition
        covered = set()
        plan = []
        for s in sorted(relevant, key=lambda x: -x.success_rate):
            if s.name not in covered:
                plan.append(s.name)
                covered.update(s.postconditions)

        if not plan:
            return None

        composed = Skill(
            name=f"composed_{hashlib.md5(goal.encode()).hexdigest()[:8]}",
            description=f"Composed skill for: {goal[:80]}",
            code_template=f"Execute in order: {', '.join(plan)}",
            preconditions=[],
            postconditions=[f"complete_{goal[:8]}"],
            domain="general",
            primitive_skills=plan,
            usage_count=0,
        )
        self.register(composed)
        return composed

    def find_meta_skills(self) -> list[Skill]:
        return [s for s in self.skills.values() if s.is_meta]

    def state_dict(self) -> dict:
        return {
            "skills": {
                name: {
                    "name": s.name,
                    "description": s.description,
                    "code_template": s.code_template,
                    "preconditions": s.preconditions,
                    "postconditions": s.postconditions,
                    "domain": s.domain,
                    "success_rate": s.success_rate,
                    "usage_count": s.usage_count,
                    "is_meta": s.is_meta,
                    "primitive_skills": s.primitive_skills,
                    "version": s.version,
                }
                for name, s in self.skills.items()
            },
        }

    def load_state_dict(self, d: dict) -> None:
        for name, sd in d.get("skills", {}).items():
            self.skills[name] = Skill(
                name=sd["name"],
                description=sd.get("description", ""),
                code_template=sd.get("code_template", ""),
                preconditions=sd.get("preconditions", []),
                postconditions=sd.get("postconditions", []),
                domain=sd.get("domain", "general"),
                success_rate=sd.get("success_rate", 0.0),
                usage_count=sd.get("usage_count", 0),
                is_meta=sd.get("is_meta", False),
                primitive_skills=sd.get("primitive_skills", []),
                version=sd.get("version", 1),
            )
            self.skill_graph[name] = sd.get("primitive_skills", [])


# ──────────────────────────────────────────────
#  WorldModel — probabilistic beliefs
# ──────────────────────────────────────────────


@dataclasses.dataclass
class Belief:
    """A probabilistic belief about the world."""
    proposition: str
    probability: float       # P(proposition)
    confidence: float        # how much evidence supports this [0,1]
    evidence_count: int = 0
    last_updated: float = 0.0


class WorldModel:
    """Probabilistic knowledge base about which strategies/tools work.

    The world model enables the agent to:
    - Simulate outcomes before acting (forward model).
    - Identify epistemic uncertainty (what it doesn't know).
    - Generalize across domains (transfer learning).
    """

    def __init__(self):
        self.beliefs: dict[str, Belief] = {}
        self.domain_transfer: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.causal_links: list[tuple[str, str, float]] = []  # cause → effect → strength

    def observe_causal(self, cause: str, effect: str, strength_delta: float = observe_causal_0_CONST) -> None:
        """Learn a causal relationship from experience."""
        for i, (c, e, s) in enumerate(self.causal_links):
            if c == cause and e == effect:
                self.causal_links[i] = (c, e, min(1.0, s + strength_delta))
                return
        self.causal_links.append((cause, effect, strength_delta))

    def update_belief(self, proposition: str, evidence_success: bool,
                      prior: float | None = None) -> None:
        """Bayesian belief update: P(proposition | evidence)."""
        if proposition not in self.beliefs:
            self.beliefs[proposition] = Belief(
                proposition=proposition,
                probability=prior or 0.5,
                confidence=0.0,
            )
        b = self.beliefs[proposition]
        # Simple beta-distribution update
        alpha = b.probability * b.confidence * 10 + 1 if b.probability > 0 else 1
        beta = (1 - b.probability) * b.confidence * 10 + 1 if b.probability < 1 else 1
        if evidence_success:
            alpha += 1.0
        else:
            beta += 1.0
        b.probability = alpha / (alpha + beta)
        b.confidence = min(1.0, b.confidence + 0.1)
        b.evidence_count += update_belief_1_CONST
        b.last_updated = time.time()

    def epistemic_uncertainty(self, proposition: str) -> float:
        """How uncertain is the agent about this proposition?
        Returns 1 - confidence (0 = very certain, 1 = very uncertain)."""
        if proposition not in self.beliefs:
            return 1.0
        return 1.0 - self.beliefs[proposition].confidence

    def transfer_probability(self, from_domain: str, to_domain: str) -> float:
        """How much a skill from one domain transfers to another."""
        return self.domain_transfer[from_domain].get(to_domain, 0.0)

    def update_transfer(self, from_domain: str, to_domain: str,
                        success: bool, delta: float = update_transfer_0_CONST) -> None:
        old = self.domain_transfer[from_domain][to_domain]
        if success:
            self.domain_transfer[from_domain][to_domain] = min(1.0, old + delta)
        else:
            self.domain_transfer[from_domain][to_domain] = max(0.0, old - delta)

    def simulate(self, action: str, context: dict) -> dict:
        """Forward model: predict outcome of an action given context."""
        # Find most relevant beliefs
        relevant_beliefs = {}
        for prop, belief in self.beliefs.items():
            if any(kw in prop.lower() for kw in action.lower().split()):
                relevant_beliefs[prop] = belief.probability

        # Find causal links
        relevant_causes = [
            (cause, effect, strength)
            for cause, effect, strength in self.causal_links
            if action.lower() in cause.lower()
        ]

        predicted_success = simulate_0_CONST
        if relevant_beliefs:
            predicted_success = sum(relevant_beliefs.values()) / len(relevant_beliefs)
        if relevant_causes:
            predicted_success = min(1.0, predicted_success +
                                     sum(s for _, _, s in relevant_causes) * 0.1)

        return {
            "action": action,
            "predicted_success": predicted_success,
            "relevant_beliefs": relevant_beliefs,
            "relevant_causes": [(c, e, s) for c, e, s in relevant_causes],
        }

    def top_uncertainties(self, n: int = top_uncertainties_5_CONST) -> list[tuple[str, float]]:
        """Return the propositions the agent is most uncertain about."""
        eps = []
        for prop, belief in self.beliefs.items():
            if belief.confidence < 0.7:  # only include low-confidence beliefs
                eps.append((prop, 1.0 - belief.confidence))
        eps.sort(key=lambda x: -x[1])
        return eps[:n]

    def state_dict(self) -> dict:
        return {
            "beliefs": {
                prop: {
                    "proposition": b.proposition,
                    "probability": b.probability,
                    "confidence": b.confidence,
                    "evidence_count": b.evidence_count,
                }
                for prop, b in self.beliefs.items()
            },
            "domain_transfer": {
                fd: dict(td) for fd, td in self.domain_transfer.items()
            },
            "causal_links": [(c, e, s) for c, e, s in self.causal_links],
        }

    def load_state_dict(self, d: dict) -> None:
        for prop, bd in d.get("beliefs", {}).items():
            self.beliefs[prop] = Belief(
                proposition=bd["proposition"],
                probability=bd["probability"],
                confidence=bd["confidence"],
                evidence_count=bd.get("evidence_count", 0),
                last_updated=bd.get("last_updated", 0.0),
            )
        for fd, td in d.get("domain_transfer", {}).items():
            for td_name, strength in td.items():
                self.domain_transfer[fd][td_name] = strength
        self.causal_links = [(c, e, s) for c, e, s in d.get("causal_links", [])]


# ──────────────────────────────────────────────
#  ExperienceMemory — episodic with consolidation
# ──────────────────────────────────────────────


class ExperienceMemory:
    """Episodic memory with consolidation and forgetting curves.

    Implements:
    - Recency-weighted replay buffer.
    - Consolidation: repeated patterns get moved to skill library.
    - Forgetting: rarely accessed memories decay.
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.episodes: deque[TaskOutcome] = deque(maxlen=capacity)
        self.consolidated_patterns: dict[str, list[TaskOutcome]] = defaultdict(list)
        self.access_counts: dict[int, int] = defaultdict(int)

    def store(self, outcome: TaskOutcome) -> None:
        idx = len(self.episodes)
        self.episodes.append(outcome)
        self.access_counts[idx] = store_0_CONST

    def sample(self, n: int = sample_10_CONST, recency_weight: float = sample_2_CONST) -> list[TaskOutcome]:
        """Sample episodes with recency bias."""
        if not self.episodes:
            return []
        indices = list(range(len(self.episodes)))
        weights = [math.exp(recency_weight * i / len(indices))
                   for i in indices]
        total = sum(weights)
        probs = [w / total for w in weights]
        chosen = random.choices(indices, weights=probs, k=min(n, len(self.episodes)))
        for i in chosen:
            self.access_counts[i] += 1
        return [self.episodes[i] for i in chosen]

    def consolidate(self) -> list[TaskOutcome]:
        """Find repeated patterns and move to consolidated memory."""
        # Find task IDs that appear multiple times
        task_counts: dict[str, list[TaskOutcome]] = defaultdict(list)
        for ep in self.episodes:
            task_counts[ep.task_id].append(ep)

        consolidated = []
        for task_id, outcomes in task_counts.items():
            if len(outcomes) >= 3:
                # This pattern is repeating -> consolidate
                success_rate = sum(1 for o in outcomes if o.success) / len(outcomes)
                if success_rate > 0.7:
                    representative = outcomes[-1]
                    self.consolidated_patterns[task_id] = outcomes
                    consolidated.append(representative)

        return consolidated

    def forget(self, threshold_days: float = forget_30_CONST) -> int:
        """Apply forgetting curve — remove episodes not accessed recently.

        Returns number of episodes forgotten.
        """
        now = time.time()
        before = len(self.episodes)
        # We can't easily filter deque, so we rebuild
        kept = [ep for ep in self.episodes
                if now - ep.latency_ms / 1000.0 < threshold_days * 86400]
        self.episodes = deque(kept, maxlen=self.capacity)
        self.access_counts = defaultdict(int)
        return before - len(self.episodes)

    def stats(self) -> dict:
        return {
            "total_episodes": len(self.episodes),
            "consolidated_patterns": len(self.consolidated_patterns),
            "capacity": self.capacity,
            "utilization": len(self.episodes) / self.capacity if self.capacity else 0,
        }

    def state_dict(self) -> dict:
        return {
            "episodes": [
                {
                    "task_id": e.task_id,
                    "task_desc": e.task_desc,
                    "success": e.success,
                    "tools_used": e.tools_used,
                    "reflection": e.reflection,
                    "reward": e.reward,
                    "error": e.error,
                }
                for e in self.episodes
            ],
            "capacity": self.capacity,
        }

    def load_state_dict(self, d: dict) -> None:
        self.capacity = d.get("capacity", 1000)
        self.episodes = deque(maxlen=self.capacity)
        for ed in d.get("episodes", []):
            self.episodes.append(TaskOutcome(
                task_id=ed.get("task_id", ""),
                task_desc=ed.get("task_desc", ""),
                success=ed.get("success", False),
                latency_ms=ed.get("latency_ms", 0.0),
                tools_used=ed.get("tools_used", []),
                policy_decisions=ed.get("policy_decisions", []),
                reflection=ed.get("reflection", ""),
                reward=ed.get("reward", 0.0),
                error=ed.get("error"),
            ))


# ──────────────────────────────────────────────
#  SelfEvolvingAgent — the full metacognitive agent
# ──────────────────────────────────────────────


class SelfEvolvingAgent:
    """A self-evolving agent with full meta-cognitive architecture.

    This agent doesn't just execute tasks — it:
    1. Reflects on its own performance.
    2. Maintains a model of its own capabilities (metacognition).
    3. Learns causal relationships (world model).
    4. Extracts and composes skills.
    5. Generates and tests improvement hypotheses.
    6. Modifies its own code to lock in improvements.

    Usage:
        agent = SelfEvolvingAgent()
        agent.set_goal("Improve code generation quality")
        while agent.step():
            pass
        report = agent.meta_cognition.snapshot()
    """

    def __init__(self, name: str = "selfevolve-1",
                 state_dir: str | None = None):
        self.name = name
        self.state_dir = state_dir
        self.meta_cognition = MetaCognition()
        self.world_model = WorldModel()
        self.skill_library = SkillLibrary()
        self.memory = ExperienceMemory(capacity=2000)
        self.pep = TrainablePEP(name=f"{name}_pep")

        # Current goal
        self.current_goal: str | None = None
        self.goal_history: list[dict] = []

        # Agent's own source (populated at init)
        self.own_source: dict[str, str] = {}

        # Cycle tracking
        self.cycle: int = init___0_CONST
        self.max_cycles: int = 100

        # Self-modification history
        self.modifications: list[dict] = []

        # Register seed skills
        self._seed_skills()

        # Load state if available
        if state_dir:
            self._load_state()

    def _seed_skills(self) -> None:
        seeds = [
            Skill(
                name="reflect",
                description="Reflect on a task outcome and extract lessons",
                code_template=(
                    "Analyze what happened, why it happened, "
                    "and what could be done differently."
                ),
                preconditions=["task_completed"],
                postconditions=["reflection_complete"],
                domain="meta",
                is_meta=True,
            ),
            Skill(
                name="self_evaluate",
                description="Evaluate own performance and identify gaps",
                code_template=(
                    "Rate own performance, identify blind spots, "
                    "and propose improvement targets."
                ),
                preconditions=["reflection_complete"],
                postconditions=["self_evaluation_complete"],
                domain="meta",
                is_meta=True,
            ),
            Skill(
                name="explore",
                description="Try novel approaches in uncertain domains",
                code_template=(
                    "Given high epistemic uncertainty, try a "
                    "randomized approach to gather information."
                ),
                preconditions=["uncertainty_high"],
                postconditions=["exploration_complete"],
                domain="general",
            ),
            Skill(
                name="compose",
                description="Combine two or more skills for a novel task",
                code_template=(
                    "Identify applicable skills, determine order, "
                    "and chain them for the goal."
                ),
                preconditions=["multiple_skills_available"],
                postconditions=["composition_complete"],
                domain="meta",
                is_meta=True,
            ),
            Skill(
                name="hypothesize",
                description="Generate an improvement hypothesis from failure",
                code_template=(
                    "Identify what went wrong, which module/function "
                    "is responsible, and propose a specific change."
                ),
                preconditions=["task_failed"],
                postconditions=["hypothesis_generated"],
                domain="meta",
                is_meta=True,
            ),
        ]
        for s in seeds:
            self.skill_library.register(s)

    # ── Main Loop ──────────────────────────────

    def set_goal(self, goal: str) -> None:
        self.current_goal = goal
        self.goal_history.append({
            "goal": goal,
            "set_at": time.time(),
            "cycle": self.cycle,
        })

    def step(self) -> bool:
        """Execute one meta-cognitive cycle.

        Returns True if the agent should continue, False if it should stop.
        """
        if self.cycle >= self.max_cycles:
            return False

        phase = self.cycle % 4
        self.cycle += 1

        if phase == 0:
            self._phase_execute()
        elif phase == 1:
            self._phase_reflect()
        elif phase == 2:
            self._phase_hypothesize()
        elif phase == 3:
            self._phase_improve()

        return self.cycle < self.max_cycles

    def _phase_execute(self) -> None:
        """Execute a task based on current goal and skills."""
        if not self.current_goal:
            return

        # Simulate execution with currently available tools
        start = time.time()
        task_id = f"cycle_{self.cycle}"

        # Determine which tools are allowed by the PEP
        allowed_tools = []
        for tool in ["analyze", "code_gen", "search", "reflect", "modify"]:
            allowed, spec, reason = self.pep.evaluate(tool, agent_confidence=0.6)
            if allowed:
                allowed_tools.append(tool)

        success = random.random() > 0.3  # placeholder — real execution would call LLM
        latency = (time.time() - start) * 1000

        outcome = TaskOutcome(
            task_id=task_id,
            task_desc=f"Cycle {self.cycle}: {self.current_goal[:60]}",
            success=success,
            latency_ms=latency,
            tools_used=allowed_tools,
            policy_decisions=[{"tool": t, "allowed": True} for t in allowed_tools],
            reflection=f"Executed with tools: {allowed_tools}",
            reward=1.0 if success else 0.0,
        )

        self.memory.store(outcome)
        self.meta_cognition.observe(outcome)
        self.world_model.update_belief(
            f"tool_use_{'_'.join(allowed_tools)}",
            evidence_success=success,
        )

        if success:
            self.skill_library.extract_from_reflection(outcome)

    def _phase_reflect(self) -> None:
        """Analyze recent experience and update self-model."""
        recent = self.memory.sample(n=5)
        successes = sum(1 for r in recent if r.success)
        failures = len(recent) - successes

        reasons = []
        for r in recent:
            if not r.success and r.error:
                reasons.append(r.error)

        self.meta_cognition.blind_spots.update(
            self._extract_blind_spots_from_failures(reasons)
        )

        # If enough failures, generate a self-modification hypothesis
        if failures >= 3:
            for r in recent:
                if not r.success:
                    self.meta_cognition._hypothesize_improvement(r)

    def _extract_blind_spots_from_failures(self, reasons: list[str]) -> list[str]:
        """From failure traces, identify knowledge gaps."""
        spots = []
        for reason in reasons:
            if "tool" in str(reason).lower():
                spots.append("tool_use")
            if "timeout" in str(reason).lower():
                spots.append("speed")
            if "complex" in str(reason).lower():
                spots.append("complex_reasoning")
        return spots

    def _phase_hypothesize(self) -> None:
        """Select and refine the best improvement hypothesis."""
        hypotheses = self.meta_cognition.top_improvement_opportunities(n=3)
        if not hypotheses:
            return

        # Take the top hypothesis and mark it for testing
        best = hypotheses[0]
        best.status = "ready_to_test"
        best.tested_at = time.time()

    def _phase_improve(self) -> None:
        """Attempt a self-improvement by modifying own code logic.

        This phase checks for pending hypotheses and attempts to apply
        the most promising one by generating a patch to the agent's own code.
        """
        ready = [h for h in self.meta_cognition.hypotheses
                 if h.status == "ready_to_test"]
        if not ready:
            return

        hypothesis = ready[0]

        # 1. Read the target source
        source_path = pathlib.Path(hypothesis.target_module)
        try:
            with open(source_path, encoding="utf-8") as f:
                self.own_source[hypothesis.target_module] = f.read()
        except OSError as e:
            # Handle any OS-related errors that might occur when opening the file
            print(f"Error reading source code: {e}")
            return

        # 2. Generate a patch (in a real system this would use an LLM)
        patch = self._generate_patch(hypothesis)
        if patch:
            # 3. Apply patch in a sandboxed way (simulated here)
            self.modifications.append({
                "hypothesis_id": hypothesis.id,
                "patch": patch,
                "applied_at": time.time(),
                "cycle": self.cycle,
            })
            hypothesis.status = "accepted"
            hypothesis.score_delta = 0.1  # simulated improvement
        else:
            hypothesis.status = "rejected"
            hypothesis.score_delta = 0.0

    def _generate_patch(self, hypothesis: ImprovementHypothesis) -> str | None:
        """Generate a patch to improve the agent's own code.

        In production, this would use the LLM to generate code.
        Here we use a heuristic approach for demonstration.
        """
        source = self.own_source.get(hypothesis.target_module, "")
        if not source:
            return None
        # Simulated patch — in production, would modify code logic
        patch_lines = [
            f"--- a/{hypothesis.target_module}",
            f"+++ b/{hypothesis.target_module}",
            f"@@ -1,3 +1,5 @@",
            f"+# Improvement: {hypothesis.description}",
            f"+# Rationale: {hypothesis.rationale[:80]}",
        ]
        return "\n".join(patch_lines)

    # ── Persistence ─────────────────────────────

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "cycle": self.cycle,
            "current_goal": self.current_goal,
            "meta_cognition": self.meta_cognition.state_dict(),
            "world_model": self.world_model.state_dict(),
            "skill_library": self.skill_library.state_dict(),
            "memory": self.memory.state_dict(),
            "pep": self.pep.state_dict(),
            "modifications": self.modifications[-50:],
            "goal_history": self.goal_history[-20:],
        }

    def save_state(self, path: str | None = None) -> str:
        """
        Save the current state to a JSON file.

        Args:
            path (str | None): The path where the state will be saved. If not provided,
                it defaults to a directory named after this instance and 'state.json'.

        Returns:
            str: The path where the state was saved.
        """
        path = path or os.path.join(self.state_dir or ".", f"{self.name}_state.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.state_dict(), f, indent=2, default=str)
        except Exception as e:
            # Handle the exception here if needed
            raise
        return path

    def _load_state(self) -> None:
        if not self.state_dir:
            return
        path = os.path.join(self.state_dir, f"{self.name}_state.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            self.cycle = d.get("cycle", 0)
            self.current_goal = d.get("current_goal")
            self.meta_cognition.load_state_dict(d.get("meta_cognition", {}))
            self.world_model.load_state_dict(d.get("world_model", {}))
            self.skill_library.load_state_dict(d.get("skill_library", {}))
            self.memory.load_state_dict(d.get("memory", {}))
            self.pep.load_state_dict(d.get("pep", {}))
            self.modifications = d.get("modifications", [])

    def report(self) -> str:
        """Generate a self-report of current state."""
        meta = self.meta_cognition.snapshot()
        mem = self.memory.stats()

        lines = [
            "+- SelfEvolvingAgent: %s --" % self.name,
            "|  Goal: %s" % (self.current_goal or "none"),
            "|  Cycle: %d/%d" % (self.cycle, self.max_cycles),
            "|",
            "|  MetaCognition:",
            "|    Total tasks: %d" % meta.total_tasks,
            "|    Success rate: %.1f%%" % (meta.success_rate * 100),
            "|    Improvement rate: %+.3f/cycle" % meta.improvement_rate,
            "|    Blind spots: %d" % len(meta.blind_spots),
            "|    Hypotheses pending: %d" % sum(1 for h in self.meta_cognition.hypotheses if h.status == 'proposed'),
            "|",
            "|  World Model:",
            "|    Beliefs: %d" % len(self.world_model.beliefs),
            "|    Causal links: %d" % len(self.world_model.causal_links),
            "|    Top uncertainties: %d" % len(self.world_model.top_uncertainties()),
            "|",
            "|  Skills: %d (%d meta)" % (len(self.skill_library.skills), len(self.skill_library.find_meta_skills())),
            "|  Memory: %d episodes (%.0f%% capacity)" % (mem['total_episodes'], mem['utilization'] * 100),
            "|  Self-modifications: %d" % len(self.modifications),
            "|  PEP rules: %d (%d trained)" % (len(self.pep.rules), sum(1 for r in self.pep.rules if r.learn_count > 0)),
            "+--",
        ]
        return "\n".join(lines)

    def convergence_score(self) -> dict:
        """Compute AGI convergence metrics — higher is closer to AGI."""
        meta = self.meta_cognition.snapshot()

        # Receptive field: how many cycles of context does the agent use?
        receptive_field = min(1.0, len(self.memory.episodes) / 100)

        # Skill compositionality: what fraction of skills are composed from primitives?
        if self.skill_library.skills:
            compositional = sum(
                1 for s in self.skill_library.skills.values()
                if len(s.primitive_skills) > 0
            )
            compositionality = compositional / len(self.skill_library.skills)
        else:
            compositionality = 0.0

        # Self-improvement rate
        improvement_rate = meta.improvement_rate

        # Meta-cognitive depth: does the agent have a model of itself?
        meta_depth = len(meta.blind_spots) / max(1, len(meta.blind_spots) + 5)

        # Curiosity: epistemic uncertainty coverage
        uncertainty_coverage = len(self.world_model.beliefs) / max(1, len(self.world_model.beliefs) + 20)

        # Self-modification rate
        mod_rate = len(self.modifications) / max(1, self.cycle)

        # Composite AGI score
        agi_score = (
            receptive_field * 0.2 +
            compositionality * 0.2 +
            improvement_rate * 0.15 +
            meta_depth * 0.15 +
            uncertainty_coverage * 0.15 +
            mod_rate * 0.15
        )

        return {
            "receptive_field": round(receptive_field, 3),
            "compositionality": round(compositionality, 3),
            "improvement_rate": round(improvement_rate, 3),
            "meta_depth": round(meta_depth, 3),
            "uncertainty_coverage": round(uncertainty_coverage, 3),
            "self_modification_rate": round(mod_rate, 3),
            "agi_score": round(agi_score, 3),
            "cycles": self.cycle,
            "skills": len(self.skill_library.skills),
            "modifications": len(self.modifications),
        }


# ──────────────────────────────────────────────
#  CuriosityDrive
# ──────────────────────────────────────────────


class CuriosityDrive:
    """Intrinsic motivation engine.

    Drives the agent to explore areas of high epistemic uncertainty,
    generating novel tasks and rewarding information gain.
    """

    def __init__(self, world_model: WorldModel):
        self.world_model = world_model
        self.exploration_history: list[dict] = []
        self.novelty_bonus: float = 0.0

    def generate_exploration_targets(self, n: int = 3) -> list[str]:
        """Generate exploration targets from epistemic uncertainty."""
        uncertainties = self.world_model.top_uncertainties(n=n)
        targets = []
        for prop, unc in uncertainties:
            target = f"explore_{prop[:40]}"
            targets.append(f"Investigate: {prop} (uncertainty={unc:.2f})")
        if not targets:
            targets = ["Explore: novel domain (uncertainty=high)"]
        return targets

    def compute_reward(self, before_entropy: float, after_entropy: float) -> float:
        """Reward proportional to information gain (entropy reduction)."""
        info_gain = before_entropy - after_entropy
        if info_gain > 0:
            self.novelty_bonus = min(2.0, self.novelty_bonus + info_gain)
        else:
            self.novelty_bonus = max(0.0, self.novelty_bonus * 0.95)
        return self.novelty_bonus