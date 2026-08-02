"""MultiAgentSociety — agents that teach, debate, specialize, and converge on truth.

Architecture:
  - Peers: equal agents sharing knowledge via skill transfer.
  - Specialist: an agent that focuses on a narrow domain.
  - Debate: structured disagreement to converge on truth.
  - Voting: collective decision-making for high-stakes choices.

This layer enables super-linear scaling: N agents coordinating
can solve problems none could solve alone.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import time
import typing as t

from .core import (
    SelfEvolvingAgent,
    Skill,
    SkillLibrary,
    TaskOutcome,
)


# ──────────────────────────────────────────────
#  Message protocol
# ──────────────────────────────────────────────


@dataclasses.dataclass
class AgentMessage:
    """Message exchanged between agents."""
    sender: str
    recipient: str  # "*" for broadcast
    msg_type: str   # "skill_transfer", "debate", "vote", "teaching"
    payload: dict
    timestamp: float = 0.0
    id: str = ""

    def __post_init__(self):
        self.timestamp = self.timestamp or time.time()
        self.id = self.id or hashlib.md5(
            f"{self.sender}:{self.recipient}:{self.timestamp}:{random.random()}".encode()
        ).hexdigest()[:12]


# ──────────────────────────────────────────────
#  Debate
# ──────────────────────────────────────────────


@dataclasses.dataclass
class DebateRound:
    """A single round of structured debate."""
    topic: str
    positions: list[dict]  # [{"agent": name, "claim": str, "evidence": str}]
    votes: dict[str, str]  # agent_name -> position they voted for
    consensus: str | None = None
    resolved: bool = False


class DebateArena:
    """Structured debate to converge on truth.

    Protocol:
    1. Topic is proposed.
    2. Each agent submits a position with evidence.
    3. Agents review each other's positions and vote.
    4. If consensus emerges (>50%), topic is resolved.
    5. If not, a second round with rebuttals.
    """

    def __init__(self, topic: str):
        self.topic = topic
        self.rounds: list[DebateRound] = []
        self.current_round = DebateRound(topic=topic, positions=[], votes={})
        self.max_rounds = 3

    def submit_position(self, agent_name: str, claim: str, evidence: str) -> None:
        self.current_round.positions.append({
            "agent": agent_name,
            "claim": claim,
            "evidence": evidence,
        })

    def cast_vote(self, agent_name: str, voted_for: str) -> None:
        self.current_round.votes[agent_name] = voted_for

    def resolve(self) -> bool:
        """Try to reach consensus. Returns True if resolved."""
        if not self.current_round.positions:
            return False

        # Count votes
        vote_counts: dict[str, int] = {}
        for v in self.current_round.votes.values():
            vote_counts[v] = vote_counts.get(v, 0) + 1

        if not vote_counts:
            return False

        # If any position has >50%, consensus
        total = sum(vote_counts.values())
        winner = max(vote_counts, key=vote_counts.get)
        if vote_counts[winner] / total > 0.5:
            self.current_round.consensus = winner
            self.current_round.resolved = True
            self.rounds.append(self.current_round)
            return True

        # No consensus yet
        if len(self.rounds) < self.max_rounds:
            self.rounds.append(self.current_round)
            self.current_round = DebateRound(
                topic=self.topic, positions=[], votes={}
            )
            return False  # need another round

        # Max rounds reached — majority wins
        self.current_round.consensus = winner
        self.current_round.resolved = True
        self.rounds.append(self.current_round)
        return True

    def consensus_claim(self) -> str | None:
        for r in self.rounds:
            if r.resolved and r.consensus:
                for p in r.positions:
                    if p["agent"] == r.consensus or p["claim"] == r.consensus:
                        return p["claim"]
        return None


# ──────────────────────────────────────────────
#  MultiAgentSociety
# ──────────────────────────────────────────────


class MultiAgentSociety:
    """A society of self-evolving agents that cooperate, compete, and learn.

    Capabilities:
    - Skill transfer: one agent teaches another.
    - Debate: structured disagreement to converge on truth.
    - Voting: collective decisions for high-confidence outcomes.
    - Specialization: agents develop deep expertise in narrow domains.
    - Collective memory: shared experience pool.
    """

    def __init__(self):
        self.agents: dict[str, SelfEvolvingAgent] = {}
        self.message_log: list[AgentMessage] = []
        self.debates: list[DebateArena] = []
        self.shared_knowledge: dict[str, float] = {}  # proposition → confidence
        self.skill_transfer_count: int = 0

    def register_agent(self, agent: SelfEvolvingAgent) -> None:
        self.agents[agent.name] = agent

    def broadcast(self, sender: str, msg_type: str, payload: dict) -> None:
        msg = AgentMessage(
            sender=sender,
            recipient="*",
            msg_type=msg_type,
            payload=payload,
        )
        self.message_log.append(msg)

    def send(self, sender: str, recipient: str, msg_type: str, payload: dict) -> None:
        msg = AgentMessage(
            sender=sender,
            recipient=recipient,
            msg_type=msg_type,
            payload=payload,
        )
        self.message_log.append(msg)

    def transfer_skill(self, from_agent: str, to_agent: str,
                       skill_name: str) -> bool:
        """Transfer a skill from one agent to another."""
        if from_agent not in self.agents or to_agent not in self.agents:
            return False

        source = self.agents[from_agent]
        target = self.agents[to_agent]

        if skill_name not in source.skill_library.skills:
            return False

        skill = source.skill_library.skills[skill_name]
        # Register a copy in the target's library
        target.skill_library.register(Skill(
            name=f"{skill_name}_from_{from_agent}",
            description=skill.description,
            code_template=skill.code_template,
            preconditions=skill.preconditions,
            postconditions=skill.postconditions,
            domain=skill.domain,
            usage_count=0,
            is_meta=skill.is_meta,
            primitive_skills=skill.primitive_skills,
        ))
        self.skill_transfer_count += 1
        return True

    def debate(self, topic: str, participants: list[str]) -> DebateArena:
        """Run a structured debate among specified agents."""
        arena = DebateArena(topic)

        # Round 1: initial positions
        for name in participants:
            if name in self.agents:
                agent = self.agents[name]
                # Agent constructs its position from its experience
                success_rate = agent.meta_cognition.snapshot().success_rate
                relevant_domain = "general"
                for d, conf in agent.meta_cognition.domain_confidence.items():
                    if d.lower() in topic.lower():
                        relevant_domain = d
                        break
                evidence = (
                    f"Success rate: {success_rate:.1%} | "
                    f"Domain confidence ({relevant_domain}): "
                    f"{agent.meta_cognition.domain_confidence.get(relevant_domain, 0):.2f}"
                )
                arena.submit_position(name, claim=f"Position of {name}", evidence=evidence)

        # Voting round
        for name in participants:
            if name in self.agents:
                # Vote for the position with best evidence (simplified)
                if arena.current_round.positions:
                    best = max(
                        arena.current_round.positions,
                        key=lambda p: random.random()  # simulated evaluation
                    )
                    arena.cast_vote(name, best["agent"])

        arena.resolve()
        self.debates.append(arena)
        return arena

    def collectively_improve(self, topic: str,
                              max_steps_per_agent: int = 8) -> dict:
        """All agents work on the same improvement target, then share results.

        Args:
            topic: Improvement goal for all agents.
            max_steps_per_agent: Max cycles each agent runs (default 8).
                Prevents runaway 100-cycle loop.

        Returns summary of collective improvement.
        """
        results = {}
        for name, agent in self.agents.items():
            agent.set_goal(topic)
            steps = 0
            while agent.step() and steps < max_steps_per_agent:
                steps += 1
            results[name] = {
                "steps": steps,
                "cycle": agent.cycle,
                "success_rate": agent.meta_cognition.snapshot().success_rate,
                "skills": len(agent.skill_library.skills),
                "modifications": len(agent.modifications),
            }

        # Share the best skills among all agents
        best_skill_per_agent = {}
        for name, agent in self.agents.items():
            best = None
            best_sr = 0
            for skill in agent.skill_library.skills.values():
                if skill.success_rate > best_sr:
                    best_sr = skill.success_rate
                    best = skill.name
            if best:
                best_skill_per_agent[name] = best

        for source_name, skill_name in best_skill_per_agent.items():
            for target_name in self.agents:
                if target_name != source_name:
                    self.transfer_skill(source_name, target_name, skill_name)

        return results

    def specialize_agent(self, agent_name: str, domain: str) -> None:
        """Direct an agent to specialize in a domain."""
        if agent_name not in self.agents:
            return
        agent = self.agents[agent_name]
        # Give the agent a focused goal
        agent.set_goal(f"Become expert in {domain}")

    def collective_report(self) -> str:
        lines = [
            "+- MultiAgentSociety --",
            "|  Agents: %d" % len(self.agents),
            "|  Messages exchanged: %d" % len(self.message_log),
            "|  Skill transfers: %d" % self.skill_transfer_count,
            "|  Debates held: %d" % len(self.debates),
            "|",
        ]
        for name, agent in self.agents.items():
            meta = agent.meta_cognition.snapshot()
            lines.append("|  [%s] cycle=%d success=%.0f%% skills=%d mods=%d" % (
                name, agent.cycle, meta.success_rate * 100,
                len(agent.skill_library.skills), len(agent.modifications)))
        lines.append("+--")
        return "\n".join(lines)
