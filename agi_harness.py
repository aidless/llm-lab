#!/usr/bin/env python3
"""AGI Self-Evolution Pipeline — end-to-end demo.

This harness demonstrates the full self-evolving agent architecture:

  Phase 1: Single Agent Meta-Cognition     — the agent reflects on own performance
  Phase 2: Recursive Self-Improvement       — the agent patches its own code
  Phase 3: Curiosity-Driven Exploration     — epistemic uncertainty → novel skill discovery
  Phase 4: Multi-Agent Society              — agents teach, debate, specialize
  Phase 5: AGI Convergence Dashboard        — measurement and visualization

Usage:
  python agi_harness.py                        # full demo
  python agi_harness.py --quick                # fast demo (1 cycle each)
  python agi_harness.py --ollama               # use Ollama LLM backend
  python agi_harness.py --ollama --model llama3.2:3b  # pick Ollama model
  python agi_harness.py --single-agent         # Phase 1 only
  python agi_harness.py --recursive            # Phase 2 only
  python agi_harness.py --society              # Phase 4 only
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import textwrap
import time

# Ensure selfevolve is importable
sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

from selfevolve.core import (
    CuriosityDrive,
    SelfEvolvingAgent,
)
from selfevolve.self_modification import (
    CodeAnalyzer,
    OllamaBackend,
    RecursiveSelfImprover,
)
from selfevolve.multi_agent import (
    MultiAgentSociety,
)


# ═══════════════════════════════════════════════
#  Phase 1: Single Agent Meta-Cognition
# ═══════════════════════════════════════════════


def phase_single_agent(n_cycles: int = 12) -> SelfEvolvingAgent:
    """Demonstrate a single self-evolving agent running meta-cognitive cycles.

    The agent:
    - Executes tasks (simulated).
    - Reflects on outcomes.
    - Extracts skills from successes.
    - Generates improvement hypotheses from failures.
    - Updates its world model (causal beliefs).
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Single Agent - Meta-Cognition Loop")
    print("=" * 60)

    agent = SelfEvolvingAgent(
        name="agi-agent-1",
        state_dir=".",
    )
    agent.set_goal("Improve task completion through reflective learning")

    print(f"  Initial state: 0 cycles, {len(agent.skill_library.skills)} seed skills")
    print(f"  Running {n_cycles} meta-cognitive cycles...")

    for i in range(n_cycles):
        continuing = agent.step()
        if i % 4 == 3 or i == n_cycles - 1:
            meta = agent.meta_cognition.snapshot()
            print(f"  Cycle {i + 1:2d}/{n_cycles}: "
                  f"success={meta.success_rate:.0%} "
                  f"skills={len(agent.skill_library.skills)} "
                  f"hypotheses={sum(1 for h in agent.meta_cognition.hypotheses if h.status != 'rejected')} "
                  f"mods={len(agent.modifications)}")
        if not continuing:
            break

    print(f"\n  Final report:")
    print(textwrap.indent(agent.report(), "    "))

    # Show convergence score
    conv = agent.convergence_score()
    print(f"\n  Convergence Score: {conv['agi_score']:.3f}")
    for k, v in conv.items():
        if k != "agi_score":
            print(f"    {k}: {v}")

    return agent


# ═══════════════════════════════════════════════
#  Phase 2: Recursive Self-Improvement
# ═══════════════════════════════════════════════


def phase_recursive_improvement(agent: SelfEvolvingAgent | None = None,
                                n_cycles: int = 3,
                                llm_backend: OllamaBackend | None = None,
                                ) -> RecursiveSelfImprover:
    """Demonstrate the agent improving its own source code.

    The agent:
    - Analyzes its own Python source with AST.
    - Finds complexity hotspots, bare excepts, magic numbers, dead code.
    - Generates patches for the most impactful findings
      (via an optional local LLM backend, otherwise heuristics).
    - Tests patches in a sandbox.
    - Promotes successful patches to its own codebase.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Recursive Self-Improvement")
    print("=" * 60)

    if agent is None:
        agent = SelfEvolvingAgent(name="agi-improver")
        for _ in range(6):
            agent.step()

    improver = RecursiveSelfImprover(
        agent=agent,
        source_root=".",
        auto_promote=True,
        llm_backend=llm_backend,
    )

    print(f"  Analyzing own source code...")
    analyzer = CodeAnalyzer(source_root=".")

    # Analyze the agent's own core module
    findings = analyzer.analyze_module("selfevolve/core.py")
    total_findings = 0
    if isinstance(findings, dict) and "error" not in findings:
        for cat, items in findings.items():
            if items:
                print(f"    {cat}: {len(items)} findings")
                total_findings += len(items)

    print(f"  Total findings: {total_findings}")

    print(f"  Running {n_cycles} self-improvement cycles...")
    for i in range(n_cycles):
        summary = improver.run_cycle(
            target_files=["selfevolve/core.py", "selfevolve/self_modification.py"]
        )
        print(f"  Cycle {i + 1}: "
              f"{summary['findings']} findings, "
              f"{summary['patches_generated']} patches, "
              f"{summary['patches_passed']} passed, "
              f"{summary['patches_promoted']} promoted")

    print(f"\n{improver.summary()}")
    return improver


# ═══════════════════════════════════════════════
#  Phase 3: Curiosity-Driven Exploration
# ═══════════════════════════════════════════════


def phase_curiosity(agent: SelfEvolvingAgent | None = None) -> None:
    """Demonstrate curiosity-driven exploration.

    The agent uses its world model's epistemic uncertainty to
    generate exploration targets, then explores them to gain
    information and discover novel skills.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Curiosity-Driven Exploration")
    print("=" * 60)

    if agent is None:
        agent = SelfEvolvingAgent(name="agi-curious")
        for _ in range(8):
            agent.step()

    curiosity = CuriosityDrive(agent.world_model)

    print(f"  World model beliefs: {len(agent.world_model.beliefs)}")
    print(f"  Causal links: {len(agent.world_model.causal_links)}")

    # Show epistemic uncertainties
    uncertainties = agent.world_model.top_uncertainties(n=5)
    if uncertainties:
        print(f"  Top epistemic uncertainties:")
        for prop, unc in uncertainties:
            print(f"    [{unc:.2f}] {prop[:60]}")
    else:
        print("  (Low uncertainty - agent is confident in most areas)")

    # Generate exploration targets
    targets = curiosity.generate_exploration_targets(n=3)
    print(f"\n  Curiosity generates exploration targets:")
    for t in targets:
        print(f"    -> {t}")

    # Simulate exploration and information gain
    print(f"\n  Simulating exploration...")
    before_entropy = 0.6  # simulated
    after_entropy = 0.3   # simulated — agent learned
    reward = curiosity.compute_reward(before_entropy, after_entropy)
    print(f"    Entropy reduction: {before_entropy:.2f} -> {after_entropy:.2f}")
    print(f"    Novelty reward: {reward:.3f}")
    print(f"    Curiosity bonus accumulated: {curiosity.novelty_bonus:.3f}")


# ═══════════════════════════════════════════════
#  Phase 4: Multi-Agent Society
# ═══════════════════════════════════════════════


def phase_multi_agent() -> MultiAgentSociety:
    """Demonstrate a society of self-evolving agents.

    Three agents specialise in different domains, share skills,
    debate a topic, and collectively improve.
    """
    print("\n" + "=" * 60)
    print("PHASE 4: Multi-Agent Society")
    print("=" * 60)

    society = MultiAgentSociety()

    # Create three specialised agents
    agents_data = [
        ("analyst", "Improve analytical reasoning through structured reflection"),
        ("coder", "Improve code generation patterns and tool use"),
        ("explorer", "Discover novel strategies in unfamiliar domains"),
    ]

    for name, goal in agents_data:
        agent = SelfEvolvingAgent(name=name)
        agent.set_goal(goal)
        society.register_agent(agent)
        print(f"  Created agent [{name}] with goal: {goal[:50]}...")

    # Run each agent for a few cycles
    print(f"\n  Running individual learning cycles...")
    for name in society.agents:
        agent = society.agents[name]
        steps = 0
        for _ in range(6):
            if agent.step():
                steps += 1
        meta = agent.meta_cognition.snapshot()
        print(f"    [{name}] cycles={steps} success={meta.success_rate:.0%} "
              f"skills={len(agent.skill_library.skills)}")

    # Skills spread through society
    print(f"\n  Skill transfer across society...")
    # Find best skill per agent and share
    for source_name in society.agents:
        agent = society.agents[source_name]
        best_skill = None
        best_sr = 0
        for skill in agent.skill_library.skills.values():
            if skill.success_rate > best_sr:
                best_sr = skill.success_rate
                best_skill = skill.name
        if best_skill:
            for target_name in society.agents:
                if target_name != source_name:
                    ok = society.transfer_skill(source_name, target_name, best_skill)
                    if ok:
                        print(f"    {source_name} --[{best_skill}]--> {target_name}")

    # Debate
    print(f"\n  Structured debate: 'Which domain benefits most from self-evolution?'")
    arena = society.debate(
        topic="Which domain benefits most from self-evolution?",
        participants=list(society.agents.keys()),
    )
    consensus = arena.consensus_claim()
    print(f"    Consensus reached: {consensus or 'no consensus'}")

    # Collective improvement
    print(f"\n  Collective improvement on shared goal...")
    results = society.collectively_improve(
        topic="Develop better error recovery strategies"
    )
    for name, r in results.items():
        print(f"    [{name}] {r['cycle']} cycles, {r['skills']} skills, {r['modifications']} mods")

    print(f"\n{society.collective_report()}")
    return society


# ═══════════════════════════════════════════════
#  Phase 5: AGI Convergence Dashboard
# ═══════════════════════════════════════════════


def phase_agi_dashboard(agent: SelfEvolvingAgent | None = None,
                        improver: RecursiveSelfImprover | None = None,
                        society: MultiAgentSociety | None = None) -> None:
    """Compute and display AGI convergence metrics across all phases.

    Measures:
    - Receptive field: temporal depth of agent's memory/planning
    - Skill compositionality: fraction of skills composed from primitives
    - Self-improvement rate: performance delta per improvement cycle
    - Meta-cognitive depth: quality of agent's self-model
    - Cross-domain transfer: how well skills transfer between domains
    - Collective intelligence: benefit of multi-agent coordination
    """
    print("\n" + "=" * 60)
    print("PHASE 5: AGI Convergence Dashboard")
    print("=" * 60)

    if agent is None:
        agent = SelfEvolvingAgent(name="dashboard-agent")
        for _ in range(10):
            agent.step()

    conv = agent.convergence_score()

    sep = "-" * 60
    print("  " + sep)
    print("  %45s  " % "AGI Convergence Metrics")
    print("  " + sep)
    print("  %-30s %6s  %6s" % ("DIMENSION", "SCORE", "WEIGHT"))
    print("  " + sep)
    print("  %-30s %6.3f  %s" % ("Receptive Field", conv['receptive_field'], "x0.20"))
    print("  %-30s %6.3f  %s" % ("Skill Compositionality", conv['compositionality'], "x0.20"))
    print("  %-30s %6.3f  %s" % ("Self-Improvement Rate", conv['improvement_rate'], "x0.15"))
    print("  %-30s %6.3f  %s" % ("Meta-Cognitive Depth", conv['meta_depth'], "x0.15"))
    print("  %-30s %6.3f  %s" % ("Uncertainty Coverage", conv['uncertainty_coverage'], "x0.15"))
    print("  %-30s %6.3f  %s" % ("Self-Modification Rate", conv['self_modification_rate'], "x0.15"))
    print("  " + sep)
    print("  %-30s %6.3f  " % ("AGI SCORE", conv['agi_score']))
    print("  " + sep)

    # If we have multi-agent data, include collective dimension
    if society:
        collective_factor = min(1.0, len(society.agents) * 0.15 +
                                       society.skill_transfer_count * 0.05)
        combined_agi = conv["agi_score"] * (1 + collective_factor)
        print(f"\n  With collective intelligence bonus (x{1 + collective_factor:.2f}):")
        print(f"  COLLECTIVE AGI SCORE: {combined_agi:.3f}")

    # Interpret
    score = conv["agi_score"]
    print(f"\n  Interpretation:")
    if score < 0.2:
        print(f"    Initial state - basic agent with static behavior")
    elif score < 0.4:
        print(f"    Learning - agent reflects and adapts to experience")
    elif score < 0.6:
        print(f"    Self-evolving - agent modifies own code and composes skills")
    elif score < 0.8:
        print(f"    Recursive - positive feedback loop in self-improvement rate")
    else:
        print(f"    Convergent - approaching AGI-capable architecture")

    if improver:
        print(f"\n  Self-modification record: {len(improver.patches_promoted)} promoted / "
              f"{len(improver.patches_rejected)} rejected / "
              f"{len(improver.patches_applied)} pending")


# ═══════════════════════════════════════════════
#  Full Pipeline
# ═══════════════════════════════════════════════


def run_full_pipeline(quick: bool = False,
                      llm_backend: OllamaBackend | None = None) -> dict:
    """Run all phases of the AGI self-evolution pipeline."""
    print("+" + "-" * 58 + "+")
    print("|     AGI Self-Evolution Pipeline                     |")
    print("|     Meta-Cognition -> Recursive Improvement           |")
    print("|     -> Curiosity Drive -> Multi-Agent Society          |")
    print("+" + "-" * 58 + "+")

    n_cycles = 16 if quick else 24
    n_improve = 4 if quick else 6

    results = {}

    t0 = time.time()

    # Phase 1
    agent = phase_single_agent(n_cycles=n_cycles)
    results["phase1_single_agent"] = agent.convergence_score()

    # Phase 2
    improver = phase_recursive_improvement(agent, n_cycles=n_improve,
                                           llm_backend=llm_backend)
    results["phase2_recursive"] = {
        "patches_promoted": len(improver.patches_promoted),
        "patches_rejected": len(improver.patches_rejected),
        "llm_patches": improver.llm_patch_count,
    }

    # Phase 3
    phase_curiosity(agent)
    results["phase3_curiosity"] = {
        "beliefs": len(agent.world_model.beliefs),
        "causal_links": len(agent.world_model.causal_links),
    }

    # Phase 4
    society = phase_multi_agent()
    results["phase4_society"] = {
        "agents": len(society.agents),
        "transfers": society.skill_transfer_count,
        "debates": len(society.debates),
    }

    # Phase 5
    phase_agi_dashboard(agent, improver, society)

    total_time = time.time() - t0

    print(f"\n  Total pipeline time: {total_time:.1f}s")
    print(f"  Final AGI convergence score: {agent.convergence_score()['agi_score']:.3f}")

    results["total_time_s"] = round(total_time, 1)
    results["final_agi_score"] = agent.convergence_score()["agi_score"]
    return results


# ═══════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="AGI Self-Evolution Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python agi_harness.py                    # full pipeline
              python agi_harness.py --quick             # fast demo
              python agi_harness.py --single-agent      # Phase 1 only
              python agi_harness.py --recursive         # Phase 2 only
              python agi_harness.py --society           # Phase 4 only
        """),
    )
    parser.add_argument("--quick", action="store_true",
                        help="Fast demo with fewer cycles")
    parser.add_argument("--ollama", action="store_true",
                        help="Use Ollama LLM backend (requires ollama running)")
    parser.add_argument("--model", type=str, default="llama3.2:3b",
                        help="Ollama model name for LLM patches "
                             "(e.g. llama3.2:3b, qwen2.5:7b)")
    parser.add_argument("--single-agent", action="store_true",
                        help="Run Phase 1 only")
    parser.add_argument("--recursive", action="store_true",
                        help="Run Phase 2 only")
    parser.add_argument("--society", action="store_true",
                        help="Run Phase 4 only")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to file")

    args = parser.parse_args()

    if args.ollama:
        backend = OllamaBackend(model=args.model)
        if backend.is_available():
            print("[OK] Ollama detected. LLM-enhanced patch generation active "
                  "(model: %s)." % backend.model)
            print("     Offline fallback to heuristic patches if requests fail.")
        else:
            backend = None
            print("[!] Ollama not reachable. Falling back to stdlib-only mode.")
    else:
        backend = None

    if args.single_agent:
        phase_single_agent(n_cycles=8)
    elif args.recursive:
        agent = SelfEvolvingAgent(name="recursive-demo")
        for _ in range(6):
            agent.step()
        phase_recursive_improvement(agent, n_cycles=3, llm_backend=backend)
    elif args.society:
        phase_multi_agent()
    else:
        results = run_full_pipeline(quick=args.quick, llm_backend=backend)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
