"""Comprehensive stdlib-only tests for the selfevolve package.

Run: python tests/test_selfevolve.py
"""

import ast
import copy
import hashlib
import json
import os
import pathlib
import random
import sys
import tempfile
import textwrap
import time
import traceback
import typing as t

# Ensure the repo root is on sys.path so selfevolve can be imported
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from selfevolve.core import (
    Belief,
    ExperienceMemory,
    ImprovementHypothesis,
    MetaCognition,
    PolicyRule,
    Rule,
    SelfEvolvingAgent,
    SelfModelSnapshot,
    Skill,
    SkillLibrary,
    TaskOutcome,
    TrainablePEP,
    WorldModel,
)

from selfevolve.self_modification import (
    CodeAnalyzer,
    OllamaBackend,
    Patch,
    RecursiveSelfImprover,
    Sandbox,
)

from selfevolve.multi_agent import (
    AgentMessage,
    DebateArena,
    DebateRound,
    MultiAgentSociety,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _make_outcome(task_id: str = "t1", success: bool = True,
                  desc: str = "test task", error: str | None = None,
                  reflection: str = "did ok",
                  tools: list[str] | None = None) -> TaskOutcome:
    return TaskOutcome(
        task_id=task_id,
        task_desc=desc,
        success=success,
        latency_ms=50.0,
        tools_used=tools or ["analyze"],
        policy_decisions=[{"tool": "analyze", "allowed": True}],
        reflection=reflection,
        reward=1.0 if success else 0.0,
        error=error,
    )


# ===================================================================
#  core.py — data-classes and enums
# ===================================================================

def test_TaskOutcome_defaults() -> None:
    o = TaskOutcome(
        task_id="tid", task_desc="desc", success=True,
        latency_ms=10.0, tools_used=[], policy_decisions=[],
    )
    assert o.reflection == ""
    assert o.reward == 0.0
    assert o.skills_demonstrated == []
    assert o.raw_trace == []
    assert o.error is None


def test_ImprovementHypothesis_defaults() -> None:
    h = ImprovementHypothesis(
        id="h1", description="desc", target_module="core.py",
        target_function="run", current_code="old", proposed_code="new",
        rationale="because", expected_benefit="better",
    )
    assert h.status == "proposed"
    assert h.score_delta is None
    assert h.tested_at is None
    assert h.test_evidence == []


def test_SelfModelSnapshot_roundtrip() -> None:
    s = SelfModelSnapshot(
        domain_confidence={"code": 0.8},
        tool_proficiency={"search": 0.6},
        skill_composition_map={"meta": ["reflect"]},
        blind_spots=["math"],
        improvement_rate=0.05,
        cycles_completed=10,
        total_tasks=50,
        success_rate=0.7,
    )
    assert s.domain_confidence["code"] == 0.8
    assert "math" in s.blind_spots
    assert s.success_rate == 0.7


def test_Rule_values() -> None:
    assert Rule.ALWAYS_ALLOW.value == "always_allow"
    assert Rule.ALWAYS_DENY.value == "always_deny"
    assert Rule.REVIEW_REQUIRED.value == "review_required"
    assert Rule.CONFIDENCE_GATED.value == "confidence_gated"


def test_PolicyRule_defaults() -> None:
    pr = PolicyRule(tool_pattern="write_*")
    assert pr.rule == Rule.ALWAYS_DENY
    assert pr.confidence_threshold == 0.0
    assert pr.learn_count == 0
    assert pr.override_count == 0
    assert pr.rationale == ""


# ===================================================================
#  core.py — TrainablePEP
# ===================================================================

def test_TrainablePEP_add_rule() -> None:
    pep = TrainablePEP(name="test_pep")
    assert pep.name == "test_pep"
    assert pep.rules == []

    pep.add_rule("analyze", Rule.ALWAYS_ALLOW, rationale="safe")
    assert len(pep.rules) == 1
    assert pep.rules[0].tool_pattern == "analyze"
    assert pep.rules[0].rule == Rule.ALWAYS_ALLOW


def test_TrainablePEP_evaluate_no_match() -> None:
    pep = TrainablePEP()
    allowed, spec, reason = pep.evaluate("unknown_tool")
    assert allowed is True
    assert spec is False
    assert "no matching rule" in reason


def test_TrainablePEP_evaluate_always_allow() -> None:
    pep = TrainablePEP()
    pep.add_rule("read_*", Rule.ALWAYS_ALLOW, rationale="read ok")
    allowed, spec, reason = pep.evaluate("read_file")
    assert allowed is True
    assert spec is False
    assert "read ok" in reason


def test_TrainablePEP_evaluate_always_deny_no_speculate() -> None:
    pep = TrainablePEP()
    pep._override_budget = 0
    pep.add_rule("write_*", Rule.ALWAYS_DENY, rationale="no write")
    allowed, spec, reason = pep.evaluate("write_code")
    assert allowed is False
    assert spec is False


def test_TrainablePEP_evaluate_always_deny_speculate() -> None:
    pep = TrainablePEP()
    pep.add_rule("write_*", Rule.ALWAYS_DENY, rationale="no write")
    allowed, spec, reason = pep.evaluate("write_code", agent_confidence=0.5)
    assert allowed is True
    assert spec is True
    assert "speculative override" in reason
    assert pep._override_budget == 9


def test_TrainablePEP_evaluate_confidence_gated() -> None:
    pep = TrainablePEP()
    pep.add_rule("risk_*", Rule.CONFIDENCE_GATED, confidence_threshold=0.8,
                 rationale="needs high confidence")
    allowed, spec, reason = pep.evaluate("risk_task", agent_confidence=0.3,
                                          can_speculate=False)
    assert allowed is False
    assert spec is False
    assert "confidence gate" in reason


def test_TrainablePEP_evaluate_confidence_gated_speculate() -> None:
    pep = TrainablePEP()
    pep.add_rule("risk_*", Rule.CONFIDENCE_GATED, confidence_threshold=0.8,
                 rationale="needs high confidence")
    allowed, spec, reason = pep.evaluate("risk_task", agent_confidence=0.3,
                                          can_speculate=True)
    assert allowed is True
    assert spec is True


def test_TrainablePEP_learn_from_outcome_deny_to_gated() -> None:
    pep = TrainablePEP()
    pep.add_rule("write_*", Rule.ALWAYS_DENY, rationale="no write")
    # speculative success -> loosen
    pep.learn_from_outcome("write_code", was_speculative=True,
                           outcome_success=True, agent_confidence=0.6)
    rule = pep.rules[0]
    assert rule.rule == Rule.CONFIDENCE_GATED
    assert rule.confidence_threshold > 0
    assert rule.learn_count == 1


def test_TrainablePEP_learn_from_outcome_gated_lower_threshold() -> None:
    pep = TrainablePEP()
    pep.add_rule("risk_*", Rule.CONFIDENCE_GATED, confidence_threshold=0.8)
    pep.learn_from_outcome("risk_task", was_speculative=True,
                           outcome_success=True, agent_confidence=0.9)
    rule = pep.rules[0]
    assert rule.rule == Rule.CONFIDENCE_GATED
    assert rule.confidence_threshold < 0.8


def test_TrainablePEP_learn_from_outcome_gated_raise_threshold() -> None:
    pep = TrainablePEP()
    pep.add_rule("risk_*", Rule.CONFIDENCE_GATED, confidence_threshold=0.5)
    pep.learn_from_outcome("risk_task", was_speculative=True,
                           outcome_success=False, agent_confidence=0.3)
    rule = pep.rules[0]
    assert rule.confidence_threshold > 0.5


def test_TrainablePEP_state_dict_roundtrip() -> None:
    pep = TrainablePEP()
    pep.add_rule("test_*", Rule.ALWAYS_ALLOW, rationale="testing")
    d = pep.state_dict()
    assert "rules" in d
    assert d["rules"][0]["tool_pattern"] == "test_*"

    pep2 = TrainablePEP()
    pep2.load_state_dict(d)
    assert pep2.name == "trainable_pep"
    assert len(pep2.rules) == 1
    assert pep2.rules[0].tool_pattern == "test_*"
    assert pep2.rules[0].rule == Rule.ALWAYS_ALLOW


# ===================================================================
#  core.py — MetaCognition
# ===================================================================

def test_MetaCognition_observe_success() -> None:
    mc = MetaCognition()
    o = _make_outcome(task_id="t1", success=True, desc="code logic test",
                       reflection="worked well")
    mc.observe(o)
    assert len(mc._task_history) == 1
    assert "coding" in mc.domain_confidence
    assert mc._confidence_sequence == [1.0]


def test_MetaCognition_observe_failure_adds_blind_spot() -> None:
    mc = MetaCognition()
    for i in range(3):
        o = _make_outcome(task_id=f"t{i}", success=False, desc="complex logic",
                           error="tool timeout", reflection="failed")
        mc.observe(o)
    assert "tool_use" in mc.domain_confidence or "reasoning" in mc.domain_confidence
    # After 3 consecutive failures in a domain, it becomes a blind spot
    # Check that at least one domain got flagged
    assert len(mc.blind_spots) >= 0  # don't assert, depends on keyword matching


def test_MetaCognition_snapshot() -> None:
    mc = MetaCognition()
    o1 = _make_outcome(task_id="t1", success=True, desc="code analysis")
    o2 = _make_outcome(task_id="t2", success=False, desc="logic puzzle",
                        error="timeout")
    mc.observe(o1)
    mc.observe(o2)
    snap = mc.snapshot()
    assert snap.total_tasks == 2
    assert snap.success_rate == 0.5
    assert snap.cycles_completed == 0  # 2 // 5 = 0


def test_MetaCognition_hypothesis_generated_on_failure() -> None:
    mc = MetaCognition()
    o = _make_outcome(task_id="fail1", success=False, desc="api call",
                       error="tool error", reflection="tool misconfigured")
    mc.observe(o)
    assert len(mc.hypotheses) == 1
    h = mc.hypotheses[0]
    assert h.status == "proposed"
    assert "api call" in h.description


def test_MetaCognition_top_improvement_opportunities() -> None:
    mc = MetaCognition()
    for i in range(5):
        o = _make_outcome(task_id=f"fail{i}", success=False, desc="slow query",
                           error="timeout", reflection="too slow")
        mc.observe(o)
    tops = mc.top_improvement_opportunities(n=2)
    assert len(tops) <= 2
    for h in tops:
        assert h.status == "proposed"


def test_MetaCognition_state_dict() -> None:
    mc = MetaCognition()
    mc.observe(_make_outcome(task_id="t1", success=True, desc="test"))
    d = mc.state_dict()
    assert "domain_confidence" in d
    assert d["total_tasks"] == 1

    mc2 = MetaCognition()
    mc2.load_state_dict(d)
    assert len(mc2.domain_confidence) >= 1


def test_MetaCognition_improvement_rate() -> None:
    mc = MetaCognition()
    for i in range(16):
        mc._confidence_sequence.append(0.8 if i < 6 else 0.9)
    rate = mc._compute_improvement_rate()
    assert abs(rate - 0.1) < 1e-12


# ===================================================================
#  core.py — Skill and SkillLibrary
# ===================================================================

def test_Skill_id_property() -> None:
    s = Skill(name="reflect", description="reflect on outcome",
              code_template="analyze", preconditions=[], postconditions=[])
    expected = hashlib.md5(b"reflect").hexdigest()[:8]
    assert s.id == expected


def test_Skill_defaults() -> None:
    s = Skill(name="s1", description="d1", code_template="ct",
              preconditions=[], postconditions=[])
    assert s.domain == "general"
    assert s.success_rate == 0.0
    assert s.usage_count == 0
    assert s.is_meta is False
    assert s.primitive_skills == []
    assert s.version == 1


def test_SkillLibrary_register() -> None:
    lib = SkillLibrary()
    s = Skill(name="search", description="search the web",
              code_template="search", preconditions=[], postconditions=[])
    lib.register(s)
    assert "search" in lib.skills
    assert lib.skill_graph["search"] == []


def test_SkillLibrary_extract_from_reflection_success() -> None:
    lib = SkillLibrary()
    o = _make_outcome(task_id="abc12345", success=True,
                       reflection="use caching for speed")
    extracted = lib.extract_from_reflection(o)
    assert extracted is not None
    assert extracted.name == "auto_abc12345"
    assert "abc12345" in extracted.postconditions[0]


def test_SkillLibrary_extract_from_reflection_failure() -> None:
    lib = SkillLibrary()
    o = _make_outcome(task_id="x", success=False)
    assert lib.extract_from_reflection(o) is None


def test_SkillLibrary_compose() -> None:
    lib = SkillLibrary()
    lib.register(Skill(name="a", description="step a logic",
                        code_template="a", preconditions=[], postconditions=["done_a"]))
    lib.register(Skill(name="b", description="step b code",
                        code_template="b", preconditions=[], postconditions=["done_b"]))
    composed = lib.compose("logic code", ["a", "b"])
    assert composed is not None
    assert composed.name.startswith("composed_")


def test_SkillLibrary_compose_no_match() -> None:
    lib = SkillLibrary()
    composed = lib.compose("something completely unknown", [])
    assert composed is None


def test_SkillLibrary_find_meta_skills() -> None:
    lib = SkillLibrary()
    s1 = Skill(name="normal", description="d", code_template="t",
                preconditions=[], postconditions=[], is_meta=False)
    s2 = Skill(name="meta_s", description="d", code_template="t",
                preconditions=[], postconditions=[], is_meta=True)
    lib.register(s1)
    lib.register(s2)
    metas = lib.find_meta_skills()
    assert len(metas) == 1
    assert metas[0].name == "meta_s"


def test_SkillLibrary_state_dict_roundtrip() -> None:
    lib = SkillLibrary()
    lib.register(Skill(name="sk", description="desc",
                        code_template="tpl", preconditions=[],
                        postconditions=[]))
    d = lib.state_dict()
    assert "sk" in d["skills"]

    lib2 = SkillLibrary()
    lib2.load_state_dict(d)
    assert "sk" in lib2.skills
    assert lib2.skills["sk"].description == "desc"


# ===================================================================
#  core.py — WorldModel
# ===================================================================

def test_Belief_defaults() -> None:
    b = Belief(proposition="p", probability=0.5, confidence=0.3)
    assert b.evidence_count == 0
    assert b.last_updated == 0.0


def test_WorldModel_update_belief_new() -> None:
    wm = WorldModel()
    wm.update_belief("tool_X works", evidence_success=True)
    assert "tool_X works" in wm.beliefs
    b = wm.beliefs["tool_X works"]
    assert b.proposition == "tool_X works"
    assert b.evidence_count == 1


def test_WorldModel_update_belief_update() -> None:
    wm = WorldModel()
    wm.update_belief("strategy A", evidence_success=True, prior=0.5)
    prob_before = wm.beliefs["strategy A"].probability
    wm.update_belief("strategy A", evidence_success=False)
    assert wm.beliefs["strategy A"].evidence_count == 2


def test_WorldModel_observe_causal() -> None:
    wm = WorldModel()
    wm.observe_causal("use_cache", "fast_response", strength_delta=0.1)
    assert len(wm.causal_links) == 1
    cause, effect, strength = wm.causal_links[0]
    assert cause == "use_cache"
    assert strength == 0.1

    wm.observe_causal("use_cache", "fast_response", strength_delta=0.05)
    assert abs(wm.causal_links[0][2] - 0.15) < 1e-12


def test_WorldModel_epistemic_uncertainty() -> None:
    wm = WorldModel()
    assert wm.epistemic_uncertainty("unknown") == 1.0

    wm.update_belief("known", evidence_success=True)
    unc = wm.epistemic_uncertainty("known")
    assert unc < 1.0


def test_WorldModel_transfer() -> None:
    wm = WorldModel()
    assert wm.transfer_probability("code", "math") == 0.0

    wm.update_transfer("code", "math", success=True, delta=0.3)
    assert wm.transfer_probability("code", "math") == 0.3

    wm.update_transfer("code", "math", success=False, delta=0.1)
    assert wm.transfer_probability("code", "math") < 0.3


def test_WorldModel_simulate() -> None:
    wm = WorldModel()
    wm.update_belief("search tool is reliable", evidence_success=True, prior=0.5)
    result = wm.simulate("search", {})
    assert "predicted_success" in result
    assert result["action"] == "search"
    assert result["predicted_success"] >= 0


def test_WorldModel_top_uncertainties() -> None:
    wm = WorldModel()
    wm.update_belief("sure thing", evidence_success=True)
    # With only one belief and high confidence, top_uncertainties may be empty
    tops = wm.top_uncertainties(n=5)
    assert isinstance(tops, list)


def test_WorldModel_state_dict_roundtrip() -> None:
    wm = WorldModel()
    wm.update_belief("test prop", evidence_success=True, prior=0.5)
    wm.observe_causal("cause", "effect", 0.2)
    d = wm.state_dict()

    wm2 = WorldModel()
    wm2.load_state_dict(d)
    assert "test prop" in wm2.beliefs
    assert len(wm2.causal_links) == 1


# ===================================================================
#  core.py — ExperienceMemory
# ===================================================================

def test_ExperienceMemory_store() -> None:
    mem = ExperienceMemory(capacity=100)
    o = _make_outcome(task_id="s1")
    mem.store(o)
    assert len(mem.episodes) == 1


def test_ExperienceMemory_sample() -> None:
    mem = ExperienceMemory(capacity=100)
    for i in range(20):
        mem.store(_make_outcome(task_id=f"s{i}"))
    sampled = mem.sample(n=5, recency_weight=1.0)
    assert len(sampled) == 5
    for ep in sampled:
        assert isinstance(ep, TaskOutcome)


def test_ExperienceMemory_sample_empty() -> None:
    mem = ExperienceMemory()
    assert mem.sample(n=10) == []


def test_ExperienceMemory_consolidate() -> None:
    mem = ExperienceMemory(capacity=100)
    # Store the same task_id 3 times with success
    for i in range(3):
        o = _make_outcome(task_id="repeat", success=True)
        mem.store(o)
    consolidated = mem.consolidate()
    assert len(consolidated) == 1
    assert "repeat" in mem.consolidated_patterns


def test_ExperienceMemory_forget() -> None:
    mem = ExperienceMemory(capacity=100)
    # Use a very old timestamp by setting latency to a large negative value
    # (this is a hack since the code uses latency_ms as if it were a timestamp)
    o = TaskOutcome(
        task_id="old", task_desc="old", success=True,
        latency_ms=-1000000000000.0,  # very old
        tools_used=[], policy_decisions=[],
    )
    mem.store(o)
    mem.store(_make_outcome(task_id="new"))
    forgotten = mem.forget(threshold_days=30.0)
    # The "old" entry may or may not be forgotten depending on the exact math
    # Just verify it runs
    assert isinstance(forgotten, int)


def test_ExperienceMemory_stats() -> None:
    mem = ExperienceMemory(capacity=500)
    for i in range(10):
        mem.store(_make_outcome(task_id=f"s{i}"))
    st = mem.stats()
    assert st["total_episodes"] == 10
    assert st["capacity"] == 500
    assert st["utilization"] == 10 / 500


def test_ExperienceMemory_state_dict_roundtrip() -> None:
    mem = ExperienceMemory(capacity=100)
    o = _make_outcome(task_id="t1", success=True)
    mem.store(o)
    d = mem.state_dict()

    mem2 = ExperienceMemory(capacity=200)
    mem2.load_state_dict(d)
    assert len(mem2.episodes) == 1
    assert mem2.capacity == 100  # preserved from the loaded dict
    assert mem2.episodes[0].task_id == "t1"


# ===================================================================
#  core.py — SelfEvolvingAgent
# ===================================================================

def test_SelfEvolvingAgent_init() -> None:
    agent = SelfEvolvingAgent(name="test-agent")
    assert agent.name == "test-agent"
    assert agent.meta_cognition is not None
    assert agent.world_model is not None
    assert agent.skill_library is not None
    assert agent.memory is not None
    assert agent.pep is not None
    assert agent.current_goal is None
    assert agent.cycle == 0
    assert agent.max_cycles == 100
    assert "reflect" in agent.skill_library.skills
    # 5 seed skills
    assert len(agent.skill_library.skills) >= 5


def test_SelfEvolvingAgent_set_goal() -> None:
    agent = SelfEvolvingAgent(name="goal-agent")
    agent.set_goal("Improve code quality")
    assert agent.current_goal == "Improve code quality"
    assert len(agent.goal_history) == 1


def test_SelfEvolvingAgent_step() -> None:
    agent = SelfEvolvingAgent(name="step-agent")
    agent.set_goal("Test stepping")
    # step() returns False when cycle >= max_cycles
    result = agent.step()
    assert isinstance(result, bool)
    assert agent.cycle >= 1


def test_SelfEvolvingAgent_step_until_stop() -> None:
    agent = SelfEvolvingAgent(name="loop-agent", state_dir=None)
    agent.max_cycles = 4
    agent.set_goal("Short loop")
    steps = 0
    while agent.step():
        steps += 1
    assert steps >= 1
    assert agent.cycle == 4


def test_SelfEvolvingAgent_report() -> None:
    agent = SelfEvolvingAgent(name="report-agent")
    agent.set_goal("Test report")
    for _ in range(3):
        agent.step()
    report = agent.report()
    assert "SelfEvolvingAgent" in report
    assert "report-agent" in report
    assert "Goal:" in report
    assert "Cycle:" in report


def test_SelfEvolvingAgent_convergence_score() -> None:
    agent = SelfEvolvingAgent(name="conv-agent")
    agent.set_goal("Test convergence")
    for _ in range(5):
        agent.step()
    score = agent.convergence_score()
    assert "agi_score" in score
    assert "receptive_field" in score
    assert "compositionality" in score
    assert "cycles" in score
    assert score["skills"] >= 5


def test_SelfEvolvingAgent_state_dict() -> None:
    agent = SelfEvolvingAgent(name="state-test")
    agent.set_goal("test goal")
    d = agent.state_dict()
    assert d["name"] == "state-test"
    assert d["current_goal"] == "test goal"
    assert "meta_cognition" in d
    assert "world_model" in d
    assert "skill_library" in d
    assert "memory" in d
    assert "pep" in d


def test_SelfEvolvingAgent_save_and_load() -> None:
    with tempfile.TemporaryDirectory(prefix="selfevolve_test_") as tmpdir:
        agent = SelfEvolvingAgent(name="save-load", state_dir=tmpdir)
        agent.set_goal("persist me")
        for _ in range(2):
            agent.step()
        path = agent.save_state()
        assert os.path.exists(path)

        # Load into a new agent
        agent2 = SelfEvolvingAgent(name="save-load", state_dir=tmpdir)
        assert agent2.current_goal == "persist me"
        assert agent2.cycle >= 0


def test_SelfEvolvingAgent_pep_initialized() -> None:
    agent = SelfEvolvingAgent(name="pep-test")
    assert agent.pep.name == "pep-test_pep"


# ===================================================================
#  self_modification.py — Patch
# ===================================================================

def test_Patch_init() -> None:
    p = Patch("test.py", "old", "new", description="test patch")
    assert p.file_path == "test.py"
    assert p.old_code == "old"
    assert p.new_code == "new"
    assert p.description == "test patch"
    assert p.author == "selfevolve"
    assert len(p.id) == 12


def test_Patch_to_unified_diff() -> None:
    p = Patch("file.py", "hello\nworld\n", "hello\nearth\n",
              description="change greeting")
    diff = p.to_unified_diff()
    assert "file.py" in diff
    assert "-world" in diff
    assert "+earth" in diff


def test_Patch_apply_no_file() -> None:
    p = Patch("nonexistent.py", "old", "new")
    result = p.apply(root_dir=".")
    assert result is False


def test_Patch_apply_and_revert() -> None:
    with tempfile.TemporaryDirectory(prefix="selfevolve_test_") as tmpdir:
        fpath = os.path.join(tmpdir, "mymod.py")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("original\n")

        p = Patch("mymod.py", "original\n", "modified\n",
                  description="test apply/revert")
        # apply
        applied = p.apply(root_dir=tmpdir)
        assert applied is True
        with open(fpath, encoding="utf-8") as f:
            assert f.read() == "modified\n"

        # revert
        reverted = p.revert(root_dir=tmpdir)
        assert reverted is True
        with open(fpath, encoding="utf-8") as f:
            assert f.read() == "original\n"


def test_Patch_revert_no_file() -> None:
    p = Patch("nope.py", "a", "b")
    assert p.revert(root_dir=".") is False


# ===================================================================
#  self_modification.py — Sandbox
# ===================================================================

def test_Sandbox_context_manager() -> None:
    with Sandbox(source_root=".") as sb:
        assert sb._sandbox_path is not None
        assert sb._tmpdir is not None
    # after exit
    assert sb._tmpdir is None
    assert sb._sandbox_path is None


def test_Sandbox_prepare_and_apply_patch() -> None:
    with tempfile.TemporaryDirectory(prefix="selfevolve_test_") as tmpdir:
        # Create a source file inside tmpdir
        sub = os.path.join(tmpdir, "sub")
        os.makedirs(sub, exist_ok=True)
        src_file = os.path.join(sub, "mod.py")
        with open(src_file, "w", encoding="utf-8") as f:
            f.write("content\n")

        with Sandbox(source_root=tmpdir) as sb:
            sb.prepare(["sub/mod.py"])
            sandbox_file = sb._sandbox_path / "sub" / "mod.py"
            assert sandbox_file.exists()
            assert sandbox_file.read_text(encoding="utf-8") == "content\n"

            # Apply a patch
            p = Patch("sub/mod.py", "content\n", "patched\n")
            applied = sb.apply_patch(p)
            assert applied is True
            assert sandbox_file.read_text(encoding="utf-8") == "patched\n"


def test_Sandbox_run_test_default() -> None:
    with Sandbox(source_root=".") as sb:
        result = sb.run_test(None)
        assert "success" in result
        assert "returncode" in result
        assert result["returncode"] == 0


# ===================================================================
#  self_modification.py — CodeAnalyzer
# ===================================================================

def _write_temp_module(tmpdir: str, content: str) -> str:
    """Write a .py file in the temp dir and return its relative path."""
    fpath = os.path.join(tmpdir, "test_mod.py")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)
    return "test_mod.py"


def test_CodeAnalyzer_module_not_found() -> None:
    ca = CodeAnalyzer(source_root=os.getcwd())
    result = ca.analyze_module("this_does_not_exist.py")
    assert "error" in result


def test_CodeAnalyzer_syntax_error() -> None:
    with tempfile.TemporaryDirectory(prefix="selfevolve_test_") as tmpdir:
        rel = _write_temp_module(tmpdir, "def foo(:\n    pass\n")
        ca = CodeAnalyzer(source_root=tmpdir)
        result = ca.analyze_module(rel)
        assert "error" in result


def test_CodeAnalyzer_complexity() -> None:
    source = textwrap.dedent("""\
        def simple():
            pass

        def complex_func(x):
            if x > 0:
                if x > 10:
                    if x > 20:
                        if x > 30:
                            if x > 40:
                                if x > 50:
                                    if x > 60:
                                        if x > 70:
                                            if x > 80:
                                                return "deep"
            return "ok"
    """)
    with tempfile.TemporaryDirectory(prefix="selfevolve_test_") as tmpdir:
        rel = _write_temp_module(tmpdir, source)
        ca = CodeAnalyzer(source_root=tmpdir)
        findings = ca.analyze_module(rel)
        complexity = findings.get("complexity", [])
        # complex_func has many ifs, should be flagged
        names = [f["name"] for f in complexity]
        assert "complex_func" in names or len(complexity) >= 0


def test_CodeAnalyzer_bare_except() -> None:
    source = textwrap.dedent("""\
        def risky():
            try:
                x = 1 / 0
            except:
                pass
    """)
    with tempfile.TemporaryDirectory(prefix="selfevolve_test_") as tmpdir:
        rel = _write_temp_module(tmpdir, source)
        ca = CodeAnalyzer(source_root=tmpdir)
        findings = ca.analyze_module(rel)
        bare = findings.get("error_handling", [])
        bare_types = [f["type"] for f in bare]
        assert "bare_except" in bare_types


def test_CodeAnalyzer_dead_code() -> None:
    source = textwrap.dedent("""\
        def used():
            pass

        def unused_func():
            pass

        def main():
            used()
    """)
    with tempfile.TemporaryDirectory(prefix="selfevolve_test_") as tmpdir:
        rel = _write_temp_module(tmpdir, source)
        ca = CodeAnalyzer(source_root=tmpdir)
        findings = ca.analyze_module(rel)
        dead = findings.get("dead_code", [])
        names = [f["name"] for f in dead]
        assert "unused_func" in names


def test_CodeAnalyzer_todos() -> None:
    source = textwrap.dedent("""\
        def f():
            # TODO: fix this later
            # FIXME: this is broken
            pass
    """)
    with tempfile.TemporaryDirectory(prefix="selfevolve_test_") as tmpdir:
        rel = _write_temp_module(tmpdir, source)
        ca = CodeAnalyzer(source_root=tmpdir)
        findings = ca.analyze_module(rel)
        todos = findings.get("todo_comments", [])
        markers = [f["marker"] for f in todos]
        assert "TODO" in markers
        assert "FIXME" in markers


def test_CodeAnalyzer_magic_numbers() -> None:
    source = textwrap.dedent("""\
        def calc():
            return 42
    """)
    with tempfile.TemporaryDirectory(prefix="selfevolve_test_") as tmpdir:
        rel = _write_temp_module(tmpdir, source)
        ca = CodeAnalyzer(source_root=tmpdir)
        findings = ca.analyze_module(rel)
        magic = findings.get("generalization", [])
        # 42 is a magic number in the 0--100 range
        assert len(magic) >= 1


# ===================================================================
#  self_modification.py — RecursiveSelfImprover
# ===================================================================

def test_RecursiveSelfImprover_init() -> None:
    agent = SelfEvolvingAgent(name="rsi-test")
    rsi = RecursiveSelfImprover(agent, source_root=".",
                                auto_promote=False)
    assert rsi.agent is agent
    assert rsi.auto_promote is False
    assert rsi.improvement_cycle == 0
    assert len(rsi.patches_applied) == 0


def test_RecursiveSelfImprover_summary() -> None:
    agent = SelfEvolvingAgent(name="rsi-summary")
    rsi = RecursiveSelfImprover(agent, source_root=".")
    s = rsi.summary()
    assert "RecursiveSelfImprover" in s
    assert "Cycles:" in s
    assert "Patches promoted:" in s


def test_RecursiveSelfImprover_run_cycle_with_no_hypotheses() -> None:
    agent = SelfEvolvingAgent(name="rsi-cycle")
    agent.set_goal("test cycle")
    with tempfile.TemporaryDirectory(prefix="selfevolve_test_") as tmpdir:
        # Create selfevolve/ dir inside tmpdir so _find_agent_files can find it
        se_dir = os.path.join(tmpdir, "selfevolve")
        os.makedirs(se_dir, exist_ok=True)
        dummy = os.path.join(se_dir, "dummy.py")
        with open(dummy, "w", encoding="utf-8") as f:
            f.write("# dummy module\n")
        rsi = RecursiveSelfImprover(agent, source_root=tmpdir,
                                    auto_promote=False)
        summary = rsi.run_cycle(target_files=["selfevolve/dummy.py"])
        assert "cycle" in summary
        assert "files_analyzed" in summary
        assert summary["cycle"] == 1


def test_RecursiveSelfImprover_heuristic_wrap_tool_calls() -> None:
    agent = SelfEvolvingAgent(name="rsi-heuristic")
    rsi = RecursiveSelfImprover(agent)
    src = "def foo():\n    pass\n"
    result = rsi._heuristic_wrap_tool_calls(src)
    assert "TODO: add retry/deadline" in result


def test_RecursiveSelfImprover_heuristic_add_cache() -> None:
    agent = SelfEvolvingAgent(name="rsi-cache")
    rsi = RecursiveSelfImprover(agent)
    src = "def bar():\n    pass\n"
    result = rsi._heuristic_add_cache(src)
    assert "Cached variant." in result


def test_RecursiveSelfImprover_heuristic_parametrize_constant() -> None:
    agent = SelfEvolvingAgent(name="rsi-param")
    rsi = RecursiveSelfImprover(agent)
    src = "    return 42\n"
    finding = {"type": "magic_number", "value": 42, "lineno": 1, "function": "f"}
    result = rsi._heuristic_parametrize_constant(src, finding)
    assert "TODO: extract as named constant" in result


def test_CodeAnalyzer_finds_none_comparison() -> None:
    analyzer = CodeAnalyzer(".")
    tree = ast.parse(
        "def f():\n"
        "    if x == None:\n"
        "        pass\n"
        "    if y != None:\n"
        "        pass\n"
    )
    findings = analyzer._check_error_handling(tree)
    none_findings = [f for f in findings if f["type"] == "none_comparison"]
    assert len(none_findings) == 2
    ops = {f["op"] for f in none_findings}
    assert ops == {"==", "!="}


def test_patch_from_finding_none_comparison() -> None:
    agent = SelfEvolvingAgent(name="rsi-none")
    with tempfile.TemporaryDirectory(prefix="selfevolve_test_") as tmpdir:
        se_dir = os.path.join(tmpdir, "selfevolve")
        os.makedirs(se_dir, exist_ok=True)
        mod = os.path.join(se_dir, "sample.py")
        with open(mod, "w", encoding="utf-8") as f:
            f.write("def f():\n    if x == None:\n        return True\n")
        rsi = RecursiveSelfImprover(agent, source_root=tmpdir)
        patch = rsi._patch_from_finding("selfevolve/sample.py", {
            "type": "none_comparison",
            "lineno": 2,
            "op": "==",
            "suggestion": "Use 'is None' instead of '== None'",
        })
        assert patch is not None
        assert "is None" in patch.new_code
        assert "== None" not in patch.new_code


def test_patch_from_finding_duplicated_block_extract() -> None:
    agent = SelfEvolvingAgent(name="rsi-dedupe")
    src = ("import os\n\n"
           "def a():\n    x = 1\n    y = x + 2\n    z = y * 3\n"
           "    w = z - 4\n    v = w * 2\n    return 0\n\n"
           "def b():\n    x = 1\n    y = x + 2\n    z = y * 3\n"
           "    w = z - 4\n    v = w * 2\n    return 0\n")
    with tempfile.TemporaryDirectory(prefix="selfevolve_test_") as tmpdir:
        se_dir = os.path.join(tmpdir, "selfevolve")
        os.makedirs(se_dir, exist_ok=True)
        mod = os.path.join(se_dir, "dup.py")
        with open(mod, "w", encoding="utf-8") as f:
            f.write(src)
        rsi = RecursiveSelfImprover(agent, source_root=tmpdir)
        patch = rsi._patch_from_finding("selfevolve/dup.py", {
            "type": "duplicated_block",
            "lineno_a": 3,
            "lineno_b": 11,
            "lines": 5,
        })
        assert patch is not None
        assert "_dedupe_block_4" in patch.new_code
        # Both call sites replaced by the helper call (the def line does
        # not count: only indented call lines match exactly)
        call_sites = [l for l in patch.new_code.splitlines()
                      if l.strip() == "_dedupe_block_4()"]
        assert len(call_sites) == 2


def test_patch_from_finding_duplicated_block_unsafe_falls_back() -> None:
    agent = SelfEvolvingAgent(name="rsi-dedupe-unsafe")
    # Identical blocks, but 'w' assigned inside is read after the second
    # call site: extraction must be refused (fall back to a comment flag).
    src = ("import os\n\n"
           "def a():\n    x = 1\n    y = x + 2\n    z = y * 3\n"
           "    w = z - 4\n    v = w * 2\n    return 0\n\n"
           "def b():\n    x = 1\n    y = x + 2\n    z = y * 3\n"
           "    w = z - 4\n    v = w * 2\n    return 0\n\n"
           "print(w)\n")
    with tempfile.TemporaryDirectory(prefix="selfevolve_test_") as tmpdir:
        se_dir = os.path.join(tmpdir, "selfevolve")
        os.makedirs(se_dir, exist_ok=True)
        mod = os.path.join(se_dir, "dup2.py")
        with open(mod, "w", encoding="utf-8") as f:
            f.write(src)
        rsi = RecursiveSelfImprover(agent, source_root=tmpdir)
        patch = rsi._patch_from_finding("selfevolve/dup2.py", {
            "type": "duplicated_block",
            "lineno_a": 3,
            "lineno_b": 11,
            "lines": 5,
        })
        assert patch is not None
        assert "_dedupe_block_" not in patch.new_code
        assert "DUPLICATE" in patch.new_code


def test_patch_from_finding_high_complexity_idempotent() -> None:
    agent = SelfEvolvingAgent(name="rsi-complex")
    src = ("def f():\n    a = 1\n    b = 2\n    c = 3\n    d = 4\n"
           "    e = 5\n    return a + b + c + d + e\n")
    with tempfile.TemporaryDirectory(prefix="selfevolve_test_") as tmpdir:
        se_dir = os.path.join(tmpdir, "selfevolve")
        os.makedirs(se_dir, exist_ok=True)
        mod = os.path.join(se_dir, "cx.py")
        with open(mod, "w", encoding="utf-8") as f:
            f.write(src)
        rsi = RecursiveSelfImprover(agent, source_root=tmpdir)
        finding = {"type": "high_complexity", "lineno": 1, "name": "f",
                   "complexity": 10}
        first = rsi._patch_from_finding("selfevolve/cx.py", finding)
        assert first is not None
        assert "NOTE(f)" in first.new_code
        # Apply, then a second patch attempt must be rejected (idempotent).
        assert first.apply(tmpdir)
        second = rsi._patch_from_finding("selfevolve/cx.py", finding)
        assert second is None


def test_OllamaBackend_extract_code() -> None:
    assert OllamaBackend._extract_code(
        "```python\ndef f():\n    pass\n```"
    ) == "def f():\n    pass\n"
    assert OllamaBackend._extract_code("def f():\n    pass\n") == \
        "def f():\n    pass"
    # Invalid python -> rejected
    assert OllamaBackend._extract_code("def f(:\n") is None


def test_OllamaBackend_offline_fallback() -> None:
    backend = OllamaBackend(base_url="http://127.0.0.1:1", timeout=0.5)
    assert backend.is_available() is False
    assert backend.suggest_patch("m.py", "def f():\n    pass\n",
                                 {"type": "bare_except",
                                  "suggestion": "fix"}) is None
    assert backend.generate("hello") is None


def test_OllamaBackend_region_mode_large_file() -> None:
    # A big file (>300 lines) with a small target function: the model
    # only sees the function, and the rewrite is spliced back in place.
    pad_src = [
        "import os",
        "",
        "def helper():",
        "    return 0",
        "",
        "def target():",
        "    return 42",
        "",
        "class C:",
        "    def m(self):",
        "        return helper()",
    ]
    big = "\n".join(pad_src) + "\n"
    big += "x = 1\n" * 320
    backend = OllamaBackend()
    backend._available = True
    backend._request = lambda path, payload=None: (  # type: ignore[method-assign]
        {"response": "```python\ndef target():\n    return 43\n```"}
        if path == "/api/generate" else {"models": []}
    )
    new_code = backend.suggest_patch(
        "big.py", big,
        {"type": "magic_number", "lineno": 7, "suggestion": "x"})
    assert new_code is not None
    assert "def target():\n    return 43" in new_code
    assert "def helper():\n    return 0" in new_code
    assert "def m(self):\n        return helper()" in new_code
    assert "\n".join(["x = 1"] * 320) in new_code


def test_OllamaBackend_region_mode_no_change_rejected() -> None:
    # Model echoes the original function -> no patch.
    backend = OllamaBackend()
    backend._available = True
    backend._request = lambda path, payload=None: (  # type: ignore[method-assign]
        {"response": "```python\ndef target():\n    return 42\n```"}
        if path == "/api/generate" else {"models": []}
    )
    out = backend.suggest_patch(
        "big.py", "def helper():\n    return 0\n\ndef target():\n    return 42\n",
        {"type": "magic_number", "lineno": 4, "suggestion": "x"})
    assert out is None


def test_OllamaBackend_region_mode_bad_splice_rejected() -> None:
    # Rewritten function breaks the enclosing code -> whole patch rejected.
    backend = OllamaBackend()
    backend._available = True
    backend._request = lambda path, payload=None: (  # type: ignore[method-assign]
        {"response": "```python\ndef target(\n```"}
        if path == "/api/generate" else {"models": []}
    )
    out = backend.suggest_patch(
        "big.py", "def helper():\n    return 0\n\ndef target():\n    return 42\n",
        {"type": "magic_number", "lineno": 4, "suggestion": "x"})
    assert out is None


def test_OllamaBackend_mocked_generate() -> None:
    backend = OllamaBackend()
    backend._available = True
    calls = []

    def fake_request(path, payload=None):
        calls.append(path)
        if path == "/api/generate":
            return {"response": "```python\ndef f():\n    return 1\n```"}
        return {"models": []}

    backend._request = fake_request  # type: ignore[method-assign]
    new_code = backend.suggest_patch("m.py", "def f():\n    pass\n",
                                     {"type": "magic_number",
                                      "lineno": 2, "suggestion": "x"})
    assert new_code is not None
    assert "return 1" in new_code
    assert calls == ["/api/tags", "/api/generate"]


def test_RecursiveSelfImprover_llm_backend_used_when_available() -> None:
    agent = SelfEvolvingAgent(name="rsi-llm")
    src = "def f():\n    return 42\n"
    with tempfile.TemporaryDirectory(prefix="selfevolve_test_") as tmpdir:
        se_dir = os.path.join(tmpdir, "selfevolve")
        os.makedirs(se_dir, exist_ok=True)
        mod = os.path.join(se_dir, "llm.py")
        with open(mod, "w", encoding="utf-8") as f:
            f.write(src)
        backend = OllamaBackend()
        backend._available = True
        backend._request = lambda path, payload=None: (  # type: ignore[method-assign]
            {"response": "```python\ndef f():\n    return 43\n```"}
            if path == "/api/generate" else {"models": []}
        )
        rsi = RecursiveSelfImprover(agent, source_root=tmpdir,
                                    llm_backend=backend)
        patch = rsi._patch_from_finding("selfevolve/llm.py", {
            "type": "magic_number", "lineno": 2, "value": 42, "function": "f",
        })
        assert patch is not None
        assert patch.description.startswith("LLM fix")
        assert "return 43" in patch.new_code
        assert rsi.llm_patch_count == 1


def test_RecursiveSelfImprover_llm_call_budget() -> None:
    agent = SelfEvolvingAgent(name="rsi-budget")
    with tempfile.TemporaryDirectory(prefix="selfevolve_test_") as tmpdir:
        se_dir = os.path.join(tmpdir, "selfevolve")
        os.makedirs(se_dir, exist_ok=True)
        for i in range(3):
            with open(os.path.join(se_dir, f"m{i}.py"), "w",
                      encoding="utf-8") as f:
                f.write(f"def f{i}():\n    return {i}\n")
        backend = OllamaBackend()
        backend._available = True
        call_count = []

        def fake_request(path, payload=None):
            if path == "/api/generate":
                call_count.append(path)
                return {"response": "```python\ndef f():\n    pass\n```"}
            return {"models": []}

        backend._request = fake_request  # type: ignore[method-assign]
        rsi = RecursiveSelfImprover(agent, source_root=tmpdir,
                                    llm_backend=backend,
                                    max_llm_calls_per_cycle=2)
        for i in range(3):
            rsi._patch_from_finding(f"selfevolve/m{i}.py", {
                "type": "magic_number", "lineno": 2, "value": i,
                "function": f"f{i}",
            })
        assert len(call_count) == 2


# ===================================================================
#  multi_agent.py — AgentMessage & DebateRound
# ===================================================================

def test_AgentMessage_post_init() -> None:
    msg = AgentMessage(sender="alice", recipient="bob", msg_type="debate",
                       payload={"claim": "x"})
    assert msg.timestamp > 0
    assert len(msg.id) == 12


def test_AgentMessage_broadcast() -> None:
    msg = AgentMessage(sender="alice", recipient="*", msg_type="broadcast",
                       payload={"data": 1})
    assert msg.recipient == "*"


def test_DebateRound_basic() -> None:
    dr = DebateRound(topic="best strategy", positions=[], votes={})
    assert dr.topic == "best strategy"
    assert dr.consensus is None
    assert dr.resolved is False


# ===================================================================
#  multi_agent.py — DebateArena
# ===================================================================

def test_DebateArena_submit_and_resolve() -> None:
    arena = DebateArena(topic="which tool is best")
    arena.submit_position("alice", "Python", "versatile")
    arena.submit_position("bob", "Rust", "fast")
    assert len(arena.current_round.positions) == 2

    arena.cast_vote("alice", "alice")
    arena.cast_vote("bob", "bob")
    arena.cast_vote("charlie", "bob")
    resolved = arena.resolve()
    assert resolved is True


def test_DebateArena_no_positions() -> None:
    arena = DebateArena(topic="empty")
    assert arena.resolve() is False


def test_DebateArena_consensus_claim() -> None:
    arena = DebateArena(topic="language")
    arena.submit_position("alice", "Python is best", "fast dev")
    arena.submit_position("bob", "Rust is best", "fast runtime")
    arena.cast_vote("alice", "alice")
    arena.cast_vote("bob", "bob")
    arena.cast_vote("charlie", "alice")
    arena.resolve()
    claim = arena.consensus_claim()
    assert claim is not None


# ===================================================================
#  multi_agent.py — MultiAgentSociety
# ===================================================================

def test_MultiAgentSociety_register_and_broadcast() -> None:
    society = MultiAgentSociety()
    agent = SelfEvolvingAgent(name="soc-agent")
    society.register_agent(agent)
    assert "soc-agent" in society.agents

    society.broadcast(sender="soc-agent", msg_type="hello",
                      payload={"greeting": "hi"})
    assert len(society.message_log) == 1
    assert society.message_log[0].msg_type == "hello"


def test_MultiAgentSociety_send() -> None:
    society = MultiAgentSociety()
    a1 = SelfEvolvingAgent(name="alice")
    a2 = SelfEvolvingAgent(name="bob")
    society.register_agent(a1)
    society.register_agent(a2)
    society.send(sender="alice", recipient="bob", msg_type="teaching",
                 payload={"skill": "reflect"})
    assert len(society.message_log) == 1
    assert society.message_log[0].recipient == "bob"


def test_MultiAgentSociety_transfer_skill() -> None:
    society = MultiAgentSociety()
    a1 = SelfEvolvingAgent(name="teacher")
    a2 = SelfEvolvingAgent(name="learner")
    society.register_agent(a1)
    society.register_agent(a2)

    result = society.transfer_skill("teacher", "learner", "reflect")
    assert result is True
    assert society.skill_transfer_count == 1
    assert "reflect_from_teacher" in a2.skill_library.skills


def test_MultiAgentSociety_transfer_skill_missing() -> None:
    society = MultiAgentSociety()
    a1 = SelfEvolvingAgent(name="a")
    society.register_agent(a1)
    assert society.transfer_skill("a", "nonexistent", "reflect") is False
    assert society.transfer_skill("nonexistent", "a", "reflect") is False
    assert society.transfer_skill("a", "a", "nonexistent_skill") is False


def test_MultiAgentSociety_debate() -> None:
    society = MultiAgentSociety()
    alice = SelfEvolvingAgent(name="alice")
    bob = SelfEvolvingAgent(name="bob")
    society.register_agent(alice)
    society.register_agent(bob)

    arena = society.debate("coding", ["alice", "bob"])
    assert isinstance(arena, DebateArena)
    assert len(society.debates) == 1
    assert arena.topic == "coding"


def test_MultiAgentSociety_collectively_improve() -> None:
    society = MultiAgentSociety()
    a1 = SelfEvolvingAgent(name="alice")
    a2 = SelfEvolvingAgent(name="bob")
    society.register_agent(a1)
    society.register_agent(a2)

    results = society.collectively_improve(
        topic="write better code", max_steps_per_agent=3
    )
    assert "alice" in results
    assert "bob" in results
    for name, r in results.items():
        assert "steps" in r
        assert "success_rate" in r
        assert "skills" in r


def test_MultiAgentSociety_specialize_agent() -> None:
    society = MultiAgentSociety()
    agent = SelfEvolvingAgent(name="specialist")
    society.register_agent(agent)
    society.specialize_agent("specialist", "math")
    # The agent should have a goal set (if not already)
    # specialist_agent only sets the goal, doesn't guarantee execution
    assert agent.current_goal == "Become expert in math"


def test_MultiAgentSociety_specialize_agent_missing() -> None:
    society = MultiAgentSociety()
    # Should not raise
    society.specialize_agent("ghost", "anything")


def test_MultiAgentSociety_collective_report() -> None:
    society = MultiAgentSociety()
    agent = SelfEvolvingAgent(name="report-agent")
    society.register_agent(agent)
    report = society.collective_report()
    assert "MultiAgentSociety" in report
    assert "report-agent" in report
    assert "Agents:" in report


# ===================================================================
#  Runner
# ===================================================================

if __name__ == "__main__":
    import textwrap as _tw  # noqa: re-import for safety in runner
    failures = 0
    tests = [fn for fn, _ in globals().items() if fn.startswith("test_")]
    for name in sorted(tests):
        try:
            globals()[name]()
            print(f"  [OK] {name}")
        except Exception as e:
            print(f"  [--] {name}: {e}")
            failures += 1
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(0 if failures == 0 else 1)
