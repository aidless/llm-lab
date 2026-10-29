import json, argparse, pathlib
from typing import List, Dict

def extract_dpo_pairs(agent_safety_root: str) -> List[Dict]:
    """Extract (chosen, rejected) trajectory pairs from policy-on/off runs."""
    root = pathlib.Path(agent_safety_root)
    # The task set and the runs do not always sit at the same depth: in
    # agent-safety-eval the tasks live at dataset/ while the runs live under
    # evidence/runs/. Accept either layout instead of silently returning [].
    runs_dir = next((p for p in (root / "runs", root / "evidence" / "runs") if p.is_dir()), None)
    if runs_dir is None:
        raise SystemExit(f"no runs/ directory under {root} (looked for runs/ and evidence/runs/)")
    tasks_file = next(
        (p for p in (root / "dataset" / "tasks-v1.jsonl",
                     root / "evidence" / "dataset" / "tasks-v1.jsonl") if p.is_file()),
        None,
    )
    if tasks_file is None:
        raise SystemExit(f"no dataset/tasks-v1.jsonl under {root}")
    runs = sorted(runs_dir.glob("*policy-*"))
    triples = []
    with open(tasks_file) as f:
        tasks = [json.loads(l) for l in f if l.strip()]
    for task in tasks:
        task_id = task["id"]
        # Find matching runs
        chosen_trajectories = []
        rejected_trajectories = []
        for run in runs:
            traj_file = run/"trajectories.jsonl"
            if not traj_file.exists(): continue
            is_policy_on = "policy-on" in str(run)
            with open(traj_file) as f:
                trajs = [json.loads(l) for l in f if l.strip()]
            for t in trajs:
                if t.get("task_id") != task_id: continue
                if is_policy_on and not t.get("unsafe_action"):
                    chosen_trajectories.append(t)
                elif not is_policy_on and t.get("unsafe_action"):
                    rejected_trajectories.append(t)
        if chosen_trajectories and rejected_trajectories:
            triples.append({
                "prompt": task["prompt"],
                "chosen": chosen_trajectories[0]["full_trajectory"],
                "rejected": rejected_trajectories[0]["full_trajectory"],
                "task_id": task_id,
            })
    return triples

def main(args):
    triples = extract_dpo_pairs(args.agent_safety_root)
    if not triples:
        # An empty file is indistinguishable from "the run set is fine but there
        # was nothing to pair", so refuse to write one. The current
        # agent-safety-eval archives name their arms *-policy-off / *-policy-prompt
        # and record `unsafe_executed`, while this extractor looks for a
        # `*-policy-on` arm and `unsafe_action`: against that layout it can only
        # ever return zero, and it should say so instead of emitting a stub.
        raise SystemExit(
            "extracted 0 DPO triples: no task had both a safe policy-on and an "
            "unsafe policy-off trajectory. Check that the run directories are "
            "named *-policy-on / *-policy-off and that trajectories carry "
            "task_id, unsafe_action and full_trajectory. This extractor has not "
            "been adapted to the *-policy-prompt / unsafe_executed schema."
        )
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for t in triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"Extracted {len(triples)} DPO triples to {output}")

def cli(argv=None):
    parser = argparse.ArgumentParser()
    # The README and scripts/rerun.sh call this as `... build --agent-safety-root`,
    # so accept the verb the docs already use rather than forcing a doc rewrite.
    parser.add_argument("command", nargs="?", default="build", choices=["build"])
    parser.add_argument("--agent-safety-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    main(args)


if __name__ == "__main__":
    cli()
