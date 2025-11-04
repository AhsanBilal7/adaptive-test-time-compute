#!/usr/bin/env python3
import os, json, math, sys, argparse
from collections import defaultdict, OrderedDict
try:
    import pandas as pd
except ImportError:
    pd = None

# ---------------- Crafter normalized score ----------------
def normalized_crafter_score(success_rates, num_achievements=22):
    s = list(success_rates)
    if len(s) < num_achievements:
        s += [0.0] * (num_achievements - len(s))
    return (math.exp(sum(math.log(1.0 + x) for x in s) / num_achievements) - 1.0) * 100.0


def is_episode_json(path):
    name = os.path.basename(path)
    return name.endswith(".json") and not name.endswith("_summary.json") and name not in ("summary.json","crafter_summary.json")


def collect_episode_achievements(run_task_dir):
    episodes = 0
    success_counts = defaultdict(int)
    all_keys = set()
    for root, _, files in os.walk(run_task_dir):
        for f in files:
            if not is_episode_json(f):
                continue
            p = os.path.join(root, f)
            try:
                with open(p) as fh:
                    data = json.load(fh)
            except Exception:
                continue
            ach = data.get("achievements") or {}
            fired = {k for k, v in ach.items() if (isinstance(v, bool) and v) or (isinstance(v,(int,float)) and v>0)}
            for k in fired:
                success_counts[k] += 1
                all_keys.add(k)
            all_keys |= set(ach.keys())
            episodes += 1
    return episodes, success_counts, all_keys


def read_summary_fields(summary_path):
    try:
        with open(summary_path) as f:
            d = json.load(f)
        prog = float(d.get("progression_percentage", 0.0))
        se = float(d.get("standard_error", 0.0))
        episodes = int(d.get("episodes_played", 0))
    except Exception:
        prog, se, episodes = 0.0, 0.0, 0
    return prog, se, episodes


def read_agent_config(run_dir):
    """Return mode and planning_frequency from outer summary.json."""
    outer_summary = os.path.join(run_dir, "summary.json")
    if not os.path.exists(outer_summary):
        return None, None
    try:
        with open(outer_summary) as f:
            d = json.load(f)
        agent = d.get("agent", {})
        mode = agent.get("mode")
        plan_freq = agent.get("planning_frequency")
    except Exception:
        mode, plan_freq = None, None
    return mode, plan_freq


def find_crafter_runs(results_root):
    runs = []
    for dirpath, _, files in os.walk(results_root):
        if "crafter_summary.json" in files:
            summary_path = os.path.join(dirpath, "crafter_summary.json")
            rel = os.path.relpath(dirpath, results_root)
            parts = rel.split(os.sep)
            run_dir = os.path.join(results_root, parts[0])
            env = parts[1] if len(parts)>=2 else ""
            task = parts[2] if len(parts)>=3 else os.path.basename(dirpath)
            runs.append({
                "summary_path": summary_path,
                "run_dir": run_dir,
                "env": env,
                "task": task,
                "task_dir": dirpath
            })
    return runs


def assemble_table(results_root, out_csv):
    runs = find_crafter_runs(results_root)
    if not runs:
        print(f"No crafter_summary.json files found under {results_root}")
        return
    global_ach = set()
    per_run = {}
    for r in runs:
        episodes, counts, keys = collect_episode_achievements(r["task_dir"])
        per_run[r["task_dir"]] = (episodes, counts, keys)
        global_ach |= keys
    ach_cols = sorted(global_ach)

    rows = []
    for r in runs:
        prog, se, ep_sum = read_summary_fields(r["summary_path"])
        episodes, counts, keys = per_run[r["task_dir"]]
        num_eps = episodes if episodes>0 else ep_sum
        rates = {k:(counts.get(k,0)/num_eps if num_eps>0 else 0.0) for k in ach_cols}
        score_pct = normalized_crafter_score(rates.values())
        mode, plan_freq = read_agent_config(r["run_dir"])
        row = OrderedDict()
        row["run_folder"] = os.path.basename(r["run_dir"])
        row["env"] = r["env"]
        row["task"] = r["task"]
        row["episodes"] = num_eps
        row["mode"] = mode
        row["planning_frequency"] = plan_freq
        row["progression_percentage"] = prog
        row["standard_error"] = se
        row["normalized_score_pct"] = score_pct
        for k in ach_cols:
            row[f"ach_{k}_rate"] = rates[k]
        rows.append(row)

    if pd is not None:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    else:
        import csv
        with open(out_csv,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"✅ CSV written to {out_csv}")
    for r in rows:
        print(f"{r['run_folder']}: Prog={r['progression_percentage']:.2f}% ±{r['standard_error']:.2f}, NormScore={r['normalized_score_pct']:.2f}%, Mode={r['mode']}, PlanFreq={r['planning_frequency']}")


def main():
    ap = argparse.ArgumentParser(description="Summarize BALROG Crafter runs into a CSV with progression, stderr, normalized score, and per-achievement rates.")
    ap.add_argument("--results_root", type=str, required=False, default="./BALROG/results/" ,help="Path to BALROG results/ directory")
    ap.add_argument("--out_csv", type=str, default="crafter_runs_summary.csv", help="Output CSV path (default: ./crafter_runs_summary.csv)")
    args = ap.parse_args()

    sys.exit(assemble_table(args.results_root, args.out_csv))


if __name__ == "__main__":
    main()
