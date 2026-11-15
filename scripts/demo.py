#!/usr/bin/env python3
"""
Demo: one decision, three world models.

Replays a single held-out state from our evaluation and shows, step by step, how each world model
(G retrieval, F fine-tuned, D JEPA top-1) leads the agent to a decision, and what that decision
costs in regret. Everything is read from the saved evaluation outputs, so it runs on a CPU in under
a second with no models loaded:

    python scripts/demo.py                         # the paper's worked example (candle, ALFWorld, n=256)
    python scripts/demo.py --slow                  # same, paced for screen recording
    python scripts/demo.py --find d_fails          # list states where G and F are right and D is wrong
    python scripts/demo.py --state <id> --budget 64
    python scripts/demo.py --regime bytesized32 --random

What the numbers are (see Section III of the paper):
  candidate actions   the expert's action + 4 distractors from the valid actions
  true outcome        the next state from replaying the real environment
  true value          judge (Qwen2.5-7B-Instruct-AWQ) rating 0-10 of the true outcome
  predicted value     the same judge's rating of the outcome the world model PREDICTED
  decision            the action with the highest predicted value
  regret              best true value - true value of the chosen action (0 = best possible choice)
"""
import argparse
import json
import os
import random
import sys
import textwrap
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REGIMES = {"alfworld": "ALFWorld (new games)", "sciworld": "ScienceWorld (new variations)",
           "sciworld_seen": "ScienceWorld (held-out variations, seen types)",
           "bytesized32": "ByteSized32 (known states, new actions)"}
ARMS = [("G", "top1", "G: retrieval (frozen 7B LLM + 5 retrieved examples)", "34"),
        ("F", "top1", "F: fine-tuned (1.5B LLM + LoRA)", "33"),
        ("D", "top1", "D: JEPA (29M predictor in embedding space)", "32")]
BUDGETS = [32, 64, 128, 256, 512, 1024]
EXAMPLE = os.path.join(ROOT, "demo", "candle_example.json")

COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
PACE = 0.0


def c(text, code):
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def bold(t): return c(t, "1")
def dim(t): return c(t, "2")


def say(line="", pause=0.0):
    print(line, flush=True)
    if PACE:
        time.sleep(PACE * (1 + pause))


def header(title):
    say()
    say(bold(c(f"── {title} " + "─" * max(0, 76 - len(title)), "36")), 0.5)


def wrap(text, width=70, indent="      "):
    return textwrap.fill(" ".join(str(text).split()), width=width, initial_indent=indent,
                         subsequent_indent=indent)


def sections(obs):
    out, key = {}, None
    for line in obs.split("\n"):
        if line.startswith("[") and "]" in line:
            key, line = line[1:line.index("]")], line[line.index("]") + 1:]
        if key:
            out[key] = (out.get(key, "") + " " + line).strip()
    return out


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def regime_of(state_id):
    env = state_id.split("/")[0]
    return env


def load_state(regime, state_id):
    for s in load_jsonl(os.path.join(ROOT, "data", "eval", f"{regime}_branched_states.jsonl")):
        if s["id"] == state_id:
            return s
    sys.exit(f"state {state_id} not found in {regime}")


def load_scores(regime, budget, arms=ARMS):
    out = {}
    for arm, mode, _, _ in arms:
        path = os.path.join(ROOT, "results", "utility", "scored", f"{regime}_{arm}_{mode}_b{budget}.jsonl")
        for r in load_jsonl(path):
            if regime == "bytesized32" and r.get("gold_seen") is True:
                continue
            out.setdefault(r["id"], {})[arm] = r
    return {k: v for k, v in out.items() if len(v) == len(arms)}


def find(regime, budget, kind):
    scores = load_scores(regime, budget)
    hits = []
    for sid, r in scores.items():
        g, f, d = r["G"]["regret"], r["F"]["regret"], r["D"]["regret"]
        ok = {"d_fails": g == 0 and f == 0 and d >= 3,
              "g_beats_f": g == 0 and f >= 3,
              "f_beats_g": f == 0 and g >= 3,
              "all_right": g == f == d == 0,
              "all_wrong": min(g, f, d) >= 3}[kind]
        if ok:
            hits.append((sid, g, f, d))
    return hits


def show(state, regime, budget, predictions=None):
    scores = load_scores(regime, budget)[state["id"]]
    acts = state["actions"]
    true_next = state["true_next"]
    true_v = scores["G"]["true_values"]
    gold = int(state["gold_index"])
    best = max(true_v)

    header("1. The situation")
    say(f"  Environment : {REGIMES[regime]}")
    say(f"  Goal        : {bold(state['goal'])}")
    sec = sections(state["obs"])
    for key, label in (("LOCATION", "Location"), ("FEEDBACK", "Last seen"), ("INVENTORY", "Carrying")):
        if sec.get(key):
            say(textwrap.fill(" ".join(sec[key].split())[:300], width=86, initial_indent=f"  {label:<12}: ",
                              subsequent_indent=" " * 16))
    note = "known state, the expert's move never seen in training" if regime == "bytesized32" else \
        "held out: this game/variation was never seen in training"
    say(dim(f"  State id    : {state['id']}  ({note})"), 1)

    header("2. The agent can do one of five things")
    for i, a in enumerate(acts):
        tag = dim("  (expert's move)") if i == gold else ""
        say(f"   [{i + 1}] {a}{tag}", 0.3)
    say(dim("  To choose, it asks its world model: \"what happens if I do this?\""), 1)

    for arm, _, name, col in ARMS:
        r = scores[arm]
        header(f"3{arm}. World model {name}")
        say(dim(f"      trained on n = {budget} transitions of this environment"))
        for i, a in enumerate(acts):
            v = r["pred_values"][i]
            mark = c(" ◀ chosen", col) if i == r["chosen"] else ""
            say(f"   [{i + 1}] {a}")
            if predictions and arm in predictions:
                say(c(wrap("predicts: " + predictions[arm][i]), col))
            say(f"      judge's rating of the predicted outcome: {bold(str(v))}/10{mark}", 0.2)
        say()
        ch = r["chosen"]
        verdict = c("best possible move, regret 0", "32;1") if r["regret"] == 0 else \
            c(f"regret {r['regret']}  (true value {true_v[ch]} vs best {best})", "31;1")
        say(f"   Decision: {bold(acts[ch])}  →  {verdict}", 1)

    header("4. What actually happens (replayed in the real environment)")
    for i, a in enumerate(acts):
        say(f"   [{i + 1}] {a}")
        say(wrap(true_next[i]))
        say(f"      true value: {bold(str(true_v[i]))}/10", 0.2)

    header("5. Scoreboard for this decision")
    for arm, _, name, col in ARMS:
        r = scores[arm]
        bar = "█" * int(r["regret"] * 4) or "·"
        say(f"   {c(arm, col + ';1')}  regret {r['regret']:>2}  {c(bar, col)}", 0.3)
    say()
    say("   Regret of the same decision as the training budget grows:")
    row = "        n = " + "".join(f"{b:>6}" for b in BUDGETS)
    say(dim(row))
    for arm, _, _, col in ARMS:
        vals = []
        for b in BUDGETS:
            s = load_scores(regime, b).get(state["id"])
            vals.append(f"{s[arm]['regret']:>6}" if s else f"{'-':>6}")
        say(f"        {c(arm, col + ';1')}   " + "".join(vals), 0.2)

    header(f"6. Across all held-out states of this environment (n = {budget})")
    allsc = load_scores(regime, budget)
    rnd = sum((max(r["G"]["true_values"]) - sum(r["G"]["true_values"]) / len(r["G"]["true_values"]))
              for r in allsc.values()) / len(allsc)
    say(f"   {len(allsc)} decisions, mean regret (lower is better):")
    rows = [(arm, col, sum(r[arm]["regret"] for r in allsc.values()) / len(allsc)) for arm, _, _, col in ARMS]
    rows.append(("random", "37", rnd))
    for arm, col, m in rows:
        say(f"   {c(f'{arm:<6}', col + ';1')} {m:5.2f}  {c('█' * int(round(m * 20)), col)}", 0.3)
    say()
    say(dim("   Every regime and budget: Fig. 1 and Table II of the paper."), 1)


def main():
    global PACE
    ap = argparse.ArgumentParser(description="One decision, three world models.")
    ap.add_argument("--state", help="state id (default: the paper's worked example)")
    ap.add_argument("--regime", choices=list(REGIMES), help="regime (inferred from --state if omitted)")
    ap.add_argument("--budget", type=int, default=256, choices=BUDGETS)
    ap.add_argument("--find", choices=["d_fails", "g_beats_f", "f_beats_g", "all_right", "all_wrong"],
                    help="list interesting states instead of showing one")
    ap.add_argument("--random", action="store_true", help="show a random state of --regime")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slow", action="store_true", help="pace the output for screen recording")
    args = ap.parse_args()
    PACE = 0.35 if args.slow else 0.0

    example = json.load(open(EXAMPLE)) if os.path.exists(EXAMPLE) else None
    if args.find:
        regime = args.regime or "alfworld"
        hits = find(regime, args.budget, args.find)
        print(f"{len(hits)} states in {regime} at n={args.budget} ({args.find}); regret G F D:")
        for sid, g, f, d in hits[:40]:
            print(f"  {g:>2} {f:>2} {d:>2}  {sid}")
        return

    if args.random:
        regime = args.regime or "alfworld"
        state_id = random.Random(args.seed).choice(sorted(load_scores(regime, args.budget)))
    else:
        state_id = args.state or example["id"]
        regime = args.regime or regime_of(state_id)
        if regime == "sciworld" and "seen" in state_id:
            regime = "sciworld_seen"
    preds = example["predictions"] if example and state_id == example["id"] and args.budget == example["budget"] else None
    show(load_state(regime, state_id), regime, args.budget, preds)


if __name__ == "__main__":
    main()
