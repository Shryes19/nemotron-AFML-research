"""Summary and integrity checks for collected transition files.

  python -m wm.stats data/raw/alfworld_train.jsonl.gz data/raw/alfworld_heldout.jsonl.gz

Prints rows per split/policy, episodes, tasks, top action verbs, won rate, and fails (exit 1)
if any task_id appears in more than one split or a transition_id is duplicated.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict

from wm.schema import read_jsonl, validate_row


def main(paths):
    tasks = defaultdict(set)
    seen, problems = set(), []
    per = Counter()
    verbs = defaultdict(Counter)
    episodes = defaultdict(set)
    won = Counter()
    for path in paths:
        for row in read_jsonl(path):
            key = (row["env"], row["split"])
            per[key + (row["policy"],)] += 1
            tasks[key].add(row["task_id"])
            episodes[key].add(row["episode_id"])
            verbs[key][row["a_t"].split()[0] if row["a_t"] else ""] += 1
            won[key] += row["won"]
            if row["transition_id"] in seen:
                problems.append("duplicate transition_id %s" % row["transition_id"])
            seen.add(row["transition_id"])
            errs = validate_row(row)
            if errs:
                problems.append("%s: %s" % (row["transition_id"], "; ".join(errs)))
    for key in sorted(tasks):
        total = sum(v for k, v in per.items() if k[:2] == key)
        print("== %s/%s: %d rows, %d episodes, %d tasks, %d won transitions"
              % (key[0], key[1], total, len(episodes[key]), len(tasks[key]), won[key]))
        print("   policy:", {k[2]: v for k, v in sorted(per.items()) if k[:2] == key})
        print("   verbs:", dict(verbs[key].most_common(8)))
    for env in {k[0] for k in tasks}:
        overlap = tasks.get((env, "train"), set()) & tasks.get((env, "heldout"), set())
        if overlap:
            problems.append("%s: %d task_ids in both train and heldout" % (env, len(overlap)))
    for p in problems[:20]:
        print("PROBLEM:", p)
    print("OK" if not problems else "%d problems" % len(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
