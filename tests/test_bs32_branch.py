"""ByteSized32 branched states on a tiny hand-made crawl: python -m pytest tests/test_bs32_branch.py -q"""
import json

from wm import bs32_branch as bb


def _state(step, last=""):
    return {"taskDesc": "Your task is to lit the bulb.", "look": "room at step %d" % step,
            "observation": "obs %d" % step, "inventory": "empty", "lastAction": last,
            "objects": [{"uuid": 1, "step": step}], "max_UUID": 1}


ACTIONS = ["connect a to b", "connect b to c", "open box", "take wire", "move wire to box",
           "look", "inventory", "examine bulb"]


def _crawl_records():
    """Gold path: s0 -connect a to b-> s1 -connect b to c-> s2 (won). Other actions stay in place."""
    recs = []
    gold = {0: "connect a to b", 1: "connect b to c"}
    cur = _state(0)
    for step in (0, 1):
        gold_tick = None
        for a in ACTIONS:
            nxt = step + 1 if a == gold[step] else step
            if nxt == step:                       # same state, only the feedback differs
                tick = dict(cur, observation="after %s at %d" % (a, step), lastAction=a)
            else:                                 # the next crawled state is exactly this result
                tick = dict(_state(nxt, a), observation="after %s at %d" % (a, step))
                gold_tick = tick
            recs.append({"game": "lit-bulb", "state_id": len(recs), "current_state": cur,
                         "action_state": {"lastAction": a}, "tick_state": tick,
                         "current_score_state": {"score": step}, "next_score_state":
                         {"score": nxt, "gameWon": nxt == 2, "gameOver": nxt == 2}})
        cur = gold_tick
    return recs


def test_gold_chain_candidates_and_training_exclusion(tmp_path):
    recs = _crawl_records()
    crawl_path, train_path = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    crawl_path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    trained = [r for r in recs if r["action_state"]["lastAction"] in ("open box", "connect a to b")]
    train_path.write_text("".join(json.dumps(r) + "\n" for r in trained))
    play = tmp_path / "play"
    play.mkdir()
    (play / "lit-bulb-playthrough.txt").write_text("Task\n> connect a to b\nObs\n> connect b to c\nObs\n")
    out = tmp_path / "bs.jsonl"
    assert bb.main(["--crawl", str(crawl_path), "--train", str(train_path), "--playthroughs", str(play),
                    "--out", str(out), "--k", "4"]) == 0
    states = [json.loads(l) for l in open(out)]
    assert len(states) == 2                                   # both states on the gold chain
    by_gold = {s["gold_action"]: s for s in states}
    s0 = by_gold["connect a to b"]
    assert s0["actions"][s0["gold_index"]] == "connect a to b" and s0["gold_seen_in_training"]
    assert "open box" not in s0["actions"]                    # trained pair, not a distractor
    for s in states:
        assert not {"look", "inventory"} & set(s["actions"]) and len(set(s["actions"])) == 4
        assert s["obs"].startswith("[GOAL] Your task is to lit the bulb.\n[ROOM] room at step")
    s1 = by_gold["connect b to c"]
    assert s1["true_won"][s1["gold_index"]] and s1["true_score"][s1["gold_index"]] == 2
    assert s1["true_reward"][s1["gold_index"]] == 1 and not s1["gold_seen_in_training"]
