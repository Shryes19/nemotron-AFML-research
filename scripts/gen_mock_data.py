"""
scripts/gen_mock_data.py
─────────────────────────
Generate synthetic (obs, action, obs_next) transitions for testing.

This lets you run the entire JEPA pipeline without waiting for
Saanvi's environment data collection.  The mock data is realistic enough
to verify: encoder loads, embeddings change, loss decreases, NN-decode returns text.

Usage:
    python scripts/gen_mock_data.py --n 64 --out data/buffers/mock_n64.jsonl
    python scripts/gen_mock_data.py --n 32 --out data/buffers/mock_n32.jsonl
"""

import json
import random
import argparse
from pathlib import Path


# ── Mock vocabulary ────────────────────────────────────────────────────────────

ROOMS = [
    "kitchen", "living room", "bedroom", "bathroom", "garden",
    "lab", "storage room", "greenhouse",
]
OBJECTS = [
    "apple", "knife", "book", "flask", "candle", "thermometer",
    "key", "box", "lighter", "hammer", "test tube", "plant",
]
ACTIONS = [
    "pick up the {obj}",
    "put down the {obj}",
    "go to the {room}",
    "examine the {obj}",
    "heat the {obj}",
    "open the {obj}",
    "mix the {obj} with water",
    "place the {obj} in the box",
]
RESULTS = [
    "You pick up the {obj}. It feels heavier than expected.",
    "You put down the {obj} on the table.",
    "You move to the {room}. It is dimly lit.",
    "The {obj} looks worn but functional.",
    "The {obj} heats up. You see steam rising.",
    "The {obj} opens with a click.",
    "The mixture bubbles slightly.",
    "The {obj} is now inside the box.",
]
CAP_TAGS = [
    "object_state", "containment_location", "numeric_physical",
    "partially_observable", "action_failure",
]
ENVIRONMENTS = ["alfworld", "scienceworld", "webshop", "bytesized32"]
GAMES = {
    "alfworld": [f"task_{i:03d}" for i in range(10)],
    "scienceworld": [f"science_{i:03d}" for i in range(8)],
    "webshop": [f"shop_{i:03d}" for i in range(6)],
    "bytesized32": [f"game_{i:03d}" for i in range(31)],
}


def make_transition(env: str, game: str, rng: random.Random) -> dict:
    room   = rng.choice(ROOMS)
    obj    = rng.choice(OBJECTS)
    action = rng.choice(ACTIONS).format(obj=obj, room=room)
    result = rng.choice(RESULTS).format(obj=obj, room=room)
    cap    = rng.choice(CAP_TAGS)

    # Vary obs length to make embeddings distinct
    context_lines = rng.randint(1, 3)
    context = "  ".join(
        f"There is a {rng.choice(OBJECTS)} nearby." for _ in range(context_lines)
    )

    obs = (
        f"You are in the {room}. {context} "
        f"Your inventory: {rng.choice(OBJECTS)}. "
        f"Available actions: {', '.join(rng.sample(ACTIONS[:4], 3)).format(obj=obj, room=room)}."
    )
    obs_next = (
        f"You are in the {room}. {result} "
        f"Your inventory: {obj}. "
        f"Available actions: {', '.join(rng.sample(ACTIONS[:4], 3)).format(obj=obj, room=room)}."
    )

    return {
        "obs":      obs,
        "action":   action,
        "obs_next": obs_next,
        "env":      env,
        "game":     game,
        "cap_tag":  cap,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",   type=int,  default=64,  help="Number of transitions")
    parser.add_argument("--out", type=str,  default="data/buffers/mock_n64.jsonl")
    parser.add_argument("--env", type=str,  default="alfworld",
                        choices=list(GAMES.keys()) + ["mixed"],
                        help="Environment to simulate (or 'mixed' for all 4)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    envs = list(GAMES.keys()) if args.env == "mixed" else [args.env]

    transitions = []
    for i in range(args.n):
        env  = envs[i % len(envs)]
        game = rng.choice(GAMES[env])
        transitions.append(make_transition(env, game, rng))

    with open(args.out, "w") as f:
        for t in transitions:
            f.write(json.dumps(t) + "\n")

    print(f"[MockData] Wrote {args.n} transitions → {args.out}")


if __name__ == "__main__":
    main()
