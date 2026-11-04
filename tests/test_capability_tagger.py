"""Golden cases for the capability tagger, taken from real ALFWorld / ScienceWorld / ByteSized32 transitions.
Run:  python -m pytest tests/test_capability_tagger.py   (or: python tests/test_capability_tagger.py)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from capability.tagger import tag_transition  # noqa: E402

ALF_OT = ("[GOAL] put a mug in coffeemachine.\n[ROOM] You are in the middle of a room. Looking quickly around you, "
          "you see a cabinet 1, a coffeemachine 1, a countertop 1, a drawer 1, and a shelf 1.\n"
          "[LOCATION] middle of the room\n[FEEDBACK] You are in the middle of a room.\n"
          "[INVENTORY] You are not carrying anything.\n[ADMISSIBLE] go to cabinet 1 | go to shelf 1 | look")

CASES = [
    # (o_t, action, o_next / feedback, expected tags (subset), expected primary)
    (ALF_OT, "go to shelf 1", "You arrive at shelf 1. On the shelf 1, you see a mug 2.",
     {"containment_location", "partial_obs"}, "containment_location"),
    (ALF_OT, "go to shelf 1", "You arrive at shelf 1. On the shelf 1, you see nothing.",
     {"containment_location", "partial_obs"}, "containment_location"),
    (ALF_OT, "open drawer 1", "You open the drawer 1. The drawer 1 is open. In it, you see a soapbar 1.",
     {"object_state", "partial_obs"}, "object_state"),
    (ALF_OT, "close drawer 1", "You close the drawer 1.", {"object_state"}, "object_state"),
    (ALF_OT, "slice apple 1 with knife 1", "Nothing happens.", {"action_failure"}, "action_failure"),
    (ALF_OT, "inventory", "You are not carrying anything.", set(), "none"),
    (ALF_OT, "take mug 2 from shelf 1", "You pick up the mug 2 from the shelf 1.",
     {"containment_location"}, "containment_location"),
    ("[ROOM] This room is called the kitchen.\n[LOCATION] kitchen\n[FEEDBACK] x\n[INVENTORY] a thermometer",
     "use thermometer on gallium", "the thermometer measures a temperature of 31 degrees celsius",
     {"numeric_physical"}, "numeric_physical"),
    ("[ROOM] kitchen\n[LOCATION] kitchen\n[FEEDBACK] x", "open door to hallway", "The door is already open.",
     {"action_failure"}, "action_failure"),
    ("[ROOM] kitchen\n[LOCATION] kitchen\n[FEEDBACK] x", "mix bowl",
     "That container is empty, so there are no items to mix.", {"action_failure"}, "action_failure"),
    ("[ROOM] a room\n[FEEDBACK] x", "take yourself", "You cannot take yourself.", {"action_failure"}, "action_failure"),
    ("[ROOM] a room\n[FEEDBACK] x", "rotate iso anticlockwise", "You rotated the iso dial to 3200",
     {"numeric_physical"}, "numeric_physical"),
    ("[ROOM] a room\n[LOCATION] kitchen\n[FEEDBACK] x", "look at stopwatch",
     "a stopwatch, which is deactivated. The time reads 0 ticks.", {"partial_obs"}, "partial_obs"),
]


def test_cases():
    for o_t, a, fb, want, prim in CASES:
        got = tag_transition(o_t, a, fb)
        assert want <= set(got["tags"]), (a, fb, got)
        assert got["primary"] == prim, (a, fb, got)


if __name__ == "__main__":
    test_cases()
    print(f"all {len(CASES)} cases pass")
