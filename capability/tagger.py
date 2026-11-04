"""
Rule-based capability tagger for text world-model transitions (o_t, a_t, o_next).

Five capability tags (multi-label; a transition can exercise several):
  object_state          an object's own state changes (open/close, on/off, sliced, cleaned, heated,
                        connected, life-cycle stage, ...)
  containment_location  something moves: the agent changes location, or an object changes container
                        or inventory
  numeric_physical      a quantity or phase changes (temperature reading, volume, weight, melting/boiling)
  partial_obs           the next observation reveals entities that are not inferable from o_t
                        (opening a drawer shows its contents, arriving somewhere shows what is there)
  action_failure        the action is rejected or has no effect ("Nothing happens.", "You can't ...")

`primary` is a single label for the cap_tag field, chosen by the priority
  action_failure > numeric_physical > object_state > containment_location > partial_obs > none
"none" means no capability is exercised (e.g. "inventory", "look" with nothing new): the next state is a
copy of what the agent already knows.

Works on the common buffer format (Saanvi's exporter): o_t / o_next may be either the full structured
observation with [GOAL]/[ROOM]/[LOCATION]/[FEEDBACK]/[INVENTORY]/[ADMISSIBLE] sections, or raw feedback
text. Pure stdlib, so it runs in either venv.
"""
from __future__ import annotations

import re

TAGS = ["object_state", "containment_location", "numeric_physical", "partial_obs", "action_failure"]
PRIMARY_ORDER = ["action_failure", "numeric_physical", "object_state",
                 "containment_location", "partial_obs"]

# ---------------------------------------------------------------- failure lexicon
# Built from the feedback strings of ALFWorld (alfworld==0.3.3), ScienceWorld (1.2.3) and ByteSized32-SP.
FAIL_PATTERNS = [
    r"^nothing happens",
    r"no known action matches",
    r"\bcan't\b", r"\bcannot\b", r"\bcan not\b",
    r"\bdon't currently have\b",
    r"\bis not (moveable|movable|a region|garbage|open|closed|on|off)\b",
    r"\bis already\b", r"\bare already\b", r"\bhas already been\b",
    r"\bnot sure how to\b", r"\bit's not clear how\b",
    r"\bthat container is empty\b", r"\bthere is only one thing\b",
    r"\bdoes not appear to produce anything new\b",
    r"\bdoes not contain any liquids\b",
    r"\bdoesn't appear to have a temperature\b",
    r"^ambiguous request",
    r"^error: unknown action",
    r"\byou need to put it in your inventory first\b",
    r"\bunable to\b", r"\byou don't have\b", r"\bnot possible\b",
    r"\bis not a valid\b", r"\byou (?:need|should) (?:to )?take\b", r"\bfirst\.$",
]
FAIL_RE = re.compile("|".join(FAIL_PATTERNS), re.I | re.M)

# ---------------------------------------------------------------- verb classes
MOVE_AGENT = {"go", "teleport", "walk", "enter", "exit", "climb"}
CONTAIN = {"take", "pick", "put", "drop", "place", "pour", "dunk", "move", "give", "sweep",
           "fill", "insert", "remove", "get", "throw", "hang", "load", "unload", "bury"}
STATE = {"open", "close", "slice", "chop", "dice", "cut", "clean", "wash", "heat", "cool", "use",
         "turn", "switch", "activate", "deactivate", "connect", "disconnect", "eat", "drink", "mix",
         "cook", "fry", "roast", "boil", "bake", "flush", "light", "burn", "dig", "plant", "water",
         "reset", "change", "rotate", "set", "wear", "apply", "lock", "unlock", "fix", "repair",
         "paint", "toggle", "press", "push", "pull", "break", "fold", "dry", "freeze", "melt",
         "feed", "focus", "attach", "detach", "add", "sharpen", "forge", "shape", "smelt", "wipe",
         "bandage", "spray", "squeeze"}
OBSERVE = {"look", "examine", "inventory", "task", "read", "help", "check", "answer", "wait",
           "wait1", "score", "measure", "inspect"}

# quantities: a number followed by a unit, and phase-of-matter vocabulary
UNIT_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(degrees?(?: celsius| fahrenheit)?|celsius|fahrenheit|°c|°f|"
    r"grams?|g\b|kg|kilograms?|ml|millilit(?:er|re)s?|lit(?:er|re)s?|l\b|cubic (?:cm|centimeters?)|"
    r"cm|centimeters?|m\b|meters?|metres?|seconds?|minutes?|volts?|v\b|ohms?|amps?|watts?|percent|%|"
    r"mph|km|units?)", re.I)
PHASE_WORDS = {"melt", "melted", "melting", "molten", "boil", "boiled", "boiling", "freeze", "frozen",
               "freezing", "evaporate", "evaporated", "evaporating", "condense", "condensed", "steam",
               "liquid", "solid", "gas", "vapor", "vapour", "ice", "combusting", "burning", "ash",
               "warm", "hot", "cold"}

INSTANCE_RE = re.compile(r"\b([a-z]+) (\d+)\b")              # ALFWorld: "mug 1", "cabinet 3"
ID_RE = re.compile(r"([a-z][a-z \-]*?) \(id: \d+\)", re.I)   # ByteSized32: "orange (ID: 4)"
WORD_RE = re.compile(r"[a-z]{4,}")
WORD_REVEAL_VERBS = {"go", "look", "examine", "read", "open", "teleport", "move", "inventory"}
CONTENT_RE =re.compile(r"((?:on|in) (?:the [a-z]+ \d+|it), you see [^.]*)")
STOP = set("""
this that there here with from into onto your have what which when then they them their these those
were been being also very some more most much many such only just than other room called agent
substance currently reading look looking around quickly middle find yourself following items carrying
nothing contains containing currently arrive arrived facing next you see sees seen move moved moves
door doors outside hallway open opened closed close inventory empty task description your about
""".split())

SECTION_RE = re.compile(r"\[([A-Z]+)\] (.*?)(?=\n\[[A-Z]+\] |\Z)", re.S)


def sections(text: str) -> dict:
    """Split a structured observation into its [SECTION] parts. Raw text -> {'FEEDBACK': text}."""
    found = {m.group(1): m.group(2).strip() for m in SECTION_RE.finditer(text or "")}
    return found if found else {"FEEDBACK": (text or "").strip()}


def verb_of(action: str) -> str:
    a = (action or "").strip().lower()
    if not a:
        return ""
    w = a.split()
    if w[0] in {"pick", "put", "turn", "switch", "look"} and len(w) > 1 and w[1] in {"up", "down", "on", "off", "at", "in", "around"}:
        return w[0]
    return w[0]


# a bare reading without a unit: "the time reads 10 ticks", "rotated the iso dial to 3200", "weighs 5"
READING_RE = re.compile(r"\b(?:reads?|reading|to|weighs?|measures?|is now|set to|shows?)\s+(-?\d+(?:\.\d+)?)\b(?!\s*\))", re.I)


def _quantities(text: str, exclude: str = "") -> set:
    text = re.sub(r"\(id: \d+\)", " ", text or "", flags=re.I)
    ex = {n for n, _ in UNIT_RE.findall(exclude or "")} | set(re.findall(r"-?\d+(?:\.\d+)?", exclude or ""))
    q = {(n, u.lower().split()[0]) for n, u in UNIT_RE.findall(text) if n not in ex}
    q |= {(n, "reading") for n in READING_RE.findall(text) if n not in ex}
    return q


def _entities(text: str) -> tuple[set, set, set]:
    t = (text or "").lower()
    inst = {f"{a} {b}" for a, b in INSTANCE_RE.findall(t)}
    ids = {m.strip() for m in ID_RE.findall(t)}
    words = set(WORD_RE.findall(t)) - STOP
    return inst, ids, words


def _state_text(sec: dict) -> str:
    """The part of an observation that describes world state (not the action's feedback)."""
    return "\n".join(sec.get(k, "") for k in ("ROOM", "LOCATION", "INVENTORY"))


def tag_transition(o_t: str, a_t: str, o_next: str, feedback: str | None = None) -> dict:
    """Return {'tags': [...], 'primary': str, 'evidence': {...}} for one transition."""
    s0, s1 = sections(o_t), sections(o_next)
    fb = (feedback if feedback is not None else s1.get("FEEDBACK", o_next or "")).strip()
    verb = verb_of(a_t)
    tags, ev = set(), {"verb": verb}

    failed = bool(FAIL_RE.search(fb))
    if failed:
        tags.add("action_failure")
        ev["fail_match"] = FAIL_RE.search(fb).group(0)

    # world state described before / after (empty when o_next is raw feedback only)
    st0, st1 = _state_text(s0), _state_text(s1)
    full_t = (o_t or "").lower()
    state_changed = bool(st1) and st0 != st1

    if not failed:
        if verb in MOVE_AGENT or (verb == "move" and len(a_t.split()) == 2):   # "move north"
            tags.add("containment_location")
        elif verb in CONTAIN:
            tags.add("containment_location")
        elif verb in STATE:
            if verb in {"eat", "drink"}:                                        # item leaves the inventory
                tags.add("containment_location")
            if not (verb == "focus" and "camera" not in fb.lower()):           # "focus on X" marks, no change
                tags.add("object_state")
        elif verb not in OBSERVE and verb:
            # unknown verb that did not fail: count it as a state change if anything changed
            if state_changed or fb:
                tags.add("object_state")

    # environment-driven change during observe/wait actions (life-cycle stage, temperature drift, ...)
    if verb in OBSERVE and state_changed and not failed:
        loc0, loc1 = s0.get("LOCATION", ""), s1.get("LOCATION", "")
        if loc0 == loc1:
            tags.add("object_state")
            ev["env_driven"] = True

    # numeric / physical: a quantity or phase that CHANGED. When the agent changes location, quantities in
    # the new room's description are newly revealed (partial_obs), not changed, so only feedback counts.
    moved = s0.get("LOCATION", "") != s1.get("LOCATION", "") or verb in MOVE_AGENT
    if verb in OBSERVE:
        # look/examine/read/inventory only REPORT values; a reading counts as a change only if the
        # world-state text itself changed (env-driven: temperature drift while waiting)
        scope = st1 if (state_changed and not moved) else ""
    else:
        scope = fb if moved else fb + "\n" + st1
    q_new = _quantities(scope, exclude=a_t) - _quantities(o_t)
    p_new = (set(WORD_RE.findall(scope.lower())) & PHASE_WORDS) - set(WORD_RE.findall(full_t))
    if (q_new or p_new) and not (failed and not state_changed):
        tags.add("numeric_physical")
        ev["new_quantities"] = sorted(map(" ".join, q_new))[:5]
        ev["new_phase_words"] = sorted(p_new)[:5]

    # partial observability: entities in the next observation that o_t gave no way to know
    reveal_text = fb
    if s1.get("ROOM") and s0.get("LOCATION", "") != s1.get("LOCATION", "") and verb in MOVE_AGENT | {"move"}:
        reveal_text += "\n" + s1.get("ROOM", "")
    elif verb == "open" and s1.get("ROOM") and s1.get("ROOM") != s0.get("ROOM"):
        reveal_text += "\n" + s1.get("ROOM", "")          # ScienceWorld: contents appear in the room text
    if verb in OBSERVE - {"wait", "wait1"} or verb in MOVE_AGENT | {"open", "examine", "look", "move", "use", "dig"}:
        i1, d1, w1 = _entities(reveal_text)
        i0, d0, w0 = _entities(o_t + "\n" + a_t)
        new_inst, new_ids = i1 - i0, d1 - d0
        # bag-of-words evidence only for observation-revealing verbs (ScienceWorld rooms/objects have no IDs)
        new_words = {w for w in (w1 - w0) if w not in full_t} if verb in WORD_REVEAL_VERBS else set()
        # "On the shelf 3, you see nothing." / "In it, you see nothing." is also new information
        # (the contents were unknown), unless o_t already contained the same statement
        empty_news = verb in {"go", "open", "examine"} and any(
            c.strip() and c.strip() not in full_t for c in CONTENT_RE.findall(fb.lower()))
        min_words = 1 if verb == "open" else 3          # opening a container: any new noun is its contents
        if not failed and (new_inst or new_ids or len(new_words) >= min_words or empty_news):
            tags.add("partial_obs")
            ev["revealed"] = sorted(new_inst | new_ids)[:6] or sorted(new_words)[:6]

    primary = next((t for t in PRIMARY_ORDER if t in tags), "none")
    return {"tags": [t for t in TAGS if t in tags], "primary": primary, "evidence": ev}


def tag_record(rec: dict) -> dict:
    """Tag one buffer record in either the exporter format (o_t/a_t/o_next/feedback_next)
    or the JEPA buffer format (obs/action/obs_next/obs_next_full/feedback_next)."""
    o_t = rec.get("o_t", rec.get("obs", ""))
    a_t = rec.get("a_t", rec.get("action", ""))
    o_next = rec.get("o_next", rec.get("obs_next_full") or rec.get("obs_next", ""))
    fb = rec.get("feedback_next")
    return tag_transition(o_t, a_t, o_next, fb)
