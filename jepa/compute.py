"""
jepa/compute.py
───────────────
Bridge to the team's shared GPU-hour logger (capability/compute_log.py,
Siddhant's PR #3), so every arm's cost is logged the same way.

    from jepa.compute import track
    with track("jepa_train", arm="D", env="alfworld", game="*", n=256):
        ...

If capability/compute_log.py is not in this checkout yet, track() does nothing
(and says so once), so the JEPA scripts keep working before and after PR #3 is
merged. Log lines go to results/compute/compute_log.jsonl (override with the
COMPUTE_LOG environment variable); summarise with scripts/compute_table.py.

Stage names agreed for the cost table:
    jepa_train, sft_train, nv_embed_encode, rag_index_build, rag_retrieve,
    utility_scoring, probe
"""

from contextlib import contextmanager

try:
    from capability.compute_log import track as _shared_track
except Exception:          # module missing (PR #3 not merged) or import error
    _shared_track = None

_warned = False


@contextmanager
def track(stage: str, arm: str = "", env: str = "", game: str = "", n="", **extra):
    global _warned
    if _shared_track is None:
        if not _warned:
            print("[compute] capability/compute_log.py not found — GPU-hour logging is off "
                  "until PR #3 is merged")
            _warned = True
        yield
        return
    with _shared_track(stage, arm=arm, env=env, game=game, n=n, **extra):
        yield


def env_label(transitions) -> str:
    """'alfworld', or 'alfworld+sciworld' for a mixed buffer."""
    envs = sorted({t.get("env", "unknown") for t in transitions})
    return "+".join(envs)
