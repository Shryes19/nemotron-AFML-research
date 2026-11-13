#!/usr/bin/env python3
"""Collect exactly n WebShop transitions under a reproducible mixed policy.

The environment factory is an explicit adapter because AgentGym releases expose
WebShop through different registration names. Supply a module with
`make_env()` returning a Gym-style environment.
"""
import argparse, importlib, json, random
from pathlib import Path

SEARCH_MARKERS = ("search results", "result [", "page of results")
def is_search_result(observation: str) -> bool:
    return any(m in observation.lower() for m in SEARCH_MARKERS)

def available_actions(info):
    return info.get("available_actions") or info.get("actions") or []

def mixed_action(actions, rng):
    """Half random, half deterministic coverage action; customize policy if available."""
    if not actions: raise RuntimeError("WebShop environment did not expose available actions")
    return rng.choice(actions) if rng.random() < .5 else sorted(map(str, actions))[0]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--factory", required=True, help="module exposing make_env()")
    p.add_argument("--out", default="webshop/webshop_mixed_n128.jsonl"); p.add_argument("--n",type=int,default=128); p.add_argument("--seed",type=int,default=7); a=p.parse_args()
    env=importlib.import_module(a.factory).make_env(); rng=random.Random(a.seed); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    observation, info=env.reset(seed=a.seed); count=0
    with out.open("w",encoding="utf-8") as f:
        while count<a.n:
            actions=available_actions(info)
            action=mixed_action(actions,rng)
            next_obs, reward, terminated, truncated, next_info=env.step(action)
            f.write(json.dumps({"observation":str(observation),"action":str(action),"next_observation":str(next_obs),"partially_observable":is_search_result(str(observation)),"source":"webshop_mixed_policy","reward":reward})+"\n")
            count+=1; observation,info=next_obs,next_info
            if terminated or truncated: observation,info=env.reset()
    print(out)
if __name__=="__main__": main()
