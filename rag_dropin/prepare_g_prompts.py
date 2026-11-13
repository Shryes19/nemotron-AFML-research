#!/usr/bin/env python3
"""Use NV-Embed-v2 to retrieve examples and persist prompts before starting vLLM.

This deliberately separates NV embedding from Qwen generation: the two models
cannot share the team's 12 GB GPU at the same time.
"""
import argparse, json
from pathlib import Path
from rag_arm import HierarchicalRetriever, prompt, read_jsonl

def main():
    p=argparse.ArgumentParser(); p.add_argument("--buffer",required=True); p.add_argument("--states",required=True)
    p.add_argument("--set",required=True,choices=["alfworld","sciworld","sciworld_seen","bytesized32"])
    p.add_argument("--budget",required=True,type=int,choices=[32,64,128,256,512,1024]); p.add_argument("--outdir",default="g_prompts")
    p.add_argument("--device",default="cuda"); a=p.parse_args()
    buffer=read_jsonl(a.buffer)
    if len(buffer)!=a.budget: raise ValueError(f"Expected {a.budget} D/F training rows, got {len(buffer)}")
    r=HierarchicalRetriever(buffer,device=a.device); states=read_jsonl(a.states)
    out=Path(a.outdir).expanduser()/f"{a.set}_b{a.budget}.jsonl"; out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8") as f:
        for s in states:
            actions=s.get("actions") or s.get("candidate_actions")
            if not actions: raise ValueError(f"{s.get('id')}: no candidate actions")
            obs=s.get("obs",s.get("observation"))
            f.write(json.dumps({"id":s["id"],"actions":actions,"prompts":[prompt(obs,str(action),r.retrieve(obs,str(action))) for action in actions]})+"\n")
    print(out)
if __name__=="__main__": main()
