#!/usr/bin/env python3
"""Compare decision-utility labels from G and action-lookup baseline outputs.

Inputs are JSONL evaluation records sharing `id` and one numeric utility field.
"""
import argparse, json
from statistics import mean

def load(path, field):
    values={}
    for line in open(path,encoding="utf-8"):
        row=json.loads(line)
        if field not in row: raise ValueError(f"{path}: missing {field} for {row.get('id')}")
        values[str(row["id"])]=float(row[field])
    return values
def main():
    p=argparse.ArgumentParser(); p.add_argument("--g",required=True); p.add_argument("--lookup",required=True)
    p.add_argument("--utility-field",default="decision_utility"); p.add_argument("--out",default="g_vs_action_lookup.json"); a=p.parse_args()
    g,base=load(a.g,a.utility_field),load(a.lookup,a.utility_field); ids=sorted(set(g)&set(base))
    if not ids: raise ValueError("No overlapping ids")
    deltas=[g[i]-base[i] for i in ids]
    report={"n_common":len(ids),"g_mean":mean(g[i] for i in ids),"action_lookup_mean":mean(base[i] for i in ids),"g_minus_lookup_mean":mean(deltas),"g_wins":sum(x>0 for x in deltas),"ties":sum(x==0 for x in deltas)}
    with open(a.out,"w",encoding="utf-8") as f: json.dump(report,f,indent=2); f.write("\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
