#!/usr/bin/env python3
"""Confirm collected observations match the tagger's JSONL contract."""
import argparse, json
def main():
    p=argparse.ArgumentParser(); p.add_argument("jsonl"); p.add_argument("--expected",type=int,default=128); a=p.parse_args()
    rows=[json.loads(x) for x in open(a.jsonl,encoding="utf-8") if x.strip()]
    required={"observation","action","next_observation","partially_observable"}
    assert len(rows)==a.expected, f"expected {a.expected} rows, got {len(rows)}"
    for i,row in enumerate(rows):
        missing=required-row.keys(); assert not missing, f"row {i}: missing {missing}"
        assert all(isinstance(row[x],str) and row[x].strip() for x in ("observation","action","next_observation")), f"row {i}: text field invalid"
        assert isinstance(row["partially_observable"],bool), f"row {i}: tag must be bool"
    print(f"valid: {len(rows)} transitions; {sum(x['partially_observable'] for x in rows)} search-result observations tagged")
if __name__=="__main__": main()
