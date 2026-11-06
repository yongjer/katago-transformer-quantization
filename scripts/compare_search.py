"""Compare analysis-engine results between runs: top-move agreement, |dWinrate|, |dLead|.

The FP16-vs-FP16 pair measures pure search noise (multi-threaded MCTS is nondeterministic);
a quantized model is "as good as FP16 at search level" if its deviation from FP16 is close to that.
"""
import json, sys, itertools
import numpy as np


def load(p):
    r = {}
    for line in open(p):
        d = json.loads(line)
        if "moveInfos" not in d: continue
        r[(d["id"], d["turnNumber"])] = (d["moveInfos"][0]["move"], d["rootInfo"]["winrate"], d["rootInfo"]["scoreLead"],
                                         d["rootInfo"].get("visits", 0))
    return r


runs = {p.split("/")[-1].replace(".jsonl", ""): load(p) for p in sys.argv[1:]}
base = "fp16_a"
print(f"{'pair':28s} {'n':>4s} {'top-move agree':>14s} {'|dWR| mean':>10s} {'|dWR| p95':>9s} {'|dLead| mean':>12s}")
for other in runs:
    if other == base: continue
    a, b = runs[base], runs[other]
    keys = sorted(set(a) & set(b))
    agree = np.mean([a[k][0] == b[k][0] for k in keys]) * 100
    dwr = np.array([abs(a[k][1] - b[k][1]) for k in keys]) * 100
    dl = np.array([abs(a[k][2] - b[k][2]) for k in keys])
    print(f"{base + ' vs ' + other:28s} {len(keys):4d} {agree:13.1f}% {dwr.mean():9.3f}% {np.percentile(dwr, 95):8.3f}% {dl.mean():12.3f}")
