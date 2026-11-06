import sys, re, collections, numpy as np
d = dict(np.load(sys.argv[1]))
acts = sorted({k.split("|")[0] for k in d})
groups = collections.defaultdict(list)
for a in acts:
    kind = re.sub(r"\d+", "#", a)
    groups[kind].append((d[a+"|max"], d[a+"|p99.99"], d[a+"|absmean"], a))
print(f"{'tensor kind':70s} {'n':>3s} {'max(med)':>9s} {'max(top)':>9s} {'max/p99.99':>10s} {'max/mean':>9s}")
for k, v in sorted(groups.items()):
    mx = np.array([x[0] for x in v]); p = np.array([x[1] for x in v]); mn = np.array([x[2] for x in v])
    print(f"{k[:70]:70s} {len(v):3d} {np.median(mx):9.2f} {mx.max():9.2f} {np.median(mx/p):10.1f} {np.median(mx/mn):9.0f}")
