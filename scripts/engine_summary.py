import json, sys, collections, re
d = json.load(open(sys.argv[1]))
layers = d.get("Layers", d) if isinstance(d, dict) else d
print("num layers", len(layers))
c = collections.Counter(); fp8 = 0; ex = []
for L in layers:
    if isinstance(L, str): c["str"] += 1; continue
    t = L.get("LayerType", "?"); tac = L.get("TacticName", "") or ""
    c[t] += 1
    ins = json.dumps(L.get("Inputs", []))
    if "FP8" in ins or "fp8" in tac.lower() or "e4m3" in tac.lower(): fp8 += 1; ex.append((L.get("Name","")[:110], tac[:90]))
print(c.most_common(20)); print("layers touching FP8:", fp8)
for e in ex[:8]: print("  ", e)
print("keys of a layer:", list(layers[0].keys()) if isinstance(layers[0], dict) else layers[0][:200])
