import sys, re, collections, onnx
from onnx import numpy_helper
m = onnx.load(sys.argv[1])
inits = {i.name: i for i in m.graph.initializer}
kinds = collections.Counter(); shapes = {}
for n in m.graph.node:
    if n.op_type not in ("MatMul", "Conv"): continue
    k = re.sub(r"\d+", "#", n.name)
    w = next((inits[x] for x in n.input if x in inits), None)
    kinds[k] += 1; shapes[k] = (n.op_type, tuple(w.dims) if w is not None else "act x act")
for k, c in sorted(kinds.items()): print(f"{c:4d}  {shapes[k][0]:6s} {str(shapes[k][1]):22s} {k}")
