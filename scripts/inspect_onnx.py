import sys, collections, onnx
m = onnx.load(sys.argv[1], load_external_data=False)
g = m.graph
print("opset", [(o.domain, o.version) for o in m.opset_import])
for i in g.input: print("IN ", i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim], i.type.tensor_type.elem_type)
for o in g.output: print("OUT", o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim])
print("nodes", len(g.node)); print(collections.Counter(n.op_type for n in g.node).most_common())
for p in m.metadata_props:
    v = p.value if len(p.value) < 200 else p.value[:200] + f"... ({len(p.value.splitlines())} lines)"
    print("META", p.key, "=", v)
print("first MatMul nodes:", [n.name for n in g.node if n.op_type=="MatMul"][:12])
