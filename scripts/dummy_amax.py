import sys, numpy as np, onnx
m = onnx.load(sys.argv[1], load_external_data=False)
inits = {i.name for i in m.graph.initializer}
acts = {n.input[0] for n in m.graph.node if n.op_type == "MatMul" and n.input[1] in inits}
np.savez(sys.argv[2], **{a + "|max": np.float64(8.0) for a in acts})
print(len(acts), "activation tensors")
