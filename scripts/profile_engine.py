"""Per-layer profile of a serialized TensorRT engine (as dumped by trtDumpDebugPlanToDir).

Groups layer time into buckets (FP8 GEMM, FP16 GEMM, attention, norm/elementwise, ...) so we can see
where time goes after quantization.
"""
import sys, re, collections
import numpy as np
import tensorrt as trt
import torch

plan, batch = sys.argv[1], int(sys.argv[2])
logger = trt.Logger(trt.Logger.WARNING)
rt = trt.Runtime(logger)
eng = rt.deserialize_cuda_engine(open(plan, "rb").read())
ctx = eng.create_execution_context()


class Prof(trt.IProfiler):
    def __init__(self):
        super().__init__(); self.t = collections.defaultdict(float); self.n = 0
    def report_layer_time(self, name, ms):
        self.t[name] += ms


bufs = {}
for i in range(eng.num_io_tensors):
    name = eng.get_tensor_name(i)
    shape = list(eng.get_tensor_shape(name))
    shape[0] = batch
    if eng.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        ctx.set_input_shape(name, shape)
for i in range(eng.num_io_tensors):
    name = eng.get_tensor_name(i)
    shape = list(ctx.get_tensor_shape(name))
    dt = torch.float32 if eng.get_tensor_dtype(name) == trt.float32 else torch.float16
    t = torch.rand(shape, dtype=dt, device="cuda")
    if name == "InputMask": t.fill_(1.0)
    bufs[name] = t
    ctx.set_tensor_address(name, t.data_ptr())
stream = torch.cuda.Stream()
for _ in range(20):
    ctx.execute_async_v3(stream.cuda_stream)
stream.synchronize()
# Wall time without profiler
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
iters = 100
start.record(stream)
for _ in range(iters):
    ctx.execute_async_v3(stream.cuda_stream)
end.record(stream); stream.synchronize()
wall = start.elapsed_time(end) / iters
prof = Prof(); ctx.profiler = prof
P = 20
for _ in range(P):
    ctx.execute_async_v3(stream.cuda_stream); stream.synchronize()
tot = sum(prof.t.values()) / P

inspector = eng.create_engine_inspector()
import json
info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
tactic = {L["Name"]: (L.get("LayerType", ""), L.get("TacticName", "") or "") for L in info["Layers"] if isinstance(L, dict)}


def bucket(name):
    lt, tac = tactic.get(name, ("", ""))
    tl = tac.lower()
    if "e4m3" in tl: return "GEMM fp8"
    if lt == "gemm" or "gemm" in tl:
        if "/scores" in name or "/sv" in name: return "attention matmul"
        return "GEMM fp16/fp32"
    if "softmax" in name.lower() or "/scores" in name or "/sv" in name or "fmha" in tl or "mha" in tl: return "attention other"
    if "norm" in name: return "RMSNorm"
    if "Conv" in lt or "conv" in name and "nhwc" not in name: return "conv/heads"
    if "Reformat" in lt or "Resh" in name or "Tran" in name or lt in ("reshape", "cast", "NoOp"): return "reshape/cast/transpose"
    return "other elementwise (silu, rope, residual, q/dq...)"


b = collections.defaultdict(float); cnt = collections.Counter()
for k, v in prof.t.items():
    b[bucket(k)] += v / P; cnt[bucket(k)] += 1
print(f"engine={plan} batch={batch} wall={wall:.3f} ms/batch ({batch / wall * 1000:.1f} eval/s)  profiled-sum={tot:.3f} ms")
for k, v in sorted(b.items(), key=lambda x: -x[1]):
    print(f"  {v:8.3f} ms  {100 * v / tot:5.1f}%  n={cnt[k]:4d}  {k}")
top = sorted(prof.t.items(), key=lambda x: -x[1])[:12]
print("  top layers:")
for k, v in top:
    print(f"    {v / P:7.3f} ms  {tactic.get(k, ('', ''))[1][:60]:60s} {k[:90]}")
