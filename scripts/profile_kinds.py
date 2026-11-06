import sys, re, collections, json
import tensorrt as trt, torch
plan, batch = sys.argv[1], int(sys.argv[2])
rt = trt.Runtime(trt.Logger(trt.Logger.ERROR)); eng = rt.deserialize_cuda_engine(open(plan, "rb").read()); ctx = eng.create_execution_context()
class P(trt.IProfiler):
    def __init__(s): super().__init__(); s.t = collections.defaultdict(float)
    def report_layer_time(s, n, ms): s.t[n] += ms
for i in range(eng.num_io_tensors):
    n = eng.get_tensor_name(i)
    if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
        sh = list(eng.get_tensor_shape(n)); sh[0] = batch; ctx.set_input_shape(n, sh)
keep = []
for i in range(eng.num_io_tensors):
    n = eng.get_tensor_name(i); t = torch.rand(list(ctx.get_tensor_shape(n)), dtype=torch.float32, device="cuda")
    if n == "InputMask": t.fill_(1)
    keep.append(t); ctx.set_tensor_address(n, t.data_ptr())
st = torch.cuda.Stream()
for _ in range(10): ctx.execute_async_v3(st.cuda_stream)
st.synchronize(); p = P(); ctx.profiler = p; R = 20
for _ in range(R): ctx.execute_async_v3(st.cuda_stream); st.synchronize()
info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
tac = {L["Name"]: (L.get("TacticName") or "") for L in info["Layers"] if isinstance(L, dict)}
def kind(n):
    t = tac.get(n, "")
    if "gemm_mha" in t or "mha" in t.lower(): return "attention (fused MHA)"
    if "gemm" in t.lower() or "h16816" in t or "xmma" in t: return "GEMM"
    m = re.match(r"__myl_([A-Za-z]+)", n)
    return "kgen:" + (m.group(1) if m else n[:30])
agg = collections.defaultdict(float); cnt = collections.Counter()
for n, v in p.t.items(): agg[kind(n)] += v / R; cnt[kind(n)] += 1
tot = sum(agg.values())
print(f"{plan} batch={batch} total {tot:.3f} ms")
for k, v in sorted(agg.items(), key=lambda x: -x[1])[:22]: print(f"  {v:7.3f} ms {100*v/tot:5.1f}% n={cnt[k]:3d} {k}")
