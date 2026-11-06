"""Insert explicit Q/DQ (FP8 E4M3 or INT8) around selected weight MatMuls of a KataGo ONNX graph.

Design choices:
  * Only MatMuls whose second input is a weight initializer are candidates (QKV/O projections,
    SwiGLU FFN, nested-bottleneck 1x1 projections). Attention score/value MatMuls, convs, norms,
    softmax and the heads stay untouched, so KataGo's fp32Nodes metadata still matches.
  * Weights: per-output-channel symmetric scale (amax / qmax), stored quantized + DequantizeLinear.
  * Activations: per-tensor static scale from calibration (amax or percentile of |x|).
  * Original node names are preserved; new nodes get "<matmul>/qdq_*" names.

Usage:
  python quantize.py calib  --onnx fp32.onnx --calib calib.npz --out amax.npz
  python quantize.py apply  --onnx fp32.onnx --amax amax.npz --out q.onnx [--mode fp8|int8]
                            [--include REGEX ...] [--exclude REGEX ...] [--act-stat max|p99.99]
"""
import argparse, re, sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

FP8_MAX = 448.0
INT8_MAX = 127.0


def weight_matmuls(model):
    inits = {i.name for i in model.graph.initializer}
    return [n for n in model.graph.node if n.op_type == "MatMul" and n.input[1] in inits]


def cmd_calib(args):
    import onnxruntime as ort
    model = onnx.load(args.onnx)
    targets = weight_matmuls(model)
    act_names = sorted({n.input[0] for n in targets})
    existing = {o.name for o in model.graph.output}
    for a in act_names:
        if a not in existing:
            model.graph.output.append(helper.make_tensor_value_info(a, TensorProto.FLOAT, None))
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    data = np.load(args.calib)
    n = len(data["InputSpatial"])
    feeds_all = {k: data[k] for k in ("InputSpatial", "InputGlobal", "InputMask")}
    amax = {a: 0.0 for a in act_names}
    chmax = {a: None for a in act_names}  # per-channel (last axis) amax, for SmoothQuant
    # Keep a subsample of |x| per tensor for percentile stats.
    samples = {a: [] for a in act_names}
    rng = np.random.default_rng(0)
    for s in range(0, n, args.batch):
        feeds = {k: v[s:s + args.batch] for k, v in feeds_all.items()}
        outs = sess.run(act_names, feeds)
        for a, x in zip(act_names, outs):
            ax = np.abs(x).ravel()
            amax[a] = max(amax[a], float(ax.max()))
            cm = np.abs(x).reshape(-1, x.shape[-1]).max(axis=0)
            chmax[a] = cm if chmax[a] is None else np.maximum(chmax[a], cm)
            samples[a].append(rng.choice(ax, size=min(20000, ax.size), replace=False))
        print(f"calib {min(s + args.batch, n)}/{n}", file=sys.stderr, flush=True)
    out = {}
    for a in act_names:
        cat = np.concatenate(samples[a])
        out[a + "|max"] = amax[a]
        for p in (99.9, 99.99, 99.999):
            out[a + f"|p{p}"] = float(np.percentile(cat, p))
        out[a + "|absmean"] = float(cat.mean())
    arrs = {k: np.float64(v) for k, v in out.items()}
    arrs.update({a + "|chmax": chmax[a].astype(np.float64) for a in act_names})
    np.savez(args.out, **arrs)
    print("wrote", args.out, "for", len(act_names), "activation tensors")


def quantize_weight(w, mode):
    # w: (K, N), output channel axis = 1
    amax = np.abs(w).max(axis=0)
    amax = np.where(amax == 0, 1e-8, amax)
    qmax = FP8_MAX if mode == "fp8" else INT8_MAX
    scale = (amax / qmax).astype(np.float32)
    if mode == "fp8":
        import ml_dtypes
        vals = np.clip(w / scale, -FP8_MAX, FP8_MAX).astype(ml_dtypes.float8_e4m3fn)
        t = helper.make_tensor("tmp", TensorProto.FLOAT8E4M3FN, w.shape, vals.tobytes(), raw=True)
        return t, scale
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return numpy_helper.from_array(q), scale


def cmd_apply(args):
    model = onnx.load(args.onnx)
    g = model.graph
    amax = dict(np.load(args.amax))
    inits = {i.name: i for i in g.initializer}
    targets = weight_matmuls(model)
    inc = [re.compile(p) for p in (args.include or [".*"])]
    exc = [re.compile(p) for p in (args.exclude or [])]
    chosen = [n for n in targets if any(p.search(n.name) for p in inc) and not any(p.search(n.name) for p in exc)]
    overrides = [(re.compile(o.split("=")[0]), o.split("=")[1]) for o in (args.mode_override or [])]
    def node_mode(name):
        for pat, m in overrides:
            if pat.search(name): return m
        return args.mode
    smooth_inc = [re.compile(p) for p in (args.smooth_include or [".*"])]

    # Group consumers by (activation tensor, mode): they share one activation Q/DQ (and smoothing).
    groups = {}
    for n in chosen:
        groups.setdefault((n.input[0], node_mode(n.name)), []).append(n)

    new_nodes_before = {}
    mode_counts = {}
    n_smoothed = 0
    for (a, mode), consumers in groups.items():
        qtype = TensorProto.FLOAT8E4M3FN if mode == "fp8" else TensorProto.INT8
        qmax = FP8_MAX if mode == "fp8" else INT8_MAX
        first = consumers[0]
        pre = []
        weights = {n.name: numpy_helper.to_array(inits[n.input[1]]).astype(np.float32) for n in consumers}
        act_in = a
        a_amax = float(amax[a + "|" + args.act_stat]) if not args.no_act else None
        do_smooth = (args.smooth_alpha is not None and not args.no_act
                     and any(p.search(first.name) for p in smooth_inc))
        if do_smooth:
            ch = np.asarray(amax[a + "|chmax"], dtype=np.float64)
            wmax = np.max([np.abs(w).max(axis=1) for w in weights.values()], axis=0).astype(np.float64)
            al = args.smooth_alpha
            sm = np.where((ch > 0) & (wmax > 0), ch ** al / np.maximum(wmax, 1e-12) ** (1 - al), 1.0)
            sm = np.clip(sm, 1e-4, 1e4)
            inv = (1.0 / sm).astype(np.float32)
            iname, mname = f"{first.name}/smooth_inv", f"{first.name}/smooth_mul"
            g.initializer.append(numpy_helper.from_array(inv, iname))
            pre.append(helper.make_node("Mul", [a, iname], [mname + ":0"], name=mname))
            act_in = mname + ":0"
            a_amax = float(np.max(ch / sm))  # exact max of the smoothed activation over calib set
            for k in weights:
                weights[k] = (weights[k] * sm[:, None]).astype(np.float32)
            n_smoothed += 1
        dq_act = None
        if not args.no_act:
            sc = max(a_amax, 1e-6) / qmax
            sname, zname = f"{first.name}/qdq_a_scale", f"{first.name}/qdq_a_zp"
            g.initializer.append(numpy_helper.from_array(np.array(sc, dtype=np.float32), sname))
            g.initializer.append(helper.make_tensor(zname, qtype, [], [0]))
            qn, dqn = f"{first.name}/qdq_a_q", f"{first.name}/qdq_a_dq"
            pre.append(helper.make_node("QuantizeLinear", [act_in, sname, zname], [qn + ":0"], name=qn))
            pre.append(helper.make_node("DequantizeLinear", [qn + ":0", sname, zname], [dqn + ":0"], name=dqn))
            dq_act = dqn + ":0"
        new_nodes_before[first.name] = pre
        for n in consumers:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            if dq_act is not None:
                n.input[0] = dq_act
            w = weights[n.name]
            w_init = inits[n.input[1]]
            if args.no_weight:
                if do_smooth:  # smoothed weights must still replace the originals
                    g.initializer.remove(w_init)
                    g.initializer.append(numpy_helper.from_array(w, w_init.name))
                continue
            qt, scale = quantize_weight(w, mode)
            qt.name = f"{n.name}/qdq_w"
            g.initializer.append(qt)
            sname, zname = f"{n.name}/qdq_w_scale", f"{n.name}/qdq_w_zp"
            g.initializer.append(numpy_helper.from_array(scale, sname))
            g.initializer.append(helper.make_tensor(zname, qtype, [w.shape[1]], [0] * w.shape[1]))
            dqn = f"{n.name}/qdq_w_dq"
            new_nodes_before.setdefault(n.name, [])
            if n.name != first.name:
                new_nodes_before[n.name] = new_nodes_before.get(n.name, [])
            new_nodes_before[n.name].append(
                helper.make_node("DequantizeLinear", [qt.name, sname, zname], [dqn + ":0"], name=dqn, axis=1))
            g.initializer.remove(w_init)
            n.input[1] = dqn + ":0"
    nodes = list(g.node)
    del g.node[:]
    for n in nodes:
        g.node.extend(new_nodes_before.get(n.name, []))
        g.node.append(n)
    entry = model.metadata_props.add()
    entry.key = "katago.info.quantization"
    desc = (f"mode={args.mode} overrides={args.mode_override} counts={mode_counts} act_stat={args.act_stat} "
            f"smooth_alpha={args.smooth_alpha} smoothed_groups={n_smoothed} no_act={args.no_act} no_weight={args.no_weight} "
            f"include={args.include} exclude={args.exclude}")
    entry.value = desc
    onnx.save(model, args.out)
    print(f"{args.out}: {desc}")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
c = sub.add_parser("calib"); c.add_argument("--onnx", required=True); c.add_argument("--calib", required=True)
c.add_argument("--out", required=True); c.add_argument("--batch", type=int, default=8); c.add_argument("--threads", type=int, default=20)
a = sub.add_parser("apply"); a.add_argument("--onnx", required=True); a.add_argument("--amax", required=True)
a.add_argument("--out", required=True); a.add_argument("--mode", default="fp8", choices=["fp8", "int8"])
a.add_argument("--include", nargs="*"); a.add_argument("--mode-override", nargs="*"); a.add_argument("--exclude", nargs="*")
a.add_argument("--smooth-alpha", type=float); a.add_argument("--smooth-include", nargs="*")
a.add_argument("--no-act", action="store_true"); a.add_argument("--no-weight", action="store_true")
a.add_argument("--act-stat", default="max", choices=["max", "p99.9", "p99.99", "p99.999"])
args = ap.parse_args()
{"calib": cmd_calib, "apply": cmd_apply}[args.cmd](args)
