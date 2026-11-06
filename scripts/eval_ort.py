"""Compare ONNX graphs against the FP32 graph with ONNX Runtime (CPU), independent of TensorRT.

Metrics on held-out selfplay positions: winrate error (from value head W/L/noresult softmax),
policy KL divergence (masked board + pass), lead error proxy (raw score channel).
"""
import argparse, glob
import numpy as np
import onnxruntime as ort

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--npz-glob", default="selfplay/out/**/*.npz")
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--skip", type=int, default=0, help="unused")
ap.add_argument("--threads", type=int, default=20)
args = ap.parse_args()

# Held-out = rows not used by make_calib (seed 0, n=1024): rebuild the same permutation.
sp, gl = [], []
for f in sorted(glob.glob(args.npz_glob, recursive=True)):
    with np.load(f) as d:
        p = d["binaryInputNCHWPacked"]; n, c, _ = p.shape
        sp.append(np.unpackbits(p, axis=2)[:, :, :361].reshape(n, c, 19, 19).astype(np.float32))
        gl.append(d["globalInputNC"].astype(np.float32))
sp = np.concatenate(sp); gl = np.concatenate(gl)
rng = np.random.default_rng(0)
calib_idx = set(rng.choice(len(sp), size=min(1024, len(sp)), replace=False).tolist())
held = np.array([i for i in range(len(sp)) if i not in calib_idx])[: args.n]
feeds = {"InputSpatial": sp[held], "InputGlobal": gl[held][:, :, None, None], "InputMask": sp[held][:, 0:1].copy()}
print("held-out positions:", len(held))


def run(path):
    so = ort.SessionOptions(); so.intra_op_num_threads = args.threads
    if path != args.ref: so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    outs = {k: [] for k in ("OutputPolicyPass", "OutputPolicy", "OutputValue", "OutputScoreValue")}
    for i in range(0, len(held), 16):
        r = s.run(list(outs), {k: v[i:i + 16] for k, v in feeds.items()})
        for k, v in zip(outs, r): outs[k].append(v)
    return {k: np.concatenate(v) for k, v in outs.items()}


def stats(o):
    v = o["OutputValue"][:, :, 0, 0]; v = np.exp(v - v.max(1, keepdims=True)); v /= v.sum(1, keepdims=True)
    wr = (v[:, 0] - v[:, 1]) * 0.5 + 0.5
    mask = feeds["InputMask"].reshape(len(held), -1)
    logits = np.concatenate([o["OutputPolicy"][:, 0].reshape(len(held), -1), o["OutputPolicyPass"][:, 0, 0, 0][:, None]], 1)
    m = np.concatenate([mask, np.ones((len(held), 1))], 1)
    logits = np.where(m > 0, logits, -1e30)
    lp = logits - logits.max(1, keepdims=True); lp -= np.log(np.exp(lp).sum(1, keepdims=True))
    return wr, lp, o["OutputScoreValue"][:, 0, 0, 0]


ref = stats(run(args.ref))
for mp in args.models:
    wr, lp, sc = stats(run(mp))
    werr = np.abs(wr - ref[0]) * 100
    kl = (np.exp(ref[1]) * (ref[1] - lp)).sum(1)
    top = (lp.argmax(1) == ref[1].argmax(1)).mean() * 100
    print(f"{mp:40s} winrateErr avg {werr.mean():.3f}% p99 {np.percentile(werr, 99):.3f}%  "
          f"policyKL avg {kl.mean():.6f} p99 {np.percentile(kl, 99):.6f}  top1 agree {top:.1f}%  "
          f"scoreRawErr avg {np.abs(sc - ref[2]).mean():.4f}")
