"""Build a calibration set of real KataGo NN inputs from selfplay .npz training files.

The selfplay npz rows are featurized by KataGo's own C++ code, so they match exactly what the
engine feeds the network. Output: an .npz with arrays named after the ONNX graph inputs.
"""
import argparse, glob
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--npz-glob", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=1024)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

spatial, glob_in = [], []
for f in sorted(glob.glob(args.npz_glob, recursive=True)):
    with np.load(f) as d:
        packed = d["binaryInputNCHWPacked"]          # (N, C, ceil(19*19/8)) bits
        n, c, _ = packed.shape
        bits = np.unpackbits(packed, axis=2)[:, :, :19 * 19].reshape(n, c, 19, 19)
        spatial.append(bits.astype(np.float32))
        glob_in.append(d["globalInputNC"].astype(np.float32))
spatial = np.concatenate(spatial); glob_in = np.concatenate(glob_in)
print("total rows", len(spatial), "spatial", spatial.shape, "global", glob_in.shape)

rng = np.random.default_rng(args.seed)
idx = rng.choice(len(spatial), size=min(args.n, len(spatial)), replace=False)
spatial = spatial[idx]; glob_in = glob_in[idx]
mask = spatial[:, 0:1].copy()  # channel 0 = on-board indicator == the mask KataGo feeds
boards = mask.reshape(len(mask), -1).sum(1)
print("board sizes in sample (on-board points):", dict(zip(*np.unique(boards, return_counts=True))))
np.savez(args.out, InputSpatial=spatial, InputGlobal=glob_in[:, :, None, None], InputMask=mask)
print("wrote", args.out)
