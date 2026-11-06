# 8-bit Quantization of KataGo's Transformer Network (FP8 / INT8, TensorRT)

[中文版說明](README.zh-TW.md)

An end-to-end study of whether KataGo's newest transformer network, `kata1-tf3-b11c768` (70M parameters, 33 attention + SwiGLU blocks), can run faster with FP8 or INT8 post-training quantization **without losing accuracy** on a consumer GPU (RTX 4060 Ti, Ada, sm_89).

**Outcome:** INT8 with SmoothQuant gave **+33–37% raw network throughput**, but only **+12–16% real search speed**. At 800 visits its move choice was statistically indistinguishable from FP16. Its raw output error, however, was about 10× FP16's, so it did not meet KataGo's own FP16 accuracy bar. I judged the gain too small for the accuracy cost and stopped there, without moving on to quantization-aware training. Every number below was measured. The negative findings are documented as carefully as the positive ones.

## Results

Each variant was compared against an Eigen (CPU) FP32 reference built from the same commit as the release binary.

| Variant | NN eval/s (batch 8) | Winrate error, avg (`testgpuerror`) | Fraction of KataGo's error limit | Top-move agreement with FP16 @800 visits |
|---|---|---|---|---|
| FP16 (baseline) | 656 | 0.156% | 0.38× | 87.0% (FP16 vs FP16 = search noise) |
| FP8, all layers | 765 (+17%) | 2.75% | 70× | 81.5% |
| INT8, all layers, naive | 872 | 26.6% (broken) | 713× | – |
| INT8 + SmoothQuant α=0.8, all layers | 901 (+37%) | 1.90% | 21× | 85.5% |
| **INT8 + SmoothQuant α=0.8, nbt projections in FP16** | 872 (+33%) | 1.56% | 14× | **88.5%** |

Search speed (`katago benchmark`): FP16 ~814–836 visits/s, INT8 ~944–968 visits/s (+16%).

Search-level deviation from FP16 run A (200 positions from self-play games, 800 visits; `results/search_compare.txt`):

| Pair | Top-move agree | mean \|ΔWinrate\| | mean \|ΔLead\| |
|---|---|---|---|
| FP16 vs FP16 (noise floor) | 87.0% | 0.224% | 0.092 |
| FP16 vs INT8 (nbt kept FP16) | 88.5% | 0.384% | 0.188 |
| FP16 vs INT8 (all layers) | 85.5% | 0.515% | 0.191 |
| FP16 vs FP8 (all layers) | 81.5% | 0.706% | 0.252 |

## Key findings

1. **No engine changes were needed.** KataGo's TensorRT backend builds from an ONNX graph and can load an external `.onnx`. So the whole pipeline is `katago dumponnx` → insert Q/DQ nodes (`scripts/quantize.py`) → `-model quantized.onnx`. I kept the original node names, so KataGo's FP32 pinning of RMSNorm and the heads (`katago.fp32Nodes.*` metadata) still applies.
2. **FP8 is a poor fit for GeForce Ada.** The engine layer dump shows TensorRT choosing `sm89_xmma_gemm_e4m3f16_e4m3f32_f32` for FP8, which accumulates in FP32. GeForce runs that mode at half rate, so its peak equals plain FP16 with FP16 accumulation, and FP8 only saves memory traffic (+17%). INT8 with INT32 accumulation runs at full rate, which is the real 2×. TensorRT refused to build an engine that mixed INT8 and FP8.
3. **The FFN has LLM-style activation outliers.** The SwiGLU output that feeds `ffn_linear2` has max/mean ≈ 344 (peaks around 518). Per-tensor INT8 there destroys the network (19% winrate error from that single group). SmoothQuant fixes it.
4. **Error comes from everywhere, not from a few bad layers.** Quantizing any one of the 11 trunk blocks alone already costs 0.17–0.6% winrate error, which is above FP16. Error stacks roughly in quadrature across blocks, so keeping a few blocks in FP16 can't recover FP16-level accuracy. The nested-bottleneck projections, which read the residual stream, are the most sensitive, and they are only about 8% of FLOPs.
5. **Amdahl caps the speedup.** In the FP16 engine, GEMMs take 62% of the time and fused MHA takes 14% (per-layer TensorRT profile). Doubling GEMM speed therefore caps the network at about 1.37×, which matches the measured +37%. MCTS at the latency-optimal batch size (~4–5) captures less than half of that.

## Method

- **Calibration data:** 1,024 positions from short KataGo self-play runs. Using self-play means C++-featurized inputs exactly match what the engine sees, with randomized rules, komi and board size.
- **Quantizer** (`scripts/quantize.py`, written directly against the ONNX graph rather than an automatic tool, for full control):
  - It quantizes the 253 weight MatMuls: QKV/out projections, the SwiGLU FFN and the nested-bottleneck 1×1 projections. Attention score/value matmuls, convolutions, norms and heads are left untouched.
  - Weights get symmetric per-output-channel scales. Activations get per-tensor static scales from max or percentile calibration.
  - SmoothQuant smoothing is optional. Consumers of the same tensor, such as Q/K/V, share one smoothing vector, and the per-tensor activation range is computed exactly from per-channel maxima.
  - Each layer can be set to FP8 or INT8, or excluded.
- **Evaluation:**
  - KataGo `testgpuerror` against an Eigen FP32 reference (2,247 positions) and `runnnevalcanarytests`.
  - ONNX Runtime CPU error on held-out positions, independent of TensorRT. This confirmed that the error comes from quantization itself, not from the engine.
  - Per-group and per-block sensitivity sweeps.
  - TensorRT per-layer profiling, and search-level comparison with the analysis engine.

## Repository layout

```
scripts/quantize.py         calibration (activation stats) + Q/DQ insertion (FP8/INT8, SmoothQuant, per-layer overrides)
scripts/make_calib.py       self-play .npz -> calibration tensors
scripts/eval_ort.py         TensorRT-independent error check with ONNX Runtime
scripts/profile_engine.py   TensorRT per-layer profile (by bucket)
scripts/profile_kinds.py    TensorRT per-layer profile (by fused-kernel kind)
scripts/act_stats.py        activation outlier summary
scripts/make_queries.py     SGF -> analysis-engine queries
scripts/compare_search.py   search-level comparison between runs
scripts/build_eigen.sh      build Eigen KataGo at the release commit (for the FP32 reference)
scripts/kg.sh               run the TensorRT KataGo binary in Docker
docker/                     tooling image (ModelOpt/ONNX/TensorRT python), Eigen runtime image
configs/, selfplay/         KataGo configs used
logs/, results/             raw measurement logs (benchmarks, testgpuerror, sensitivity sweeps, engine layer dumps)
```

## Reproduce

Requirements: Docker with the NVIDIA runtime, an Ada or newer GPU, the KataGo v1.18.1 TensorRT release (commit `92ee95c`) packaged as the image `katago:1.18.1-trt10.16.1`, and the network `kata1-tf3-b11c768-s12382M-d6533M.bin.gz`.

```bash
K=./scripts/kg.sh
$K dumponnx -model kata1-tf3-b11c768-s12382M-d6533M.bin.gz -out onnx/tf3_fp32.onnx
# calibration positions
$K selfplay -config selfplay/selfplay_calib.cfg -models-dir selfplay/models -output-dir selfplay/out -max-games-total 40
docker run --rm -v $PWD:/work -w /work katago-fp8-tools python scripts/make_calib.py --npz-glob "selfplay/out/**/*.npz" --out calib/calib_1024.npz
docker run --rm -v $PWD:/work -w /work katago-fp8-tools python scripts/quantize.py calib \
    --onnx onnx/tf3_fp32.onnx --calib calib/calib_1024.npz --out calib/amax_1024_ch.npz
# quantize (best variant)
python scripts/quantize.py apply --onnx onnx/tf3_fp32.onnx --amax calib/amax_1024_ch.npz \
    --out model-int8sq.onnx --mode int8 --smooth-alpha 0.8 --exclude normactconv
# FP32 reference from Eigen, then accuracy + speed
git clone https://github.com/lightvector/KataGo katago-src && git -C katago-src checkout 92ee95c
./scripts/build_eigen.sh   # then run testgpuerror with the Eigen binary and -reference-file to create calib/ref19_eigen.txt
$K testgpuerror -model model-int8sq.onnx -config configs/gtp_tuned.cfg -reference-file calib/ref19_eigen.txt
$K benchmarknn  -model model-int8sq.onnx -config configs/gtp_tuned.cfg -batch-size 8
```

## What would come next

- **Quantization-aware distillation** from the official raw checkpoint (`media.katagotraining.org/uploaded/networks/zips/kata1/<name>.zip`), aiming to bring INT8 error back to FP16 level.
- **Match games at equal time** to turn the accuracy and speed trade-off into Elo.

I decided against both, because even a perfect INT8 model is capped at about +16% search speed on this hardware.

## Credits

[KataGo](https://github.com/lightvector/KataGo) by David J. Wu and contributors. Network weights are from [katagotraining.org](https://katagotraining.org) and are not redistributed here.
