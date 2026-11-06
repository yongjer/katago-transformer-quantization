# KataGo tf3 transformer：FP8 / INT8 量化實驗

模型：`kata1-tf3-b11c768-s12382M-d6533M`（70M 參數）。硬體：RTX 4060 Ti 16GB（sm_89，165 W 功耗上限）。
Runtime：`katago:1.18.1-trt10.16.1` Docker image（TensorRT 10.16.1）。日期：2026-11-06。

## 結論

| 變體 | 純 NN eval/s（batch 8） | 搜尋 visits/s | testgpuerror winrate 平均誤差（vs Eigen FP32） | 800 visits 選點一致率（vs FP16） |
|---|---|---|---|---|
| FP16（現狀） | 656 | ~814–836 | 0.156%（上限的 0.38x） | 87.0%（FP16 跑兩次的雜訊） |
| FP8 全部 | 765（+17%） | — | 2.75%（70x） | 81.5% |
| INT8 + SmoothQuant α=0.8，全部 | 901（+37%） | ~944–968（+16%） | 1.90%（21x） | 85.5% |
| **INT8 + SmoothQuant α=0.8，nbt 投影保留 FP16** | 872（+33%） | analysis 快約 12% | 1.56%（14x） | **88.5%** |

- 建議試用的模型：`models/kata1-tf3-b11c768-s12382M-d6533M-int8sq.onnx`。只能用在 TensorRT backend，直接取代 `-model` 的 `.bin.gz` 即可。
- 在 800 visits 下，它的選點和 FP16 分不出差別。winrate 和目數的偏差約是搜尋雜訊的 1.7–2 倍（winrate 平均 0.38% vs 0.22%，目數 0.19 vs 0.09）。
- 這個模型**沒有通過** KataGo 自己 testgpuerror 的 FP16 標準。純 NN 輸出比 FP16 差一個數量級。還沒做對局 Elo 測試。

## 關鍵發現

1. **不用改 KataGo**：TRT backend 能載入外部 `.onnx`。流程是 `katago dumponnx` → 插入 Q/DQ → 直接用 `-model x.onnx`，`fp32Nodes` metadata 照樣有效。
2. **GeForce Ada 上 FP8 不划算**：TRT 選的 FP8 GEMM 是 FP32 累加（`sm89_xmma_gemm_e4m3f16_e4m3f32_f32`），GeForce 在這個模式下砍半速，峰值等於 FP16 的 FP16 累加。INT8 才是真正的 2 倍。INT8 和 FP8 混用時 TRT 建不出 engine。
3. **SwiGLU 輸出有 LLM 式的離群值**：max/mean = 344，最大值 518。per-tensor INT8 會毀掉 `ffn_linear2`，需要 SmoothQuant。
4. **網路對量化雜訊非常敏感，而且誤差平均分佈**：任何單一 block 量化成 INT8，誤差就超過 FP16，所以「挑層」救不回來。nbt 投影（讀 residual stream）最敏感，但只佔約 8% 運算量。
5. **速度上限**：FP16 engine 裡 GEMM 佔 62%、MHA 佔 14%，所以 GEMM 加倍的上限約 1.37 倍（實測 +37%）。實際搜尋因為 batch 小（平均 4–5）、受延遲限制，只拿到 +12–16%。

## 下一步（評估後決定不做：就算 INT8 完美，這張卡的搜尋加速上限也只有約 +16%）

- **QAT／量化感知蒸餾**：官方 checkpoint 在 `https://media.katagotraining.org/uploaded/networks/zips/kata1/kata1-tf3-b11c768-s12382M-d6533M.zip`。用 fake-quant 學生模型對 FP32 老師做蒸餾，目標是把 NN 誤差壓回 FP16 等級。速度收益仍然受限於上面的 +16%。
- **對局 Elo 測試**：FP16 vs INT8，用等時間換算的 visits 比例（約 1:1.15）。要有統計意義需要數百局，這張卡大概要跑一晚。

## 重現

```bash
K=./scripts/kg.sh                                    # Docker 包裝的 KataGo
$K dumponnx -model <model.bin.gz> -out onnx/tf3_fp32.onnx
# 校準資料：selfplay（selfplay/selfplay_calib.cfg）→ scripts/make_calib.py → calib/calib_1024.npz
docker run ... katago-fp8-tools python scripts/quantize.py calib --onnx onnx/tf3_fp32.onnx --calib calib/calib_1024.npz --out calib/amax_1024_ch.npz
uv run --with onnx --with numpy --with ml_dtypes scripts/quantize.py apply --onnx onnx/tf3_fp32.onnx \
    --amax calib/amax_1024_ch.npz --out model-int8sq.onnx --mode int8 --smooth-alpha 0.8 --exclude normactconv
$K testgpuerror -model model-int8sq.onnx -config configs/gtp_tuned.cfg -reference-file calib/ref19_eigen.txt
```

- `calib/ref19_eigen.txt`：Eigen（CPU）FP32 reference，由 `katago-src/cpp/build-eigen/katago` 產生，和 release binary 是同一個 commit `92ee95c`。
- 其他腳本：`eval_ort.py`（ORT CPU，不經 TRT 的誤差）、`profile_engine.py`、`profile_kinds.py`（TRT 逐層 profile）、`make_queries.py` 加 `compare_search.py`（搜尋層級比較）。
- `onnx/` 底下的中間檔約 10 GB，不需要時可以刪掉，只保留 `models/`。
