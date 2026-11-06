# Tooling image for quantizing KataGo ONNX graphs (ModelOpt, onnx, TensorRT python, ORT).
FROM nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir numpy onnx onnx-graphsurgeon onnxruntime polygraphy \
      "tensorrt-cu13==10.16.1.11" \
 && pip install --no-cache-dir "nvidia-modelopt[onnx]"
WORKDIR /work
