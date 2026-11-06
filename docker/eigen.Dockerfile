FROM nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends libzip4 zlib1g libgomp1 && rm -rf /var/lib/apt/lists/*
