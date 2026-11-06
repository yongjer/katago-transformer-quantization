#!/usr/bin/env bash
# Builds KataGo (same commit as the TRT release binary) with the Eigen CPU backend, inside Docker.
set -euo pipefail
W="$(cd "$(dirname "$0")/.." && pwd)"
docker run --rm --user 0 -v $W:$W -w $W/katago-src/cpp nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04 bash -c '
  set -e
  apt-get update -qq && apt-get install -y -qq --no-install-recommends cmake libeigen3-dev libzip-dev zlib1g-dev git >/dev/null
  git config --global --add safe.directory "*"
  mkdir -p build-eigen && cd build-eigen
  cmake .. -DUSE_BACKEND=EIGEN -DUSE_AVX2=1 -DCMAKE_BUILD_TYPE=Release -DNO_GIT_REVISION=1 >/dev/null
  make -j20 2>&1 | tail -3
  chown -R '"$(id -u):$(id -g)"' .
'
