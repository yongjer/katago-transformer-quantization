#!/usr/bin/env bash
# Run the TRT KataGo release binary in Docker with $HOME mounted (same as docker/katago-docker).
exec docker run --rm -i --gpus all --ipc=host --ulimit memlock=-1 --user "$(id -u):$(id -g)" \
  -e HOME="$HOME" -v "$HOME:$HOME" -w "$PWD" "${KATAGO_IMAGE:-katago:1.18.1-trt10.16.1}" "$@"
