#!/bin/bash

echo "Run docker"
docker run --gpus '"device=7"' -w /home/workspace --rm -it -p 10007:8888 -v /data/wl_data:/wl_data wav2lip-hq /bin/bash
