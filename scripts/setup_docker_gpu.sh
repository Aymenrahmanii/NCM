#!/usr/bin/env bash
# Let Docker Engine (installed inside WSL Ubuntu) give containers the GPU.
# Needs sudo once:  bash scripts/setup_docker_gpu.sh
# Not needed with Docker Desktop, which ships GPU support for WSL2.
set -e
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
sudo apt-get update -qq
sudo apt-get install -y -qq nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
echo "Test: the next line should print your GPU"
docker run --rm --gpus all ubuntu nvidia-smi --query-gpu=name --format=csv,noheader
