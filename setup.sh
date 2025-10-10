#!/usr/bin/env bash
set -euo pipefail

# Simple run instructions for MNIST on TPU v4-8
# Usage:
#   bash setup.sh

cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
python3 2_mnist/main.py
