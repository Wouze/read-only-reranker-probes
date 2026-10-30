#!/usr/bin/env bash
# Download the probe and ridge checkpoints into data/fast06, where the code looks.
set -euo pipefail
cd "$(dirname "$0")/.."
hf download Wouze/read-only-reranker-probes --local-dir data/fast06
