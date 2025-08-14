

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/logs"
cd "$SCRIPT_DIR"
mkdir -p logs
python observation_generation.py "$@" \
  > logs/observation_generation.log 2>&1
