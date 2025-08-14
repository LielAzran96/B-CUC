# #!/bin/bash

# echo "Starting BCUC performance analysis..."


# #!/bin/bash
# set -euo pipefail

# # Directory of this script (absolute)
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# DEFAULT_INITIAL_Q="0.6275"

# INITIAL_Q="${2:-$DEFAULT_INITIAL_Q}"

# echo "Starting the analysis..."
# echo "Script dir: ${SCRIPT_DIR}"
# echo "Initial Q : ${INITIAL_Q}"

# # Always run the Python from the script dir so relative imports work
# cd "$SCRIPT_DIR"

# # Create logs directory if it doesn't exist
# mkdir -p logs

# # Call the python script in this directory
# python performance.py > logs/performance.log 2>&1 \
#   --initial_Q "$INITIAL_Q" \

# echo "Analysis complete! Results saved in logs/performance.log"


#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

echo "Running performance.py … logs/performance.log"

python performance.py "$@" \
  > logs/performance.log 2>&1

echo "Analysis complete! Results saved in logs/performance.log"