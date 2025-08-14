# #!/bin/bash
# set -euo pipefail

# # Directory of this script (absolute)
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# # Defaults: locate the observations folder one level above this script dir
# DEFAULT_FILE_PATH="${SCRIPT_DIR}/../observations/observations_random_zeros_and_actions_for_Q0.0001_steps_10000.npz"
# DEFAULT_INITIAL_Q="0.0001"
# DEFAULT_PERSON="person_0.0001"

# DIR_PATH="${1:-$DEFAULT_DIR_PATH}"
# INITIAL_Q="${2:-$DEFAULT_INITIAL_Q}"
# PERSON_NUMBER="${3:-$DEFAULT_PERSON}"

# echo "Starting the analysis..."
# echo "Script dir: ${SCRIPT_DIR}"
# echo "Using file: ${FILE_PATH}"
# echo "Initial Q : ${INITIAL_Q}"
# echo "Person     : ${PERSON_NUMBER}"

# # Always run the Python from the script dir so relative imports work
# cd "$SCRIPT_DIR"
# mkdir -p logs

# # Call the python script in this directory
# python q_Q_explosion_analysis.py > logs/q_Q_explosion_analysis.log 2>&1 \
#   --dir_path "$FILE_PATH" \
#   --initial_Q "$INITIAL_Q" \
#   --person_number "$PERSON_NUMBER"

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

python q_Q_explosion_analysis.py "$@" \
  > logs/q_Q_explosion_analysis.log 2>&1

echo "Analysis complete! Results saved in logs/q_Q_explosion_analysis.log"