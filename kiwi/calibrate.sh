#!/usr/bin/env bash
# The calibration app, on the board, in the venv install.sh made: open http://<board>:8080 from another machine.
# Ctrl-C to stop; the /IOTCONNECT app (./run.sh) cannot run beside it.
set -e
cd "$(dirname "$(readlink -f "$0")")"

# What the image brings, not pip (src/requirements.txt): stop here rather than part-way through.
venv/bin/python -c "import numpy, cv2, tflite_runtime, gi"

export PYTHONUNBUFFERED=1
exec venv/bin/python src/calibration/calibration.py --host :: "$@"
