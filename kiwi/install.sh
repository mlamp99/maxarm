#!/usr/bin/env bash
# Install the tiles demo on the FRDM-IMX95, as root, where its package was unpacked (README.md):
#
#   ./install.sh
#
# Again after unpacking a newer package over it: config/ -- the calibration, the device's identity -- is not in the
# package and stays. Then ./calibrate.sh, then ./run.sh.
set -e
cd "$(dirname "$(readlink -f "$0")")"

scripts/install-ch341.sh

if [ ! -d venv ]; then
  # --system-site-packages: the image's numpy, cv2, tflite_runtime and gi, never pip's (src/requirements.txt).
  python3 -m venv --system-site-packages venv
fi
. venv/bin/activate
scripts/install.sh

echo "Done. Calibrate: ./calibrate.sh; then the app: ./run.sh"
