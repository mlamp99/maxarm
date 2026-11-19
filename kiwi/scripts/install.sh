#!/usr/bin/env bash
# The apps' Python packages, into the python3 on PATH -- an activated venv's, if there is one -- and the device's
# certificate and key in config/ if it has none yet. On the board ../install.sh runs it in the app's venv; on a PC,
# activate a venv first if you want one:
#
#   scripts/install.sh
set -e
cd "$(dirname "$(readlink -f "$0")")/.."

if [ "$(uname -m)" = aarch64 ]; then
  python3 -m pip install -r src/requirements.txt
else
  python3 -m pip install -r src/requirements-pc.txt
fi

mkdir -p config
if [ ! -f config/device-pkey.pem ]; then
  openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -days 3650 \
    -keyout config/device-pkey.pem -out config/device-cert.pem -subj "/CN=maxarm"
  echo "Made config/device-cert.pem: register the device with /IOTCONNECT with it."
fi
