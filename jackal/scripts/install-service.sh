#!/usr/bin/env bash
# Start one of the two apps at boot, as a systemd service, from this source tree. As root:
#
#   scripts/install-service.sh calibration    # the calibration app's pages, port 8080
#   scripts/install-service.sh app            # the /IOTCONNECT app
#   scripts/install-service.sh none           # neither
#
# Only one can be on: each holds the arm and the camera. Logs: journalctl -u maxarm-calibration, -u maxarm-app.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

CHOICE="${1:-}"
case "$CHOICE" in calibration|app|none) ;; *) sed -n '2,8p' "$0"; exit 2 ;; esac

for name in calibration app; do
  sed "s|@JACKAL@|$PWD|g" "scripts/systemd/maxarm-$name.service" > "/etc/systemd/system/maxarm-$name.service"
done
systemctl daemon-reload
for name in calibration app; do
  if [ "$name" != "$CHOICE" ]; then
    systemctl disable --now "maxarm-$name.service"
  fi
done
if [ "$CHOICE" != none ]; then
  systemctl enable --now "maxarm-$CHOICE.service"
fi
