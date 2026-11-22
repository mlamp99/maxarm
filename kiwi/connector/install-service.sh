#!/usr/bin/env bash
# Start the connector installed here (install.sh) at every boot, as the systemd service iotc-aaf-connector. As root:
#
#   connector/install-service.sh            install it, and start it now
#   connector/install-service.sh --remove   stop it, and take it out of /etc/systemd/system
#
# Installed, run.sh restarts and stops the service rather than a copy of its own. Logs: journalctl -u
# iotc-aaf-connector. Again after moving this directory: the service names its path.
set -e
cd "$(dirname "$(readlink -f "$0")")"

NAME=iotc-aaf-connector
UNIT="/etc/systemd/system/$NAME.service"

if [ "$1" = "--remove" ]; then
  systemctl disable --now "$NAME.service" 2> /dev/null || true
  rm -f "$UNIT"
  systemctl daemon-reload
  echo "Removed the service $NAME. Start the connector by hand again: connector/run.sh"
  exit 0
fi

[ -x venv/bin/connector ] || { echo "The connector is not installed here: connector/install.sh first."; exit 1; }

# One connector on the Ara240 and the port: the one run.sh started, or the service as it was, goes first.
./run.sh --stop
sed "s|@DIR@|$PWD|g" "$NAME.service" > "$UNIT"
systemctl daemon-reload
systemctl enable "$NAME.service"
systemctl restart "$NAME.service"
echo "The service $NAME is starting, and will at every boot. The model takes minutes to load."
echo "Ready when: curl http://127.0.0.1:3000/v1/models; logs: journalctl -u $NAME -f"
