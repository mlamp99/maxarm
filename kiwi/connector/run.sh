#!/usr/bin/env bash
# Start the connector installed here (install.sh), in the background, replacing one already running:
#
#   connector/run.sh            start, or restart, on port 3000
#   connector/run.sh --stop     stop it
#
# Loading the model onto the Ara240 takes minutes, and the port opens only once every model "enabled" in
# config/server_config.json is loaded: until then the app's `agent` command says the model is not answering yet.
# With the service installed (install-service.sh), this restarts or stops the service; it starts again at boot.
set -e
cd "$(dirname "$(readlink -f "$0")")"

[ -x venv/bin/connector ] || { echo "The connector is not installed here: connector/install.sh first."; exit 1; }

SERVICE=iotc-aaf-connector
if [ -f "/etc/systemd/system/$SERVICE.service" ]; then
  if [ "$1" = "--stop" ]; then
    systemctl stop "$SERVICE.service"
    echo "Stopped the service $SERVICE; it starts again at the next boot."
  else
    systemctl restart "$SERVICE.service"
    echo "Restarted the service $SERVICE. The model takes minutes to load. Logs: journalctl -u $SERVICE -f"
  fi
  exit 0
fi

# bin/conn[e]ctor cannot match this script's own command line, nor an ssh session's that runs it.
PIDS=$(pgrep -f "bin/conn[e]ctor" || true)
if [ -n "$PIDS" ]; then
  echo "Stopping the connector: $PIDS"
  # SIGKILL: while the model loads, the server never gets to its SIGTERM handler.
  kill -9 $PIDS
  sleep 2
fi

[ "$1" = "--stop" ] && exit 0

# setsid and </dev/null, so that it outlives the ssh session that started it.
setsid nohup venv/bin/connector --config config/server_config.json --host 127.0.0.1 --port 3000 \
  > server.log 2>&1 < /dev/null &

echo "The connector is starting on port 3000, logging to connector/server.log. The model takes minutes to load."
echo "Ready when the log says 'Uvicorn running on http://127.0.0.1:3000'; then: curl http://127.0.0.1:3000/v1/models"
