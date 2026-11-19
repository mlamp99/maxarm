#!/usr/bin/env bash
# Wait, at boot, for USB to bring up the arm's serial port and a USB camera: started before the port is there, the
# calibration app connects to no arm until CONNECT is pressed, and the /IOTCONNECT app stops. Gives up after WAIT_S
# and lets the app say what is missing.
ARM_DEVICE="${MAXARM_DEVICE:-/dev/ttyUSB0}"
WAIT_S=40

for (( waited = 0; waited < WAIT_S; waited++ )); do
  if [ -e "$ARM_DEVICE" ] && compgen -G "/dev/v4l/by-id/usb-*" > /dev/null; then
    exit 0
  fi
  sleep 1
done
echo "still missing after $WAIT_S s: $ARM_DEVICE or a USB camera"
exit 0
