#!/usr/bin/env bash
# The CH340/CH341 USB serial driver that the FRDM-IMX95 image leaves out, so the arm's ESP32 appears as /dev/ttyUSB0.
# As root, on the board. A driver already there is left as it is. Else the one built for this exact kernel is taken
# from files/ch341/<uname -r>/; for a kernel the package has none for, scripts/build-ch341.sh builds it first, which
# takes a while. From then on it loads whenever the arm is plugged in.
set -e
cd "$(dirname "$(readlink -f "$0")")/.."

if modinfo ch341 > /dev/null 2>&1; then
  echo "ch341: already installed"
  exit 0
fi

if [ ! -f files/ch341/$(uname -r)/ch341.ko ]; then
  echo "ch341: none built for kernel $(uname -r) in files/ch341/; building it (~20 minutes, 2.2 GB in ~/kbuild)"
  scripts/build-ch341.sh
  mkdir -p files/ch341/$(uname -r)
  cp ${HOME}/kbuild/ch341/ch341.ko files/ch341/$(uname -r)/
fi

mkdir -p /lib/modules/$(uname -r)/updates
cp files/ch341/$(uname -r)/ch341.ko /lib/modules/$(uname -r)/updates/
depmod -a
modprobe ch341
echo "ch341: installed"
