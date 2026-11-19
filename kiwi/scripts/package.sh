#!/usr/bin/env bash
# The package the board installs from: tiles.tgz, at the top of this tree, unpacked on the board as README.md says.
# Everything that runs on the board -- with build-ch341.sh, for a kernel files/ch341/ has no driver for -- nothing per
# rig (config/), nothing only the PC uses (training/, agenttools/). Owned by root, so it unpacks as root's.
set -e
cd "$(dirname "$(readlink -f "$0")")/.."

tar czf tiles.tgz --exclude=__pycache__ --owner=0 --group=0 \
  install.sh run.sh calibrate.sh README.md src web files \
  scripts/install.sh scripts/install-ch341.sh scripts/build-ch341.sh \
  scripts/install-service.sh scripts/wait-for-rig.sh scripts/systemd
ls -l tiles.tgz
