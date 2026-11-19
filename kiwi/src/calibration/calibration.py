#!/usr/bin/env python3
"""The calibration web app. One page per calibration, each a tab: the desk first, then the camera; then the tests,
and the tiles.

    ../.venv/bin/python src/calibration/calibration.py                  # then open http://localhost:8080
    ../.venv/bin/python src/calibration/calibration.py --host ::        # every address, for another machine's browser

The arm is on MAXARM_DEVICE, /dev/ttyUSB0 by default. Connecting resets the
board, which homes the arm: 4 s, sometimes 15. This process is the only thing
that talks to the arm, and the only one using the camera; the pages ask it. So
the /IOTCONNECT app (`src/main.py`) cannot run beside it. Ctrl-C holds the arm
where it is and closes the port.

The pages' own files -- HTML, scripts, styles -- are in the pilot's `web/`.
"""

import argparse
import fcntl
import ipaddress
import socket
import struct
import sys
from pathlib import Path
from typing import List, Tuple

HERE = Path(__file__).resolve().parent
WEB = HERE.parents[1] / "web"
sys.path.insert(1, str(HERE.parent))

from maxarm import MaxArm                                    # noqa: E402
from maxarm.spaces import CONFIG_DIR                         # noqa: E402

from applib.camera.feed import Feed                          # noqa: E402
from applib.motion import Motion                             # noqa: E402
from camera.api import CameraSession                         # noqa: E402
from common import arm_api                                   # noqa: E402
from common.server import Server                             # noqa: E402
from common.worker import Worker                             # noqa: E402
from desk.api import DeskSession                             # noqa: E402
from tests.api import TestsSession                           # noqa: E402
from tiles.api import TilesSession                           # noqa: E402


def run(host: str, port: int, config_dir: Path = CONFIG_DIR) -> None:
    worker = Worker(Motion(MaxArm(config_dir=None), config_dir))
    feed = Feed(config_dir)
    server = Server(WEB, home="desk", pages=["camera", "tests", "tiles"])
    arm_api.register(server, worker)
    DeskSession(worker, config_dir).register(server)
    CameraSession(worker, feed, config_dir).register(server)
    TestsSession(worker, feed, config_dir).register(server)
    TilesSession(worker, feed, config_dir).register(server)
    worker.start()
    feed.start()
    print(f"calibration: http://localhost:{port}/  -- Ctrl-C to stop", flush=True)
    if host in ("0.0.0.0", "::"):
        for interface, address in local_addresses(is_ipv6_wanted=host == "::"):
            print(f"  {interface:<8} http://{address}:{port}/", flush=True)
    try:
        server.serve(host, port)
    except KeyboardInterrupt:
        print("\nholding the arm where it is, and closing the port", flush=True)
    finally:
        feed.close()
        worker.shutdown()


_SIOCGIFADDR = 0x8915                     # an interface's IPv4 address, in a struct ifreq


def local_addresses(is_ipv6_wanted: bool) -> List[Tuple[str, str]]:
    """Where another machine can reach this one, by interface: each one's IPv4 address, and its global IPv6 ones,
    in brackets as a URL takes them. Loopback is left out, and so is link-local IPv6 (fe80::): a browser will not
    take the interface such a URL needs."""
    found = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        for _, interface in socket.if_nameindex():
            try:
                reply = fcntl.ioctl(probe.fileno(), _SIOCGIFADDR, struct.pack("256s", interface[:15].encode()))
            except OSError:
                continue                                    # no IPv4 address on it
            address = socket.inet_ntoa(reply[20:24])
            if not address.startswith("127."):
                found.append((interface, address))
    addresses_v6 = Path("/proc/net/if_inet6")
    if is_ipv6_wanted and addresses_v6.exists():
        # Each line: address in hex, interface index, prefix length, scope, flags, interface; scope 00 is global.
        for line in addresses_v6.read_text().splitlines():
            hex_address, _, _, scope, _, interface = line.split()
            if scope == "00":
                found.append((interface, f"[{ipaddress.IPv6Address(int(hex_address, 16))}]"))
    return found


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1",
                        help="the address to serve on: :: (or 0.0.0.0, IPv4 only) to reach it from another machine")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv[1:])
    run(args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
