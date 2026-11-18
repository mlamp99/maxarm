#!/usr/bin/env python3
"""Holds the arm's port for an interactive session: one connect (one re-home), then commands one at a time over a
local socket, the arm staying wherever the last one left it. MOVES THE ARM, on command only.

Positions are true arm space through the formula alone (raw.py, no calibration correction): x 0 is base pulse 500.
z is true arm z (the desk at 44).

    PYTHONPATH=src ../.venv/bin/python3 agenttools/servo/hold.py &        # the server
    ../.venv/bin/python3 agenttools/servo/say.py read                      # a command; the reply on stdout

Commands:
    read                          pulses (mean of steady reads), raw and true position, supply
    down X Y Z                    MaxArm.down_to: up to 5 mm over, across, slowly straight down, read as PRECISE
    move X Y Z [fast|slow|precise]
    pulse ID P [MS]               one servo straight to pulse P (bus_servo.run), then read
    home                          the reset pose, climbing first
    eval EXPR                     a board expression's value
    quit                          disconnect and stop
Every command and reply goes to out/hold.jsonl.
"""
import json
import os
import socketserver
import threading
import time
from pathlib import Path

from maxarm import converter, geometry, raw
from maxarm.arm import MaxArm, MoveType

PORT = 8770
LOG = Path("agenttools/servo/out/hold.jsonl")
converter.use(converter.Formula())
arm = MaxArm(os.environ.get("MAXARM_DEVICE", "/dev/ttyUSB0"))
arm.connect()
lock = threading.Lock()
log = LOG.open("a")


def state():
    pulses = arm._read_steady()
    rawp = geometry.forward(geometry.pulse_to_deg(pulses)) if pulses else None
    true = raw.formula_true(rawp) if rawp else None
    supply = arm._protocol.evaluate("bus_servo.get_vin(1)")
    r = lambda v: None if v is None else [round(float(a), 2) for a in v]
    return {"pulses": r(pulses), "raw": r(rawp), "true": r(true), "supply_mv": supply}


def handle(line):
    word, *args = line.split(maxsplit=1)
    rest = args[0].split() if args else []
    if word == "read":
        return state()
    if word in ("down", "move"):
        x, y, z = map(float, rest[:3])
        if word == "down":
            result = arm.down_to(x, y, z)
        else:
            kind = MoveType(rest[3]) if len(rest) > 3 else MoveType.SLOW
            result = arm.move(x, y, z, kind)
        return {"status": result.status.value, "detail": result.detail, **state()}
    if word == "pulse":
        servo, pulse = int(rest[0]), int(rest[1])
        ms = int(rest[2]) if len(rest) > 2 else 1000
        arm._protocol.run(f"bus_servo.run({servo}, {pulse}, {ms})")
        time.sleep(ms / 1000 + 0.8)
        arm._travel_ended, arm._settled = time.monotonic() - 10, None
        now = state()
        if now["true"]:
            arm._position = tuple(now["true"])
        return now
    if word == "home":
        result = arm.home()
        return {"status": result.status.value, **state()}
    if word == "eval":
        return {"value": repr(arm._protocol.evaluate(args[0]))}
    raise ValueError(f"unknown command {word!r}")


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        line = self.rfile.readline().decode().strip()
        with lock:
            try:
                if line == "quit":
                    reply = {"bye": True}
                else:
                    reply = handle(line)
            except Exception as error:      # the session goes on; the reply says what went wrong
                reply = {"error": f"{type(error).__name__}: {error}"}
            log.write(json.dumps({"t": time.strftime("%H:%M:%S"), "command": line, "reply": reply}) + "\n")
            log.flush()
        self.wfile.write((json.dumps(reply) + "\n").encode())
        if line == "quit":
            threading.Thread(target=server.shutdown).start()


socketserver.TCPServer.allow_reuse_address = True
server = socketserver.TCPServer(("127.0.0.1", PORT), Handler)
print(f"holding the arm; commands on 127.0.0.1:{PORT}", flush=True)
try:
    server.serve_forever()
finally:
    arm.disconnect()
