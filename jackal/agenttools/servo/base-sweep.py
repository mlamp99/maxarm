#!/usr/bin/env python3
"""The base servo turned by pulse to each end at home height, to see where it stops and why. MOVES THE ARM.

Connecting re-homes the arm (the board's main.py): shoulder and elbow stay at home, only servo 1 is driven, by
`bus_servo.run(1, pulse, ms)` -- no IK, no firmware clamp (set_servo_in_range clamps servos 2 and 3 only). After each
step: the servo's position read five times (median), its supply. A direction ends when the servo stays more than
STALL_PULSES short of its target after a second wait (a stop: the servo's own limit or the mechanics) or at the end
of its 0-1000 range. Then back to 500 and home. Every step is appended to out/base-sweep.jsonl as it happens.

    PYTHONPATH=src ../.venv/bin/python3 agenttools/servo/base-sweep.py
"""
import json
import statistics
import time
from pathlib import Path

from maxarm.protocol import ReplProtocol
from maxarm.transport import SerialTransport

OUT = Path("agenttools/servo/out/base-sweep.jsonl")
STEP_MS = 700
SETTLE_S = 1.0
STALL_PULSES = 12
# pulse 1000 is the firmware's bearing -120 (left), 0 is +120 (right); 500 straight out
LEGS = {"left": list(range(550, 901, 50)) + list(range(925, 1001, 25)),
        "right": list(range(450, 99, -50)) + list(range(75, -1, -25))}

board = ReplProtocol(SerialTransport())
board.connect()
board.check_board()
log = OUT.open("a")


def note(**row):
    row["t"] = time.strftime("%H:%M:%S")
    log.write(json.dumps(row) + "\n")
    log.flush()
    print(row, flush=True)


def read_base():
    reads = [board.evaluate("bus_servo.get_position(1)") for _ in range(5)]
    good = [r for r in reads if isinstance(r, int)]
    return (statistics.median(good) if good else None), reads


def send(pulse, ms=STEP_MS):
    board.run(f"bus_servo.run(1, {int(pulse)}, {int(ms)})")


note(what="start", servo_methods=board.evaluate("[n for n in dir(bus_servo) if not n.startswith('_')]"),
     pulses=board.evaluate("[bus_servo.get_position(i) for i in (1, 2, 3)]"),
     supply_mv=board.evaluate("bus_servo.get_vin(1)"))
try:
    for leg, targets in LEGS.items():
        for target in targets:
            send(target)
            time.sleep(STEP_MS / 1000 + SETTLE_S)
            at, reads = read_base()
            if at is None or abs(at - target) > STALL_PULSES:
                time.sleep(1.5)
                at, reads = read_base()
            stopped = at is None or abs(at - target) > STALL_PULSES
            note(what="step", leg=leg, target=target, at=at, reads=reads, short=None if at is None else target - at,
                 supply_mv=board.evaluate("bus_servo.get_vin(1)"), stopped=stopped)
            if stopped:
                send(at if at is not None else 500, 300)     # stop pushing where it is
                break
        send(500, 2500)
        time.sleep(3.0)
        at, reads = read_base()
        note(what="back", leg=leg, at=at, reads=reads)
finally:
    send(500, 2500)
    time.sleep(3.0)
    board.run("arm.go_home(2000)")
    time.sleep(2.5)
    note(what="end", pulses=board.evaluate("[bus_servo.get_position(i) for i in (1, 2, 3)]"))
    board.disconnect()
