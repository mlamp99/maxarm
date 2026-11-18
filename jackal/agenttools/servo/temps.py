#!/usr/bin/env python3
"""The three servos' own temperatures (bus read 26, as agenttools/probe/servo-settings.py), now. Nothing moved but
the connect's reset, which homes the arm.

    MAXARM_DEVICE=/dev/ttyUSB1 PYTHONPATH=src ../.venv/bin/python3 agenttools/servo/temps.py
"""
import os
import time

from maxarm.protocol import ReplProtocol
from maxarm.transport import SerialTransport

HELPER = """
def _probe(i, cmd):
    b = bus_servo
    b.uart.read()
    t = [0x55, 0x55, i, 3, cmd]
    t.append(b.check_sum(t))
    b.send_data(t)
    n = time.ticks_ms()
    while b.uart.any() == 0:
        if time.ticks_ms() - n > 50:
            return None
    return list(b.rx_buf) if b.servo_receive_handle() else None
"""

protocol = ReplProtocol(SerialTransport(os.environ.get("MAXARM_DEVICE", "/dev/ttyUSB0")))
protocol.connect()
protocol.run("import time")
protocol.run(HELPER)
found = {}
for servo in (1, 2, 3):
    reply = protocol.evaluate(f"_probe({servo}, 26)")
    found[servo] = reply[5] if reply and reply[4] == 26 else None
print(time.strftime("%H:%M:%S"), "temperature, C: base", found[1], "shoulder", found[2], "elbow", found[3])
protocol.disconnect()
