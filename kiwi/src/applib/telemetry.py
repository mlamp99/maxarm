"""What the device tells the cloud, kept in one small object nobody waits on (the property-alarm demo's design).

The values are made on the command threads -- a detection's letters, the tiles in
trouble -- and sent on one thread only, `iotc.py`'s publisher. The producers write
here and the publisher reads here, so no command waits on the network and the app
runs the same with no cloud at all.

    telemetry.set(tiles="team", err_no_reach="", err_too_close="ea")   # sent on every tick from now on
    telemetry.wake()                                                    # and now, not at the end of the tick
    telemetry.set_once(answer="HORSE: an animal people ride.")          # sent once, now, then left out

A value of None is left out of the message, not sent as null: a missing value
says "no reading this time", and the dashboard keeps the last one. A value set
once -- the language model's answer -- is an answer, not a reading: it goes in
the next message alone, and two set within one tick go in two messages, in
order (the property-alarm demo's `set_once`).
"""

from __future__ import annotations

from threading import Event, Lock
from typing import Dict, List, Union

TelemetryValue = Union[str, float, int, bool, None]
MAX_QUEUED_ONCE = 4         # a value's at most: a booth with no network must not grow this without end


class TelemetryState:
    def __init__(self) -> None:
        self._values: Dict[str, TelemetryValue] = {}
        self._once: Dict[str, List[TelemetryValue]] = {}
        self._lock = Lock()
        self._wake = Event()

    def set(self, **values: TelemetryValue) -> None:
        """The latest of one or more values: sent on the next tick, and every tick after."""
        with self._lock:
            self._values.update(values)

    def set_once(self, **values: TelemetryValue) -> None:
        """Values to send once each, in the order set, and the publisher woken for the first."""
        with self._lock:
            for name, value in values.items():
                queued = self._once.setdefault(name, [])
                queued.append(value)
                del queued[:-MAX_QUEUED_ONCE]
        self.wake()

    def wake(self) -> None:
        """Ask the publisher to send now instead of at the end of its interval."""
        self._wake.set()

    def wait_for_wake(self, timeout_s: float) -> bool:
        """The publisher's sleep: True if it was woken rather than timed out."""
        is_woken = self._wake.wait(timeout_s)
        self._wake.clear()
        return is_woken

    def collect(self) -> Dict[str, TelemetryValue]:
        """Everything to send now, the values that are None left out, and one of each value set once; the publisher
        is woken again while more are queued."""
        with self._lock:
            values = dict(self._values)
            for name, queued in self._once.items():
                if queued:
                    values[name] = queued.pop(0)
            is_more_queued = any(self._once.values())
        if is_more_queued:
            self.wake()
        return {name: value for name, value in values.items() if value is not None}
