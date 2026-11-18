#!/usr/bin/env python3
"""Landings measured by the servos' reads: where they settle against the target, by place, height and the way the
cup is brought down. MOVES THE ARM: by the owner's release (17 Nov evening, "play around with the arm").

The ways (each a pass over every place, the places in a shuffled order so the swing before a landing varies):
    lib      the library's landing as it is (`landing.py`): across at the hover height to 6 base pulses on the lower
             side, the base turned onto the target aimed 2 past, one slow descent, the shoulder and elbow given
             EXTRA_PULSES; read steady. Not landed again (whether the library would have is logged).
    settle   the same, read steady at the approach and after the base's turn before going on: does the swing
             before (its momentum) change where the base and the descent settle?
    elbow    the base as `settle`; then the shoulder and elbow together to the shoulder's landing pulse and the
             elbow ELBOW_LAST short of its own, read steady; then the elbow alone the rest, aimed E3_ELBOW past:
             the last move one joint, always the same length and way (as the base's).
    falling  as lib, the last base turn from the higher pulses (the library's second way), aimed 1 past.

Heights (true arm z, the desk at 44): 1 mm over a 40 mm cube's top, as FIND RED CUBE lands, from the camera page's
hover (desk 60); and 5 mm over the desk, from the desk hover (20). Places: bearings -35..+80 deg, 120-265 mm out,
and -45 only from 240 mm out (the owner's mouse at about -60, 100-215 mm out; cables past +90 near the arm).

    PYTHONPATH=src ../.venv/bin/python3 agenttools/servo/landings.py dry        # the plan checked, no arm
    MAXARM_DEVICE=/dev/ttyUSB1 PYTHONPATH=src ../.venv/bin/python3 agenttools/servo/landings.py run
A file agenttools/servo/out/STOP ends the run after the landing under way (the arm homed).
One JSON line per landing in agenttools/servo/out/landings-<time>.jsonl.
"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

from maxarm import converter, geometry, landing, raw
from maxarm.arm import MOVE_FAST_MM_S, MOVE_SLOW_MM_S, MaxArm

converter.use(converter.Formula())              # the formula alone: no camera correction under the study

DESK = raw.DESK_Z_MM
CUBE_TOP, LOW = DESK + 41.0, DESK + 5.0
HOVER = {CUBE_TOP: DESK + 60.0, LOW: DESK + 20.0}
BEARINGS = (-35.0, -20.0, 0.0, 20.0, 40.0, 60.0, 80.0)
REACHES = (120.0, 150.0, 180.0, 210.0, 240.0, 265.0)
FAR_LEFT = ((-45.0, 240.0), (-45.0, 265.0))
PASSES = ("lib", "settle", "elbow", "falling", "lib")
ELBOW_LAST = 10                                 # pulses: the elbow's last move alone
E3_ELBOW = 2.0                                  # aimed past by this: its stop short, a first guess
ELBOW_MS = 300
LOWEST_ON_THE_WAY_MM = {CUBE_TOP: DESK + 30.0, LOW: DESK + 2.0}   # no path between commands may dip under this
OUT = Path("agenttools/servo/out")
STOP = OUT / "STOP"


def places():
    found = [(b, r) for b in BEARINGS for r in REACHES] + list(FAR_LEFT)
    return [(b, r, r * math.sin(math.radians(b)), -r * math.cos(math.radians(b))) for b, r in found]


def pulses_of(point):
    return landing._pulses(point)


def true_of(pulses):
    return raw.formula_true(geometry.forward(geometry.pulse_to_deg(pulses)))


def lowest_between(a, b, steps=24):
    """The lowest true z the cup passes going from pulses a to b, each servo straight in its own pulses."""
    return min(true_of(tuple(p + (q - p) * k / steps for p, q in zip(a, b)))[2] for k in range(steps + 1))


def plan(place, z, way):
    """The commands of one landing, in pulses, and the target's exact pulses."""
    _, reach, x, y = place
    exact = pulses_of((x, y, z))
    high = pulses_of((x, y, HOVER[z]))
    e2, e3 = landing.extra_pulses(reach)
    side = 1 if way == "falling" else -1
    aimed = exact[0] - side * (landing.AIM_PAST_FALLING if side > 0 else landing.AIM_PAST_RISING)
    steps = {"approach": (exact[0] + side * landing.APPROACH_PULSES, high[1], high[2]),
             "over": (aimed, high[1], high[2])}
    if way == "elbow":
        steps["pre"] = (aimed, exact[1] + e2, exact[2] - ELBOW_LAST)
        steps["elbow"] = (aimed, exact[1] + e2, exact[2] + E3_ELBOW)
    else:
        steps["down"] = (aimed, exact[1] + e2, exact[2] + e3)
    return exact, steps


def problems(place, z, way, cross_from=None):
    """What would make this landing unsafe or impossible, as text; empty if none."""
    found = []
    try:
        exact, steps = plan(place, z, way)
    except geometry.Unreachable as reason:
        return [f"unreachable: {reason}"]
    for name, pulses in steps.items():
        reason = geometry.limit_reason(geometry.forward(geometry.pulse_to_deg(pulses)))
        if reason:
            found.append(f"{name}: {reason}")
    names = list(steps)
    for a, b in zip(names, names[1:]):
        low = lowest_between(steps[a], steps[b])
        if low < min(z, LOWEST_ON_THE_WAY_MM[z]) - 0.3:
            found.append(f"{a}->{b} dips to {low - DESK:.1f} over the desk")
    if cross_from is not None:
        low = lowest_between(cross_from, steps["approach"])
        if low < LOWEST_ON_THE_WAY_MM[z]:
            found.append(f"crossing dips to {low - DESK:.1f} over the desk")
    return found


def schedule(seed=17):
    """Every landing in order: (pass index, way, height, place)."""
    order = []
    for height in (CUBE_TOP, LOW):
        for index, way in enumerate(PASSES):
            chosen = places()
            random.Random(seed + index * 7 + int(height)).shuffle(chosen)
            order += [(index, way, height, place) for place in chosen]
    return order


def dry():
    order = schedule()
    previous, bad = None, 0
    for index, way, height, place in order:
        text = problems(place, height, way, previous)
        if text:
            bad += 1
            print(f"{way:8s} z{height - DESK:+.0f} b{place[0]:+.0f} r{place[1]:.0f}: {'; '.join(text)}")
        try:
            previous = plan(place, height, way)[1].get("down") or plan(place, height, way)[1]["elbow"]
            previous = (previous[0], *pulses_of((place[2], place[3], HOVER[height]))[1:])   # lifted to the hover
        except geometry.Unreachable:
            previous = None
    print(f"{len(order)} landings, {bad} with a problem")


# --- on the arm ------------------------------------------------------------

class Run:
    def __init__(self, arm: MaxArm, log):
        self.arm, self.log = arm, log

    def drive(self, pulses, speed):
        failed = self.arm._drive(landing._point(pulses), speed, None)
        if failed is not None:
            raise RuntimeError(f"{failed.status.value}: {failed.detail}")

    def elbow_alone(self, pulse, ms):
        self.arm._protocol.run(f"bus_servo.run(3, {int(round(pulse))}, {int(ms)})")
        time.sleep(ms / 1000.0)
        self.arm._travel_ended, self.arm._settled = time.monotonic(), None
        here = self.read()
        self.arm._position = landing._point(here)
        return here

    def read(self):
        pulses = self.arm._steady()
        if pulses is None:
            raise RuntimeError("no steady read")
        return [round(p, 3) for p in pulses]

    def here_pulses(self):
        pulses = self.arm._read_steady()
        return pulses

    def up_to(self, z):
        """Straight up (or nowhere) to true z where the cup is: fast, nothing held."""
        x, y, now = self.arm._position
        if now < z - 0.05:
            failed = self.arm._drive((x, y, z), MOVE_FAST_MM_S, None)
            if failed is not None:
                raise RuntimeError(f"{failed.status.value}: {failed.detail}")

    def landing(self, index, way, height, place, previous):
        bearing, reach, x, y = place
        exact, steps = plan(place, height, way)
        record = {"t": time.strftime("%H:%M:%S"), "pass": index, "way": way, "height": round(height - DESK, 2),
                  "bearing": bearing, "reach": reach, "target": [round(x, 3), round(y, 3), round(height, 3)],
                  "exact": [round(p, 3) for p in exact], "steps": {k: [round(p, 3) for p in v] for k, v in
                                                                  steps.items()},
                  "previous": previous, "from_base": None, "reads": {}}
        started = time.monotonic()
        self.up_to(HOVER[height])
        here = self.arm._protocol.read_pulses()
        record["from_base"] = here[0] if here else None
        self.drive(steps["approach"], MOVE_FAST_MM_S)
        if way in ("settle", "elbow"):
            record["reads"]["approach"] = self.read()
        self.drive(steps["over"], MOVE_FAST_MM_S)
        if way in ("settle", "elbow"):
            record["reads"]["over"] = self.read()
        if way == "elbow":
            self.drive(steps["pre"], MOVE_SLOW_MM_S)
            record["reads"]["pre"] = self.read()
            record["reads"]["final"] = self.elbow_alone(steps["elbow"][2], ELBOW_MS)
        else:
            self.drive(steps["down"], MOVE_SLOW_MM_S)
            record["reads"]["final"] = self.read()
        final = record["reads"]["final"]
        landed = true_of(final)
        record["landed"] = [round(v, 3) for v in landed]
        record["off"] = {"base": round(final[0] - exact[0], 3), "s2": round(final[1] - exact[1], 3),
                         "s3": round(final[2] - exact[2], 3), "z": round(landed[2] - height, 3),
                         "reach": round(math.hypot(*landed[:2]) - math.hypot(x, y), 3),
                         "side_mm": round(math.radians((final[0] - exact[0]) * 0.24) * math.hypot(x, y), 3)}
        way_obj = landing.Landing(None, None, None, (x, y, height), exact[0])
        record["library_lands_again"] = landing.is_worth_again(way_obj, final[0])
        record["seconds"] = round(time.monotonic() - started, 2)
        self.log.write(json.dumps(record) + "\n")
        self.log.flush()
        self.up_to(HOVER[height])
        return record


def run():
    OUT.mkdir(exist_ok=True)
    if STOP.exists():
        STOP.unlink()
    path = OUT / f"landings-{time.strftime('%m%d-%H%M%S')}.jsonl"
    arm = MaxArm(os.environ.get("MAXARM_DEVICE", "/dev/ttyUSB0"))
    arm.connect()
    order = schedule()
    print(f"{len(order)} landings into {path}", flush=True)
    with path.open("a") as log:
        go = Run(arm, log)
        previous = None
        try:
            for count, (index, way, height, place) in enumerate(order, 1):
                if STOP.exists():
                    print("STOP file: ending", flush=True)
                    break
                if problems(place, height, way):
                    print(f"skipped {way} b{place[0]} r{place[1]} z{height - DESK:+.0f}", flush=True)
                    continue
                record = go.landing(index, way, height, place, previous)
                previous = [place[0], place[1]]
                off = record["off"]
                print(f"{count:3d} {way:8s} z{record['height']:+.0f} b{place[0]:+4.0f} r{place[1]:3.0f}  base "
                      f"{off['base']:+.2f}  s2 {off['s2']:+.2f}  s3 {off['s3']:+.2f}  z {off['z']:+.2f}  reach "
                      f"{off['reach']:+.2f}  ({record['seconds']:.1f} s)", flush=True)
        finally:
            try:
                go.up_to(HOVER[CUBE_TOP])
                arm.home()
            finally:
                arm.disconnect()


def confirm():
    """The library's own landing, `MaxArm.down_to()`, as the app calls it (its travel height the app's hover), at
    the study's places and at places between them, both heights; logged as the passes are, way "confirm"."""
    OUT.mkdir(exist_ok=True)
    if STOP.exists():
        STOP.unlink()
    path = OUT / f"confirm-{time.strftime('%m%d-%H%M%S')}.jsonl"
    between = [(b, r) for b in (-28.0, -10.0, 10.0, 30.0, 50.0, 70.0) for r in (135.0, 195.0, 255.0)]
    chosen = places() + [(b, r, r * math.sin(math.radians(b)), -r * math.cos(math.radians(b))) for b, r in between]
    turns = []                                  # each base turn the library judged: (pulses off, turned again?)
    judge = landing.is_worth_again

    def counted(way, base_read):
        again = judge(way, base_read)
        turns.append((round(base_read - way.base, 3), again))
        return again
    landing.is_worth_again = counted
    arm = MaxArm(os.environ.get("MAXARM_DEVICE", "/dev/ttyUSB0"))
    arm.connect()
    print(f"into {path}", flush=True)
    count = 0
    with path.open("a") as log:
        try:
            for height in (CUBE_TOP, LOW):
                order = list(chosen)
                random.Random(int(height) + 99).shuffle(order)
                for place in order:
                    if STOP.exists():
                        raise KeyboardInterrupt
                    bearing, reach, x, y = place
                    started = time.monotonic()
                    here = arm._position
                    if here[2] < HOVER[height] - 0.05:
                        arm._drive((here[0], here[1], HOVER[height]), MOVE_FAST_MM_S, None)
                    arm.set_clearance(HOVER[height] - height - 5.0)
                    turns.clear()
                    try:
                        result = arm.down_to(x, y, height)
                    finally:
                        arm.set_clearance(0.0)
                    if not result:
                        print(f"refused b{bearing} r{reach}: {result.detail}", flush=True)
                        continue
                    final = list(arm._settled)
                    exact = pulses_of((x, y, height))
                    landed = true_of(final)
                    off = {"base": round(final[0] - exact[0], 3), "s2": round(final[1] - exact[1], 3),
                           "s3": round(final[2] - exact[2], 3), "z": round(landed[2] - height, 3),
                           "reach": round(math.hypot(*landed[:2]) - reach, 3),
                           "side_mm": round(math.radians((final[0] - exact[0]) * 0.24) * reach, 3)}
                    count += 1
                    record = {"t": time.strftime("%H:%M:%S"), "pass": 9, "way": "confirm",
                              "height": round(height - DESK, 2), "bearing": bearing, "reach": reach,
                              "target": [round(x, 3), round(y, 3), round(height, 3)],
                              "exact": [round(p, 3) for p in exact], "final": [round(p, 3) for p in final],
                              "landed": [round(v, 3) for v in landed], "off": off, "detail": result.detail,
                              "seconds": round(time.monotonic() - started, 2), "library_lands_again": False,
                              "steps": {}, "from_base": None, "reads": {"final": final},
                              "turns": list(turns)}
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(f"{count:3d} confirm z{record['height']:+.0f} b{bearing:+4.0f} r{reach:3.0f}  base "
                          f"{off['base']:+.2f}  s2 {off['s2']:+.2f}  s3 {off['s3']:+.2f}  z {off['z']:+.2f}  "
                          f"reach {off['reach']:+.2f}  ({record['seconds']:.1f} s)", flush=True)
        except KeyboardInterrupt:
            print("stopped", flush=True)
        finally:
            try:
                here = arm._position
                arm._drive((here[0], here[1], HOVER[CUBE_TOP]), MOVE_FAST_MM_S, None)
                arm.home()
            finally:
                arm.disconnect()


if __name__ == "__main__":
    {"dry": dry, "run": run, "confirm": confirm}[sys.argv[1]]()
