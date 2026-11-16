#!/usr/bin/env python3
"""Is the camera's model short of lens distortion, or its field of view off? One camera fitted to every cube's edges
on file (CALIBRATE's three and the Fine Tune points, all one unmoved camera): a pinhole as the app has it, then with
radial distortion (k1, then k1 and k2). Each cube a 40 mm cube free to stand and turn; the world held by point 1.
Prints the focal length and field of view, the distortion, and how far each cube's edge points sit off its edges.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/anchor/lens.py
"""
import math
from pathlib import Path

import cv2
import numpy as np

from applib.camera import edges, store
from applib.camera.model import CUBE_MM, cube_corners, least_squares
from applib.camera.sighting import in_frame, place, start

CONFIG = Path("config")
fit = store.load_camera(CONFIG).cube_fit()
points = [point for point in store.load_refine_points(CONFIG) if point.edges]
K, R, centre = cv2.decomposeProjectionMatrix(fit.true_to_pixel)[:3]
K = K / K[2, 2]
centre = (centre[:3] / centre[3]).ravel()
rvec = cv2.Rodrigues(R)[0].ravel()
tvec = -R @ centre
cubes = []
for point in points:
    sighting = point.sighting
    matrix = in_frame(fit.true_to_pixel, fit.frame_size, sighting)
    cubes.append(place(matrix, sighting, start(matrix, sighting)[:3]))
CAMERA = 11                 # f, cx, cy, rvec 3, tvec 3, k1, k2
first = np.array([K[0, 0], K[0, 2], K[1, 2], *rvec, *tvec, 0.0, 0.0, *np.concatenate(cubes)])
steps = np.array([0.05, 0.05, 0.05, 1e-6, 1e-6, 1e-6, 0.005, 0.005, 0.005, 1e-5, 1e-5] + [0.005, 0.005, 1e-5] * len(cubes))


def project(params, world):
    f, cx, cy = params[:3]
    rot = cv2.Rodrigues(params[3:6])[0]
    cam = world @ rot.T + params[6:9]
    x, y = cam[:, 0] / cam[:, 2], cam[:, 1] / cam[:, 2]
    r2 = x * x + y * y
    factor = 1 + params[9] * r2 + params[10] * r2 * r2
    return np.column_stack((f * x * factor + cx, f * y * factor + cy))


def misses(params):
    out = []
    for index, point in enumerate(points):
        x, y, turn = params[CAMERA + 3 * index: CAMERA + 3 * index + 3]
        starts, ends, found = point.sighting.edges
        out.append(edges.signed_distances(project(params, cube_corners((x, y), turn, CUBE_MM)), starts, ends, found))
    return np.concatenate(out)


def report(name, params):
    w, h = fit.frame_size
    f = params[0]
    rot = cv2.Rodrigues(params[3:6])[0]
    cam = -rot.T @ params[6:9]
    print(f"{name}: f {f:.1f} px, FoV h {2 * math.degrees(math.atan(w / 2 / f)):.2f} deg (pinhole), principal "
          f"({params[1]:.0f}, {params[2]:.0f}), k1 {params[9]:+.4f} k2 {params[10]:+.4f}, camera {cam[2]:.1f} mm up")
    if params[9] or params[10]:
        corner = math.hypot(w / 2, h / 2) / f
        r2 = corner ** 2
        print(f"    at the frame's corner the picture is {100 * (params[9] * r2 + params[10] * r2 * r2):+.1f}% of the "
              f"pinhole's radius; full-frame FoV h {2 * math.degrees(math.atan(_undistort(params, w / 2 / f))):.2f} deg")
    for index, point in enumerate(points):
        x, y, turn = params[CAMERA + 3 * index: CAMERA + 3 * index + 3]
        starts, ends, found = point.sighting.edges
        r = np.abs(edges.signed_distances(project(params, cube_corners((x, y), turn, CUBE_MM)), starts, ends, found))
        print(f"    #{point.id} {point.cube or 'red':5}: off its edges median {np.median(r):.2f} px, 90% "
              f"{np.percentile(r, 90):.2f}, max {r.max():.1f}")


def _undistort(params, distorted):
    """The pinhole radius whose distorted radius is `distorted`."""
    r = distorted
    for _ in range(30):
        r = distorted / (1 + params[9] * r * r + params[10] * r ** 4)
    return r


held = list(range(CAMERA, CAMERA + 3))          # point 1's place: the world's
pinhole = [i for i in range(len(first)) if i not in held + [9, 10]]
solved = least_squares(misses, first, steps, huber=1.0, free=pinhole, iterations=80)[0]
report("pinhole, every cube", solved)
with_k1 = least_squares(misses, solved, steps, huber=1.0, free=pinhole + [9], iterations=80)[0]
report("with k1", with_k1)
with_k2 = least_squares(misses, with_k1, steps, huber=1.0, free=pinhole + [9, 10], iterations=80)[0]
report("with k1, k2", with_k2)


# How far the saved camera (three cubes) and the five-cube pinhole put the same pixels apart on the desk, at the desk
# and at a cube's top: in reach and round about the arm's axis (camera.json's anchor), by bearing.
def pinhole_matrix(params):
    K5 = np.array([[params[0], 0, params[1]], [0, params[0], params[2]], [0, 0, 1.0]])
    return K5 @ np.column_stack((cv2.Rodrigues(params[3:6])[0], params[6:9]))


def to_plane(matrix, pixel, z):
    plane = np.column_stack((matrix[:, 0], matrix[:, 1], matrix[:, 3] + z * matrix[:, 2]))
    x, y, w = np.linalg.solve(plane, (*pixel, 1.0))
    return np.array([x / w, y / w])


P3, P5 = fit.true_to_pixel, pinhole_matrix(solved)
saved = store.load_camera(CONFIG)
axis = np.array(saved.anchor["axis"]) if saved.anchor else np.zeros(2)
axis_true = fit.desk_to_true @ axis
turn = math.radians(saved.anchor["turn_deg"]) if saved.anchor else 0.0
print("\nsaved camera (3 cubes) against the five-cube pinhole: the same pixel's desk point, mm, about the arm's axis")
for z in (0.0, CUBE_MM):
    print(f"  at height {z:.0f}:")
    for reach in (130, 190, 250):
        row = []
        for bearing in range(-60, 61, 30):
            b = math.radians(bearing)
            arm = np.array([reach * math.sin(b), -reach * math.cos(b)])
            c, s = math.cos(turn), math.sin(turn)
            desk = np.array([c * arm[0] - s * arm[1], s * arm[0] + c * arm[1]]) + axis
            true = fit.desk_to_true @ desk
            pixel = (P5 @ np.array([*true, z, 1.0]))
            pixel = pixel[:2] / pixel[2]
            back = to_plane(P3, pixel, z) - true                    # true mm: saved minus five-cube
            back_desk = np.linalg.solve(fit.desk_to_true, back)
            back_arm = np.array([c * back_desk[0] + s * back_desk[1], -s * back_desk[0] + c * back_desk[1]])
            out = back_arm @ (arm / reach)
            across = back_arm @ np.array([-arm[1], arm[0]]) / reach
            row.append(f"{bearing:+3d}: out {out:+.1f} round {across:+.1f}")
        print(f"    reach {reach}: " + "  ".join(row))
