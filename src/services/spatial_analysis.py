"""
Spatial analysis utilities (Team 3): heatmaps, movement paths, and seat map rendering.
"""
from typing import Dict, List, Tuple
import numpy as np
import cv2


def generate_heatmap(
    positions_by_student: Dict[str, List[Tuple[int, int]]],
    frame_size: Tuple[int, int],
    blur_ksize: int = 31
) -> np.ndarray:
    h, w = frame_size
    acc = np.zeros((h, w), dtype=np.float32)
    for pts in positions_by_student.values():
        for (x, y) in pts:
            if 0 <= x < w and 0 <= y < h:
                # Add a small blob per position for better visibility
                cv2.circle(acc, (x, y), 8, 1.0, thickness=-1)
    if blur_ksize > 1:
        acc = cv2.GaussianBlur(acc, (blur_ksize, blur_ksize), 0)
    acc_norm = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(acc_norm, cv2.COLORMAP_JET)
    return heat


def render_movement_paths(
    positions_by_student: Dict[str, List[Tuple[int, int]]],
    frame_size: Tuple[int, int]
) -> np.ndarray:
    h, w = frame_size
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    # Assign deterministic colors per student id
    def color_for(sid: str) -> Tuple[int, int, int]:
        seed = sum(ord(c) for c in sid)
        np.random.seed(seed % 2**32)
        return tuple(int(v) for v in np.random.randint(0, 255, size=3))
    for sid, pts in positions_by_student.items():
        if len(pts) < 2:
            continue
        color = color_for(sid)
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i-1], pts[i], color, 2)
        # label last position
        cv2.putText(canvas, sid, pts[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return canvas


def render_seat_map(
    positions_by_student: Dict[str, List[Tuple[int, int]]],
    frame_size: Tuple[int, int]
) -> np.ndarray:
    h, w = frame_size
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    for sid, pts in positions_by_student.items():
        if not pts:
            continue
        # approximate seat as median position
        xs = sorted(p[0] for p in pts)
        ys = sorted(p[1] for p in pts)
        cx = xs[len(xs)//2]
        cy = ys[len(ys)//2]
        cv2.circle(canvas, (cx, cy), 10, (0, 128, 255), thickness=-1)
        cv2.putText(canvas, sid, (cx+12, cy-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return canvas


