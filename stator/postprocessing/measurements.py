"""
Geometric measurement functions for segmentation contours.

Functions
---------
normalize_measurement_family          : normalise class-name strings to a family key
compute_contour_measurements          : bounding rect, diameter, area, perimeter, pairwise distances
compute_single_contour_measurements   : the same metrics for a single contour
compute_edge_center_points            : left/right/top/bottom/center key-points
compute_edge_center_distances         : edge-to-edge and center-to-center distances between detections
compute_aligned_same_class_distances  : center distances for aligned same-family pairs (blocker-aware)
compute_cross_diametric_opposite_distances : cross-diametric paired distances (magnets & mech parts)
"""

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Family normalisation
# ---------------------------------------------------------------------------

def normalize_measurement_family(name: object) -> str:
    """Normalise a class-name string to a canonical measurement family key."""
    n = str(name or "").strip().lower().replace(" ", "_").replace("-", "_")
    if n in {
        "mechanical_part", "mechanicalpart", "mechanical",
        "michanical_part", "michanicalpart", "michanical",
        "mecanical_part", "mecanicalpart", "mecanical",
    }:
        return "mechanical_part"
    if n == "magnet":
        return "magnet"
    if n == "circle":
        return "circle"
    return n


# ---------------------------------------------------------------------------
# Multi-contour measurements
# ---------------------------------------------------------------------------

def compute_contour_measurements(
    contours: List[np.ndarray],
    mask: np.ndarray,
    px_to_mm: float,
) -> List[dict]:
    """
    Compute geometric measurements on a list of contours.

    Returns a list of measurement dicts, each with:
        type, value_px, value_mm, unit, contour_idx, points (for drawing)
    """
    measurements: List[dict] = []

    for i, cnt in enumerate(contours):
        if len(cnt) < 5:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        measurements.append({
            "type": "bounding_width",
            "label": f"Width #{i}",
            "value_px": round(float(w), 1),
            "value_mm": round(float(w) * px_to_mm, 2),
            "contour_idx": i,
            "pt1": [int(x), int(y + h // 2)],
            "pt2": [int(x + w), int(y + h // 2)],
        })
        measurements.append({
            "type": "bounding_height",
            "label": f"Height #{i}",
            "value_px": round(float(h), 1),
            "value_mm": round(float(h) * px_to_mm, 2),
            "contour_idx": i,
            "pt1": [int(x + w // 2), int(y)],
            "pt2": [int(x + w // 2), int(y + h)],
        })

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        diameter_px = 2 * radius
        measurements.append({
            "type": "outer_diameter",
            "label": f"Outer Diam #{i}",
            "value_px": round(float(diameter_px), 1),
            "value_mm": round(float(diameter_px) * px_to_mm, 2),
            "contour_idx": i,
            "center": [int(cx), int(cy)],
            "radius": int(radius),
        })

        area_px = cv2.contourArea(cnt)
        measurements.append({
            "type": "area",
            "label": f"Area #{i}",
            "value_px": round(float(area_px), 1),
            "value_mm": round(float(area_px) * (px_to_mm ** 2), 2),
            "contour_idx": i,
        })

        perimeter_px = cv2.arcLength(cnt, True)
        measurements.append({
            "type": "perimeter",
            "label": f"Perimeter #{i}",
            "value_px": round(float(perimeter_px), 1),
            "value_mm": round(float(perimeter_px) * px_to_mm, 2),
            "contour_idx": i,
        })

    # Pairwise minimum distances
    for i in range(len(contours)):
        for j in range(i + 1, len(contours)):
            if len(contours[i]) < 2 or len(contours[j]) < 2:
                continue
            pts_a = contours[i].reshape(-1, 2).astype(np.float64)
            pts_b = contours[j].reshape(-1, 2).astype(np.float64)
            dists = cdist(pts_a, pts_b)
            min_dist = float(np.min(dists))
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            pt_a = pts_a[idx[0]].astype(int).tolist()
            pt_b = pts_b[idx[1]].astype(int).tolist()
            measurements.append({
                "type": "min_distance",
                "label": f"Dist #{i}↔#{j}",
                "value_px": round(min_dist, 1),
                "value_mm": round(min_dist * px_to_mm, 2),
                "contour_idx": [i, j],
                "pt1": pt_a,
                "pt2": pt_b,
            })

    return measurements


# ---------------------------------------------------------------------------
# Single-contour measurements
# ---------------------------------------------------------------------------

def compute_single_contour_measurements(
    contour: np.ndarray,
    px_to_mm: float,
    class_name: str = "",
) -> List[dict]:
    """
    Compute geometric measurements for a single detection contour.

    Returns a list of measurement dicts each with:
        type, label, value_px, value_mm, and optional drawing points.
    """
    measurements: List[dict] = []
    if contour is None or len(contour) < 5:
        return measurements

    prefix = f"{class_name} " if class_name else ""

    x, y, w, h = cv2.boundingRect(contour)
    measurements.append({
        "type": "width",
        "label": f"{prefix}Width",
        "value_px": round(float(w), 1),
        "value_mm": round(float(w) * px_to_mm, 2),
        "pt1": [int(x), int(y + h // 2)],
        "pt2": [int(x + w), int(y + h // 2)],
    })
    measurements.append({
        "type": "height",
        "label": f"{prefix}Height",
        "value_px": round(float(h), 1),
        "value_mm": round(float(h) * px_to_mm, 2),
        "pt1": [int(x + w // 2), int(y)],
        "pt2": [int(x + w // 2), int(y + h)],
    })

    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    diameter_px = 2 * radius
    measurements.append({
        "type": "diameter",
        "label": f"{prefix}Diameter",
        "value_px": round(float(diameter_px), 1),
        "value_mm": round(float(diameter_px) * px_to_mm, 2),
        "center": [int(cx), int(cy)],
        "radius": int(radius),
    })

    area_px = cv2.contourArea(contour)
    measurements.append({
        "type": "area",
        "label": f"{prefix}Area",
        "value_px": round(float(area_px), 1),
        "value_mm": round(float(area_px) * (px_to_mm ** 2), 2),
    })

    perimeter_px = cv2.arcLength(contour, True)
    measurements.append({
        "type": "perimeter",
        "label": f"{prefix}Perimeter",
        "value_px": round(float(perimeter_px), 1),
        "value_mm": round(float(perimeter_px) * px_to_mm, 2),
    })

    return measurements


# ---------------------------------------------------------------------------
# Key-point extraction
# ---------------------------------------------------------------------------

def compute_edge_center_points(contour: np.ndarray) -> Dict[str, List[int]]:
    """Return edge midpoint and center key-points for a contour."""
    x, y, w, h = cv2.boundingRect(contour)
    center_x = int(x + w / 2)
    center_y = int(y + h / 2)

    m = cv2.moments(contour)
    if abs(m.get("m00", 0.0)) > 1e-6:
        center_x = int(m["m10"] / m["m00"])
        center_y = int(m["m01"] / m["m00"])

    return {
        "left":   [int(x),         int(y + h / 2)],
        "right":  [int(x + w),     int(y + h / 2)],
        "top":    [int(x + w / 2), int(y)],
        "bottom": [int(x + w / 2), int(y + h)],
        "center": [center_x,       center_y],
    }


# ---------------------------------------------------------------------------
# Inter-detection distances
# ---------------------------------------------------------------------------

def compute_edge_center_distances(
    entries: List[Dict[str, object]],
    px_to_mm: float,
    show_edges: bool = True,
    show_centers: bool = True,
) -> List[dict]:
    """
    Compute edge-to-edge and center-to-center distances between all entry pairs.

    Each entry must provide:
        name   : str
        points : Dict[str, List[int]]  (left/right/top/bottom/center)
        det_idx: int
    """
    measurements: List[dict] = []
    edge_keys = ("left", "right", "top", "bottom")

    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            a = entries[i]
            b = entries[j]
            name_a = str(a.get("name", ""))
            name_b = str(b.get("name", ""))
            pts_a = a.get("points", {})
            pts_b = b.get("points", {})

            if show_centers:
                pt_a = pts_a.get("center")
                pt_b = pts_b.get("center")
                if pt_a and pt_b:
                    dist_px = math.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1])
                    measurements.append({
                        "type": "center_distance",
                        "label": f"Center {name_a} ↔ {name_b}",
                        "value_px": round(dist_px, 1),
                        "value_mm": round(dist_px * px_to_mm, 2),
                        "pt1": pt_a,
                        "pt2": pt_b,
                        "det_indices": [a.get("det_idx"), b.get("det_idx")],
                    })

            if show_edges:
                for ka in edge_keys:
                    for kb in edge_keys:
                        p1 = pts_a.get(ka)
                        p2 = pts_b.get(kb)
                        if not p1 or not p2:
                            continue
                        dist_px = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                        measurements.append({
                            "type": "edge_distance",
                            "label": f"Edge {name_a}:{ka} ↔ {name_b}:{kb}",
                            "value_px": round(dist_px, 1),
                            "value_mm": round(dist_px * px_to_mm, 2),
                            "pt1": p1,
                            "pt2": p2,
                            "det_indices": [a.get("det_idx"), b.get("det_idx")],
                        })

    return measurements


def compute_aligned_same_class_distances(
    entries: List[Dict[str, object]],
    px_to_mm: float,
    alignment_ratio: float = 0.4,
) -> List[dict]:
    """
    Compute center distances for aligned same-family pairs with blocker filtering.

    Only pairs that are vertically or horizontally aligned (within *alignment_ratio*
    of their span) are considered. A pair is skipped when a member of the opposite
    family sits between the two centers (blocker rule).
    """
    measurements: List[dict] = []

    def _family(name: object) -> Optional[str]:
        n = normalize_measurement_family(name)
        return n if n in {"mechanical_part", "magnet"} else None

    norm = []
    for e in entries:
        family = _family(e.get("name"))
        center = (e.get("points") or {}).get("center")
        if family is None or not center:
            continue
        norm.append({"family": family, "center": center, "det_idx": e.get("det_idx")})

    for i in range(len(norm)):
        for j in range(i + 1, len(norm)):
            a = norm[i]
            b = norm[j]

            if a["family"] != b["family"]:
                continue

            pt_a = a["center"]
            pt_b = b["center"]
            dx = abs(pt_a[0] - pt_b[0])
            dy = abs(pt_a[1] - pt_b[1])
            span = max(dx, dy)
            if span < 1:
                continue

            align_tol = max(8.0, float(span) * alignment_ratio)
            line_tol  = max(6.0, align_tol * 0.75)

            vertical_ok   = dx <= align_tol
            horizontal_ok = dy <= align_tol
            if not vertical_ok and not horizontal_ok:
                continue

            if vertical_ok and (not horizontal_ok or dx <= dy):
                orientation = "vertical"
            else:
                orientation = "horizontal"

            opposite_family = "magnet" if a["family"] == "mechanical_part" else "mechanical_part"
            blocked = False

            if orientation == "vertical":
                x_line = (pt_a[0] + pt_b[0]) / 2.0
                y_min, y_max = sorted((pt_a[1], pt_b[1]))
                for k, c in enumerate(norm):
                    if k in (i, j) or c["family"] != opposite_family:
                        continue
                    cx, cy = c["center"]
                    if abs(cx - x_line) <= line_tol and y_min <= cy <= y_max:
                        blocked = True
                        break
            else:
                y_line = (pt_a[1] + pt_b[1]) / 2.0
                x_min, x_max = sorted((pt_a[0], pt_b[0]))
                for k, c in enumerate(norm):
                    if k in (i, j) or c["family"] != opposite_family:
                        continue
                    cx, cy = c["center"]
                    if abs(cy - y_line) <= line_tol and x_min <= cx <= x_max:
                        blocked = True
                        break

            if blocked:
                continue

            dist_px = math.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1])
            measurements.append({
                "type": "aligned_pair_distance",
                "label": f"Aligned {a['family']} ({orientation})",
                "value_px": round(dist_px, 1),
                "value_mm": round(dist_px * px_to_mm, 2),
                "pt1": pt_a,
                "pt2": pt_b,
                "det_indices": [a.get("det_idx"), b.get("det_idx")],
            })

    return measurements


def compute_cross_diametric_opposite_distances(
    entries: List[Dict[str, object]],
    px_to_mm: float,
    circle_center: Optional[Tuple[int, int]] = None,
) -> List[dict]:
    """
    Compute cross-diametric distances between opposite same-family detections.

    For both magnets and mechanical parts two diagonal pairs are measured:
        - BL↔TR  (diag-asc)
        - BR↔TL  (diag-desc)
    """
    measurements: List[dict] = []

    def _family(name: object) -> Optional[str]:
        n = normalize_measurement_family(name)
        return n if n in {"mechanical_part", "magnet"} else None

    norm = []
    uid = 0
    for e in entries:
        family = _family(e.get("name"))
        center = (e.get("points") or {}).get("center")
        if family is None or not center:
            continue
        norm.append({"uid": uid, "family": family, "center": center, "det_idx": e.get("det_idx")})
        uid += 1

    if not norm:
        return []

    base_cx, base_cy = 0, 0
    if circle_center:
        base_cx, base_cy = circle_center
    else:
        base_cx = int(sum(e["center"][0] for e in norm) / len(norm))
        base_cy = int(sum(e["center"][1] for e in norm) / len(norm))

    families = {
        "mechanical_part": [e for e in norm if e["family"] == "mechanical_part"],
        "magnet":          [e for e in norm if e["family"] == "magnet"],
    }

    for family, fam_entries in families.items():
        if len(fam_entries) < 2:
            continue

        if circle_center:
            ccx, ccy = base_cx, base_cy
        else:
            ccx = int(sum(e["center"][0] for e in fam_entries) / len(fam_entries))
            ccy = int(sum(e["center"][1] for e in fam_entries) / len(fam_entries))

        quads: Dict[str, list] = {"tl": [], "tr": [], "bl": [], "br": []}
        for e in fam_entries:
            x, y = e["center"]
            if   x <= ccx and y <= ccy: quads["tl"].append(e)
            elif x >  ccx and y <= ccy: quads["tr"].append(e)
            elif x <= ccx and y >  ccy: quads["bl"].append(e)
            else:                        quads["br"].append(e)

        def _rep(candidates: list):
            if not candidates:
                return None
            return max(candidates, key=lambda e: (e["center"][0] - ccx) ** 2 + (e["center"][1] - ccy) ** 2)

        tl = _rep(quads["tl"])
        tr = _rep(quads["tr"])
        bl = _rep(quads["bl"])
        br = _rep(quads["br"])

        candidate_pairs = []

        if family == "mechanical_part":
            if len(fam_entries) >= 4:
                by_y        = sorted(fam_entries, key=lambda e: (e["center"][1], e["center"][0]))
                top_two     = by_y[:2]
                bottom_two  = by_y[-2:]
                top_left    = min(top_two,    key=lambda e: e["center"][0])
                top_right   = max(top_two,    key=lambda e: e["center"][0])
                bottom_left = min(bottom_two, key=lambda e: e["center"][0])
                bottom_right= max(bottom_two, key=lambda e: e["center"][0])
                candidate_pairs.append((bottom_left,  top_right,  "diag-asc (BL-TR)"))
                candidate_pairs.append((bottom_right, top_left,   "diag-desc (BR-TL)"))
            else:
                if bl is not None and tr is not None:
                    candidate_pairs.append((bl, tr, "diag-asc (BL-TR)"))
                if br is not None and tl is not None:
                    candidate_pairs.append((br, tl, "diag-desc (BR-TL)"))
        else:
            if tr is not None and bl is not None:
                candidate_pairs.append((tr, bl, "diag-asc (TR-BL)"))
            if tl is not None and br is not None:
                candidate_pairs.append((tl, br, "diag-desc (TL-BR)"))

        # Angular fallback when strict pairing yields too few lines
        target_lines = 2
        if len(candidate_pairs) < target_lines and len(fam_entries) >= 2:
            sorted_by_angle = []
            for e in fam_entries:
                dx_a, dy_a = e["center"][0] - ccx, e["center"][1] - ccy
                sorted_by_angle.append((math.atan2(dy_a, dx_a), e))
            sorted_by_angle.sort(key=lambda x: x[0])

            n = len(fam_entries)
            temp = []
            for i in range(n):
                for j in range(i + 1, n):
                    diff = abs(sorted_by_angle[i][0] - sorted_by_angle[j][0])
                    if diff > math.pi:
                        diff = 2 * math.pi - diff
                    temp.append((abs(math.pi - diff), sorted_by_angle[i][1], sorted_by_angle[j][1]))
            temp.sort(key=lambda x: x[0])

            used_ids: set = set()
            new_candidates = []
            for score, a_e, b_e in temp:
                if a_e["uid"] not in used_ids and b_e["uid"] not in used_ids:
                    dx_p = b_e["center"][0] - a_e["center"][0]
                    dy_p = b_e["center"][1] - a_e["center"][1]
                    label = "diag-desc" if dx_p * dy_p > 0 else "diag-asc"
                    new_candidates.append((a_e, b_e, label))
                    used_ids.add(a_e["uid"])
                    used_ids.add(b_e["uid"])
                if len(new_candidates) >= target_lines:
                    break

            if family == "mechanical_part" and len(candidate_pairs) >= 1:
                pass
            elif len(new_candidates) >= len(candidate_pairs):
                candidate_pairs = new_candidates

        for a_e, b_e, orientation in candidate_pairs:
            dx_p = b_e["center"][0] - a_e["center"][0]
            dy_p = b_e["center"][1] - a_e["center"][1]
            dist_px = math.hypot(dx_p, dy_p)
            measurements.append({
                "type": "cross_diametric_opposite_distance",
                "label": f"Opposite {family} ({orientation})",
                "value_px": round(dist_px, 1),
                "value_mm": round(dist_px * px_to_mm, 2),
                "pt1": a_e["center"],
                "pt2": b_e["center"],
                "det_indices": [a_e.get("det_idx"), b_e.get("det_idx")],
            })

    return measurements
