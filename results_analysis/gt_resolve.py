"""Resolve prediction folder names to GT paths when names differ from dataset layout (e.g. kid1 -> Human/human154)."""

import csv
import re
from pathlib import Path


def build_gt_catalog(dataset_root):
    """(rel_path_posix, gt_file_path) for every sequence with annotations."""
    root = Path(dataset_root)
    by_rel = {}
    for p in root.rglob("groundtruth_rect.txt"):
        rel = p.parent.relative_to(root).as_posix()
        by_rel[rel] = p
    for p in root.rglob("groundtruth.txt"):
        rel = p.parent.relative_to(root).as_posix()
        if rel not in by_rel:
            by_rel[rel] = p
    return sorted(by_rel.items())


def _first_gt_xywh_line(gt_path):
    import re as _re

    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = _re.split(r"[\s,]+", line)
            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except ValueError:
                    vals = []
                    break
            if not vals:
                continue
            if len(vals) >= 8:
                xs, ys = vals[0::2], vals[1::2]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                return [x1, y1, x2 - x1, y2 - y1]
            if len(vals) >= 4:
                return vals[:4]
    return None


def _first_pred_xywh_line(pred_path):
    with open(pred_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",") if p.strip() != ""]
            if len(parts) < 4:
                continue
            try:
                return [float(parts[i]) for i in range(4)]
            except ValueError:
                continue
    return None


def _iou_xywh(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _l1(a, b):
    return sum(abs(float(a[i]) - float(b[i])) for i in range(4))


def _pool_for_pred_leaf(pred_leaf):
    pl = pred_leaf.lower()
    if re.match(r"^(kid|girl|boy|man)\d+$", pl):
        return "human"
    if pl.startswith("basketball"):
        return "ball"
    if pl.startswith("cola"):
        return "bottle"
    return "all"


def _filter_catalog(catalog, pool):
    if pool == "all":
        return list(catalog)
    prefix = {"human": "Human/", "ball": "Ball/", "bottle": "Bottle/"}[pool]
    sub = [(rel, p) for rel, p in catalog if rel.startswith(prefix)]
    return sub if sub else list(catalog)


def load_alias_csv(csv_path):
    """pred_folder (any case) -> dataset relative path."""
    if not csv_path.is_file():
        return {}
    out = {}
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = (row.get("pred_folder") or row.get("pred") or "").strip()
            v = (row.get("dataset_relative_path") or row.get("gt_path") or "").strip().replace("\\", "/")
            if not k or not v:
                continue
            out[k.lower()] = v
    return out


def resolve_by_init_bbox(pred_leaf, pred_path, dataset_root, catalog, aliases):
    """
    When direct path / list.txt / leaf index fail, map short eval names to GT.

    Uses first-frame IoU and L1 on xywh; restricts search pool for kid/girl/man/boy/basketball/cola.
    """
    root = Path(dataset_root)
    key = pred_leaf.lower()
    if key in aliases:
        rel = aliases[key]
        for name in ("groundtruth_rect.txt", "groundtruth.txt"):
            cand = root / rel / name
            if cand.is_file():
                return cand, "sequence_alias.csv"

    p0 = _first_pred_xywh_line(pred_path)
    if p0 is None:
        return None, None

    pool = _pool_for_pred_leaf(pred_leaf)
    candidates = _filter_catalog(catalog, pool)

    scored = []
    for rel, gtpath in candidates:
        g0 = _first_gt_xywh_line(gtpath)
        if g0 is None:
            continue
        iou = _iou_xywh(p0, g0)
        l1 = _l1(p0, g0)
        scored.append((iou, l1, rel, gtpath))

    if not scored:
        return None, None

    scored.sort(key=lambda x: (-x[0], x[1]))
    if pool in ("ball", "bottle") and scored[0][0] < 0.05:
        if pool == "bottle":
            bottle_only = _filter_catalog(catalog, "bottle")
            l1cand = []
            for rel, gtpath in bottle_only:
                g0 = _first_gt_xywh_line(gtpath)
                if g0 is None:
                    continue
                l1cand.append((_l1(p0, g0), rel, gtpath))
            if l1cand:
                l1cand.sort(key=lambda x: x[0])
                return l1cand[0][2], "shadow_init_bbox_bottle_l1"
        scored = []
        if pool == "ball":
            prefixes = ("Ball/", "Human/")
        else:
            prefixes = (
                "Ball/",
                "Bottle/",
                "Cup/",
                "Dish/",
                "Kettle/",
                "Pot/",
                "Can/",
                "Box/",
                "Ipad/",
                "Cell_phone/",
            )
        for rel, gtpath in catalog:
            if not rel.startswith(prefixes):
                continue
            g0 = _first_gt_xywh_line(gtpath)
            if g0 is None:
                continue
            iou = _iou_xywh(p0, g0)
            l1 = _l1(p0, g0)
            scored.append((iou, l1, rel, gtpath))
        scored.sort(key=lambda x: (-x[0], x[1]))
        if not scored:
            return None, None

    best_iou, best_l1, best_rel, best_path = scored[0]
    second_iou = scored[1][0] if len(scored) > 1 else -1.0
    margin = best_iou - second_iou

    if best_iou >= 0.35 and margin >= 0.05:
        return best_path, "shadow_init_bbox"
    if margin >= 0.12:
        return best_path, "shadow_init_bbox"
    if margin >= 0.045 and best_iou >= 0.07:
        return best_path, "shadow_init_bbox"
    if margin >= 0.03 and best_iou >= 0.25:
        return best_path, "shadow_init_bbox"
    if margin < 0.03 and best_iou >= 0.2 and pool != "human":
        return best_path, "shadow_init_bbox"

    top_k = min(8, len(scored))
    tie = scored[:top_k]
    best_l1_row = min(tie, key=lambda x: x[1])
    if pool == "human":
        if best_l1_row[0] >= 0.12:
            return best_l1_row[3], "shadow_init_bbox_l1_tiebreak"
        if best_l1_row[0] >= 0.08 and best_l1_row[1] <= 200.0:
            return best_l1_row[3], "shadow_init_bbox_l1_tiebreak"
        if best_l1_row[0] >= 0.05 and margin >= 0.04:
            return best_path, "shadow_init_bbox"

    return None, None
