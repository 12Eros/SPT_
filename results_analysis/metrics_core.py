"""Bounding-box metrics (x1, y1, w, h). No project imports."""


def xywh_to_corners(b):
    x1, y1, w, h = b[0], b[1], b[2], b[3]
    return x1, y1, x1 + w, y1 + h


def iou_xywh(a, b):
    ax1, ay1, ax2, ay2 = xywh_to_corners(a)
    bx1, by1, bx2, by2 = xywh_to_corners(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def center_xywh(b):
    x1, y1, w, h = b[0], b[1], b[2], b[3]
    return x1 + 0.5 * w, y1 + 0.5 * h


def center_error_px(a, b):
    ca = center_xywh(a)
    cb = center_xywh(b)
    dx, dy = ca[0] - cb[0], ca[1] - cb[1]
    return (dx * dx + dy * dy) ** 0.5


def success_auc(iou_per_frame, thresholds):
    """Mean over thresholds of fraction of frames with IoU >= t."""
    if not iou_per_frame:
        return 0.0
    scores = []
    n = len(iou_per_frame)
    for t in thresholds:
        scores.append(sum(1 for i in iou_per_frame if i >= t) / n)
    return sum(scores) / len(scores)


def precision_at(iou_per_frame, cle_per_frame, thr_px, use_iou=False):
    if not iou_per_frame:
        return 0.0
    n = len(iou_per_frame)
    if use_iou:
        return sum(1 for i in iou_per_frame if i >= thr_px) / n
    return sum(1 for c in cle_per_frame if c <= thr_px) / n
