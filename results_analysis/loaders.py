"""Load GT / prediction trajectories and resolve dataset paths. No project imports."""

import os
import re
from pathlib import Path


def _read_list_txt(dataset_root):
    p = Path(dataset_root) / "list.txt"
    if not p.is_file():
        return None
    lines = []
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s.replace("\\", "/"))
    return lines


def build_sequence_index(dataset_root):
    """
    Map lowercase sequence leaf name -> list of relative paths 'Category/seq'
    (multiple entries are rare; caller may warn).
    """
    rels = _read_list_txt(dataset_root)
    index = {}
    if not rels:
        return index
    for rel in rels:
        rel = rel.strip("/")
        leaf = rel.split("/")[-1].lower()
        index.setdefault(leaf, []).append(rel)
    return index


def find_gt_file(dataset_root, pred_parent_relpath, seq_leaf, seq_index, leaf_single_map=None):
    """
    pred_parent_relpath: path from predictions root to folder containing *_001.txt,
    using '/' separators (may be '').

    leaf_single_map: optional dict leaf_lower -> single dataset relative path from a full scan.
    """
    root = Path(dataset_root)
    if pred_parent_relpath:
        for name in ("groundtruth_rect.txt", "groundtruth.txt"):
            cand = root / pred_parent_relpath / name
            if cand.is_file():
                return cand

    if leaf_single_map is not None:
        rel = leaf_single_map.get(seq_leaf.lower())
        if rel:
            for name in ("groundtruth_rect.txt", "groundtruth.txt"):
                cand = root / rel / name
                if cand.is_file():
                    return cand

    keys = []
    if seq_index is not None:
        keys = seq_index.get(seq_leaf.lower(), [])

    if len(keys) == 1:
        for name in ("groundtruth_rect.txt", "groundtruth.txt"):
            cand = root / keys[0] / name
            if cand.is_file():
                return cand

    for name in ("groundtruth_rect.txt", "groundtruth.txt"):
        matches = list(root.glob(f"**/{seq_leaf}/{name}"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # Prefer path that appears in list.txt order
            if keys:
                for k in keys:
                    for m in matches:
                        try:
                            if m.parent.samefile(root / k):
                                return m
                        except OSError:
                            pass
            return sorted(matches, key=lambda x: len(str(x)))[0]
    return None


def load_gt_xywh(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[\s,]+", line)
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
                xs = vals[0::2]
                ys = vals[1::2]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                rows.append([x1, y1, x2 - x1, y2 - y1])
            elif len(vals) >= 4:
                rows.append(vals[:4])
    return rows


def load_pred_xywh(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",") if p.strip() != ""]
            if len(parts) < 4:
                continue
            try:
                vals = [float(parts[i]) for i in range(4)]
            except ValueError:
                continue
            rows.append(vals)
    return rows


def discover_prediction_files(predictions_root):
    root = Path(predictions_root)
    out = []
    for p in root.rglob("*_001.txt"):
        if p.name.endswith("_001.txt") and "all_boxes" not in p.name:
            out.append(p)
    return sorted(out)


def leaf_single_map_from_catalog(catalog):
    """Map lowercase sequence folder name -> relative path when unique in dataset."""
    buckets = {}
    for rel, _ in catalog:
        leaf = rel.split("/")[-1].lower()
        buckets.setdefault(leaf, []).append(rel)
    return {leaf: paths[0] for leaf, paths in buckets.items() if len(paths) == 1}
