"""
Evaluate tracking results against UniMod1K / RGBD1K-style annotations.

Usage:
    python analyze.py <dataset_root> <predictions_root>

Writes metrics under this package directory (results_analysis/).
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from . import loaders
    from . import metrics_core
    from . import gt_resolve
except ImportError:
    import loaders
    import metrics_core
    import gt_resolve


def _iou_thresholds():
    return [i / 20.0 for i in range(21)]


def evaluate_pair(gt_boxes, pr_boxes):
    """Align by min length; skip frames with invalid GT (w,h <= 0)."""
    t = min(len(gt_boxes), len(pr_boxes))
    ious, cles = [], []
    used = 0
    for i in range(t):
        g = gt_boxes[i]
        p = pr_boxes[i]
        if g[2] <= 0 or g[3] <= 0:
            continue
        ious.append(metrics_core.iou_xywh(g, p))
        cles.append(metrics_core.center_error_px(g, p))
        used += 1
    if used == 0:
        return None
    auc = metrics_core.success_auc(ious, _iou_thresholds())
    mean_iou = sum(ious) / used
    mean_cle = sum(cles) / used
    prec_20 = sum(1 for c in cles if c <= 20.0) / used
    return {
        "num_frames_gt": len(gt_boxes),
        "num_frames_pred": len(pr_boxes),
        "num_frames_eval": used,
        "mean_iou": mean_iou,
        "success_auc": auc,
        "mean_center_error_px": mean_cle,
        "precision_20px": prec_20,
        "iou_list": ious,
        "cle_list": cles,
    }


def main(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compare tracking *_001.txt boxes to dataset groundtruth_rect.txt / groundtruth.txt"
    )
    parser.add_argument("dataset_root", type=str, help="Root folder containing sequences (e.g. RGBD1K_train_labelled)")
    parser.add_argument("predictions_root", type=str, help="Root folder containing tracker output (nested *_001.txt)")
    args = parser.parse_args(argv)

    dataset_root = Path(args.dataset_root).resolve()
    pred_root = Path(args.predictions_root).resolve()

    if not dataset_root.is_dir():
        print(f"ERROR: dataset path is not a directory: {dataset_root}", file=sys.stderr)
        return 2
    if not pred_root.is_dir():
        print(f"ERROR: predictions path is not a directory: {pred_root}", file=sys.stderr)
        return 2

    seq_index = loaders.build_sequence_index(dataset_root)
    pred_files = loaders.discover_prediction_files(pred_root)

    if not pred_files:
        print(f"ERROR: no *_001.txt under {pred_root}", file=sys.stderr)
        return 2

    catalog = gt_resolve.build_gt_catalog(dataset_root)
    leaf_single = loaders.leaf_single_map_from_catalog(catalog)
    leaf_buckets = {}
    for rel, _ in catalog:
        lf = rel.split("/")[-1].lower()
        leaf_buckets.setdefault(lf, []).append(rel)

    alias_path = here / "sequence_alias.csv"
    aliases = gt_resolve.load_alias_csv(alias_path)

    per_seq = []
    missing = []
    warnings = []

    for pf in pred_files:
        try:
            rel_parent = pf.parent.relative_to(pred_root)
            pred_parent_relpath = rel_parent.as_posix()
        except ValueError:
            pred_parent_relpath = pf.parent.name

        seq_leaf = pf.parent.name
        if len(leaf_buckets.get(seq_leaf.lower(), [])) > 1:
            warnings.append(
                f"Dataset has multiple sequences named '{seq_leaf}' under different classes: "
                f"{leaf_buckets[seq_leaf.lower()][:5]}{'...' if len(leaf_buckets[seq_leaf.lower()]) > 5 else ''}"
            )

        gt_path = loaders.find_gt_file(
            dataset_root, pred_parent_relpath, seq_leaf, seq_index, leaf_single_map=leaf_single
        )
        match_method = "path_list_or_leaf_index"
        if gt_path is None:
            gt_path, m2 = gt_resolve.resolve_by_init_bbox(seq_leaf, pf, dataset_root, catalog, aliases)
            match_method = m2 or "unresolved"
        if gt_path is None:
            missing.append(str(pf.relative_to(pred_root)))
            continue

        if seq_index:
            hits = seq_index.get(seq_leaf.lower(), [])
            if len(hits) > 1 and match_method == "path_list_or_leaf_index":
                warnings.append(f"Multiple list.txt entries for '{seq_leaf}': {hits}; using {gt_path.parent}")

        if match_method and str(match_method).startswith("shadow"):
            warnings.append(f"{seq_leaf}: GT matched by heuristic ({match_method}) -> {gt_path.parent.name}")

        gt_boxes = loaders.load_gt_xywh(gt_path)
        pr_boxes = loaders.load_pred_xywh(pf)

        if len(gt_boxes) != len(pr_boxes):
            warnings.append(
                f"{seq_leaf}: length mismatch gt={len(gt_boxes)} pred={len(pr_boxes)}; evaluating first "
                f"{min(len(gt_boxes), len(pr_boxes))} aligned frames"
            )

        stats = evaluate_pair(gt_boxes, pr_boxes)
        if stats is None:
            missing.append(f"{pf.relative_to(pred_root)} (no valid GT frames)")
            continue

        gt_rel = str(gt_path.relative_to(dataset_root)).replace("\\", "/")
        category = gt_rel.split("/")[0] if "/" in gt_rel else gt_rel.split("\\")[0]

        row = {
            "sequence": seq_leaf,
            "category": category,
            "gt_file": gt_rel,
            "pred_file": str(pf.relative_to(pred_root)).replace("\\", "/"),
            "gt_match_method": match_method,
            **{k: v for k, v in stats.items() if k not in ("iou_list", "cle_list")},
        }
        per_seq.append(row)

    def mean_key(key):
        if not per_seq:
            return 0.0
        return sum(r[key] for r in per_seq) / len(per_seq)

    by_cat = defaultdict(list)
    for r in per_seq:
        by_cat[r["category"]].append(r)

    per_category_macro = {}
    for cat in sorted(by_cat.keys()):
        rows = by_cat[cat]
        n = len(rows)
        per_category_macro[cat] = {
            "num_sequences": n,
            "mean_iou": sum(x["mean_iou"] for x in rows) / n,
            "mean_success_auc": sum(x["success_auc"] for x in rows) / n,
            "mean_center_error_px": sum(x["mean_center_error_px"] for x in rows) / n,
            "mean_precision_20px": sum(x["precision_20px"] for x in rows) / n,
        }

    overall = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "predictions_root": str(pred_root),
        "num_prediction_files": len(pred_files),
        "num_sequences_evaluated": len(per_seq),
        "num_missing_or_invalid": len(missing),
        "mean_iou": mean_key("mean_iou"),
        "mean_success_auc": mean_key("success_auc"),
        "mean_center_error_px": mean_key("mean_center_error_px"),
        "mean_precision_20px": mean_key("precision_20px"),
        "per_category_macro": per_category_macro,
        "notes": {
            "missing_entries": "missing 列表表示该预测文件未能在数据集中关联到真值序列（常见原因：预测目录为短名如 kid1，而数据集中为 Human/human154 等）。不是单帧检测漏检。",
            "overall_mean": "mean_iou、mean_success_auc 等为：先对每条序列计算指标，再对所有已评估序列做算术平均（宏平均，每条序列权重相同）。",
            "per_category_macro": "per_category_macro 按真值路径的顶层大类（如 Human、Clothes）分组后，在组内对序列再做宏平均。",
            "shadow_match": "若 gt_match_method 以 shadow 开头，表示通过首帧框 IoU/L1 等启发式关联到 GT，若与您的测试划分不一致，可在 sequence_alias.csv 中手工指定 pred_folder -> dataset_relative_path 覆盖。",
        },
        "warnings": warnings,
        "missing": missing,
    }

    out_dir = here
    summary_path = out_dir / "summary.json"
    csv_path = out_dir / "per_sequence.csv"
    cat_csv_path = out_dir / "per_category.csv"
    log_path = out_dir / "run_log.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"overall": overall, "per_sequence": per_seq}, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "sequence",
        "category",
        "gt_file",
        "pred_file",
        "gt_match_method",
        "num_frames_gt",
        "num_frames_pred",
        "num_frames_eval",
        "mean_iou",
        "success_auc",
        "mean_center_error_px",
        "precision_20px",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in per_seq:
            w.writerow(r)

    cat_fields = ["category", "num_sequences", "mean_iou", "mean_success_auc", "mean_center_error_px", "mean_precision_20px"]
    with open(cat_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cat_fields)
        w.writeheader()
        for cat, stats in sorted(per_category_macro.items()):
            row = {"category": cat, **stats}
            w.writerow(row)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_root}\nPredictions: {pred_root}\n\n")
        f.write(overall["notes"]["missing_entries"] + "\n\n")
        if warnings:
            f.write("Warnings:\n")
            for w in warnings:
                f.write(f"  - {w}\n")
            f.write("\n")
        if missing:
            f.write("Missing / invalid:\n")
            for m in missing:
                f.write(f"  - {m}\n")

    print(json.dumps(overall, indent=2, ensure_ascii=False))
    print(f"\nWrote: {summary_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {cat_csv_path}")
    if missing:
        print(f"Wrote issues list: {log_path}")
    return 0 if per_seq else 1


if __name__ == "__main__":
    raise SystemExit(main())
