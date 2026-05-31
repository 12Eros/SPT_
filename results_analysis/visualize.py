"""
Visualization module for tracking evaluation results.

- Generates the 3 summary plots (extremes bar, length-vs-IoU scatter, precision-success bubble)
- Automatically generates up to N high-quality qualitative comparison figures
  (RGB + Depth with green GT boxes and red prediction boxes) for the best-performing sequences.

Default: top 5 sequences by mean_iou → 5 beautiful multi-modal figures.

Dependencies (optional but recommended):
    pip install matplotlib pillow numpy

If not installed, plotting is gracefully skipped.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    HAS_VIZ_DEPS = True
except ImportError:
    HAS_VIZ_DEPS = False
    plt = None
    Image = None
    Rectangle = None

# ----------------------------- Helpers -----------------------------

def _ensure_plots_dir(out_dir: Path) -> Path:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    return plots


def _load_xywh_list(path: Path) -> List[List[float]]:
    """Reuse logic similar to loaders but return list."""
    import re
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.name.endswith("_001.txt"):
                # prediction: comma separated, may have leading stray number
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if len(parts) < 4:
                    continue
                try:
                    rows.append([float(parts[i]) for i in range(4)])
                except ValueError:
                    continue
            else:
                # GT: space or comma
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
                    xs, ys = vals[0::2], vals[1::2]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    rows.append([x1, y1, x2 - x1, y2 - y1])
                elif len(vals) >= 4:
                    rows.append(vals[:4])
    return rows


def _find_sequence_dir(dataset_root: Path, seq_name: str, expected_frames: Optional[int] = None) -> Optional[Path]:
    """
    Robustly locate the sequence directory for a given leaf name.
    Prefers folders that actually contain the expected number of GT lines.
    """
    root = Path(dataset_root)
    candidates = []
    for gt_name in ("groundtruth.txt", "groundtruth_rect.txt"):
        for p in root.rglob(gt_name):
            if p.parent.name.lower() == seq_name.lower():
                candidates.append(p.parent)

    if not candidates:
        # last resort: any folder whose name matches
        for p in root.rglob("color"):
            if p.parent.name.lower() == seq_name.lower():
                candidates.append(p.parent)

    if not candidates:
        return None

    if expected_frames is not None and len(candidates) > 1:
        best = None
        best_diff = 10**9
        for c in candidates:
            gt = c / "groundtruth.txt" if (c / "groundtruth.txt").is_file() else c / "groundtruth_rect.txt"
            if not gt.is_file():
                continue
            n = sum(1 for line in open(gt, encoding="utf-8", errors="replace") if line.strip())
            diff = abs(n - expected_frames)
            if diff < best_diff:
                best_diff = diff
                best = c
        if best is not None:
            return best

    # prefer the one under a "normal" category (not top-level same name)
    for c in candidates:
        rel = c.relative_to(root).as_posix()
        if "/" in rel and rel.split("/")[0].lower() != seq_name.lower():
            return c

    return candidates[0]


def _get_frame_path(seq_dir: Path, frame_idx_0based: int) -> Tuple[Path, Path]:
    """Return (color_path, depth_path) for 1-based 8-digit naming."""
    n = frame_idx_0based + 1
    stem = f"{n:08d}"
    color = seq_dir / "color" / f"{stem}.jpg"
    if not color.is_file():
        color = seq_dir / "color" / f"{stem}.jpeg"
    depth = seq_dir / "depth" / f"{stem}.png"
    return color, depth


def _draw_box(ax, box, color, linewidth=2, label=None):
    x, y, w, h = box
    rect = Rectangle((x, y), w, h, linewidth=linewidth, edgecolor=color, facecolor="none")
    ax.add_patch(rect)
    if label:
        ax.text(x, y - 5, label, color=color, fontsize=8, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))


# ----------------------------- Plot 1: Extremes horizontal bar -----------------------------

def plot_iou_extremes(per_seq: List[Dict[str, Any]], out_dir: Path, top_k: int = 5) -> Optional[Path]:
    if not HAS_VIZ_DEPS:
        print("[visualize] matplotlib/pillow not installed, skip plot_iou_extremes")
        return None

    sorted_rows = sorted(per_seq, key=lambda r: r["mean_iou"], reverse=True)
    top = sorted_rows[:top_k]
    bottom = sorted_rows[-top_k:][::-1]  # low to high for visual order

    labels = [r["sequence"] for r in top + bottom]
    values = [r["mean_iou"] for r in top + bottom]
    colors = ["#2E86AB"] * len(top) + ["#E63946"] * len(bottom)   # blue for high, red for low

    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(labels))))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean IoU")
    ax.set_title(f"Top-{top_k} vs Bottom-{top_k} Sequences by Mean IoU")
    ax.axvline(np.mean([r["mean_iou"] for r in per_seq]), color="gray", linestyle="--", alpha=0.7, label="overall mean")
    ax.legend(loc="lower right")

    for bar, v in zip(bars, values):
        ax.text(v + 0.003, bar.get_y() + bar.get_height()/2, f"{v:.3f}", va="center", fontsize=9)

    plots_dir = _ensure_plots_dir(out_dir)
    out_path = plots_dir / "01_iou_extremes.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ----------------------------- Plot 2: Length vs IoU scatter + trend -----------------------------

def plot_length_vs_iou(per_seq: List[Dict[str, Any]], out_dir: Path) -> Optional[Path]:
    if not HAS_VIZ_DEPS:
        print("[visualize] matplotlib/pillow not installed, skip plot_length_vs_iou")
        return None

    x = np.array([r["num_frames_gt"] for r in per_seq], dtype=float)
    y = np.array([r["mean_iou"] for r in per_seq], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, c="#4C72B0", s=35, alpha=0.7, edgecolors="white", linewidths=0.5, label="sequences")

    # Linear regression (degree 1)
    if len(x) >= 2:
        try:
            coeffs = np.polyfit(x, y, 1)
            trend = np.poly1d(coeffs)
            x_line = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_line, trend(x_line), color="#C44E52", linewidth=2.0, linestyle="--",
                    label=f"linear trend (slope={coeffs[0]:.2e})")
        except Exception:
            pass

    ax.set_xlim(400, 3200)
    ax.set_xlabel("Sequence length (num_frames_gt)")
    ax.set_ylabel("Mean IoU")
    ax.set_title("Sequence Length vs Tracking Performance (Mean IoU)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plots_dir = _ensure_plots_dir(out_dir)
    out_path = plots_dir / "02_length_vs_iou.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ----------------------------- Plot 3: Precision vs Success bubble -----------------------------

def plot_precision_success_bubble(per_seq: List[Dict[str, Any]], out_dir: Path) -> Optional[Path]:
    if not HAS_VIZ_DEPS:
        print("[visualize] matplotlib/pillow not installed, skip plot_precision_success_bubble")
        return None

    x = np.array([r["precision_20px"] for r in per_seq], dtype=float)
    y = np.array([r["mean_iou"] for r in per_seq], dtype=float)
    # bubble size proportional to center error (larger error = bigger bubble)
    sizes = np.array([r["mean_center_error_px"] for r in per_seq], dtype=float)
    sizes = 20 + (sizes / sizes.max()) * 800   # scale nicely

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(x, y, s=sizes, c=y, cmap="RdYlGn", alpha=0.65, edgecolors="black", linewidths=0.4)

    ax.set_xlabel("Precision@20px")
    ax.set_ylabel("Mean IoU (Success)")
    ax.set_title("Precision vs Success Rate (bubble size = Mean Center Error)")
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Mean IoU")

    plots_dir = _ensure_plots_dir(out_dir)
    out_path = plots_dir / "03_precision_vs_success_bubble.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ----------------------------- Qualitative comparison (multi-modal RGB + Depth with boxes) -----------------------------

def _get_evenly_spaced_frames(n: int, k: int = 3) -> List[int]:
    """Return k evenly spaced 0-based indices in [0, n-1]."""
    if n <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [0]
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def plot_qualitative_comparison(
    seq_name: str,
    dataset_root: Path,
    predictions_root: Path,
    per_seq_row: Optional[Dict[str, Any]],
    out_dir: Path,
    frames: Optional[Tuple[int, ...]] = None,
    rank: Optional[int] = None,
    title_prefix: Optional[str] = None,
    filename_prefix: str = "04_qualitative",
) -> Optional[Path]:
    """
    Draw RGB + Depth side-by-side for 3 (or custom) time points.
    Green = GT, Red = Prediction.
    Automatically chooses good frames if `frames` is None.
    Adds ranking and metrics to the title when available.

    title_prefix: e.g. "Best performing" or "Worst performing"
    filename_prefix: used for output file naming (e.g. "04_qualitative" or "05_worst_qualitative")
    """
    if not HAS_VIZ_DEPS:
        print("[visualize] matplotlib/pillow not installed, skip qualitative plot")
        return None

    seq_dir = _find_sequence_dir(dataset_root, seq_name,
                                  expected_frames=per_seq_row.get("num_frames_gt") if per_seq_row else None)
    if seq_dir is None:
        print(f"[visualize] Cannot locate sequence dir for {seq_name}")
        return None

    # Load GT boxes
    gt_file = seq_dir / "groundtruth.txt" if (seq_dir / "groundtruth.txt").is_file() else seq_dir / "groundtruth_rect.txt"
    gt_boxes = _load_xywh_list(gt_file)

    # Load prediction boxes
    pred_candidates = [
        predictions_root / seq_name / f"{seq_name}_001.txt",
        predictions_root / "rgbd-unsupervised" / seq_name / f"{seq_name}_001.txt",
    ]
    pred_path = None
    for c in pred_candidates:
        if c.is_file():
            pred_path = c
            break
    if pred_path is None:
        for p in predictions_root.rglob(f"**/{seq_name}_001.txt"):
            if "all_boxes" not in p.name:
                pred_path = p
                break
    if pred_path is None:
        print(f"[visualize] Cannot find prediction file for {seq_name}")
        return None

    pred_boxes = _load_xywh_list(pred_path)

    n = min(len(gt_boxes), len(pred_boxes))
    if n == 0:
        print("[visualize] No aligned frames for qualitative plot")
        return None

    # Auto-select 3 good frames if not provided
    if frames is None:
        frames = tuple(_get_evenly_spaced_frames(n, 3))

    chosen = [min(f, n-1) for f in frames]

    # Figure setup
    num_rows = len(chosen)
    fig_height = max(8, 3.2 * num_rows)
    fig = plt.figure(figsize=(14, fig_height))
    gs = fig.add_gridspec(num_rows, 2, hspace=0.28, wspace=0.06)

    # Title with extra info
    title_parts = []
    if title_prefix:
        title_parts.append(title_prefix)
    if rank is not None:
        title_parts.append(f"#{rank}")
    title_parts.append(seq_name)
    if per_seq_row:
        miou = per_seq_row.get("mean_iou", 0)
        title_parts.append(f"mean_IoU={miou:.4f}")

    fig.suptitle("Qualitative Tracking — " + "  |  ".join(title_parts) + "\n(Green=Ground Truth, Red=Prediction)", 
                 fontsize=13, fontweight="bold")

    for row_idx, t_idx in enumerate(chosen):
        color_path, depth_path = _get_frame_path(seq_dir, t_idx)
        gt_b = gt_boxes[t_idx] if t_idx < len(gt_boxes) else None
        pr_b = pred_boxes[t_idx] if t_idx < len(pred_boxes) else None

        frame_label = f"Frame {t_idx + 1} / {n}"

        # RGB
        ax_rgb = fig.add_subplot(gs[row_idx, 0])
        try:
            rgb = np.array(Image.open(color_path).convert("RGB"))
            ax_rgb.imshow(rgb)
        except Exception as e:
            ax_rgb.text(0.5, 0.5, f"Failed to load RGB\n{e}", ha="center", va="center")
        ax_rgb.set_title(f"{frame_label} - RGB", fontsize=10)
        ax_rgb.axis("off")
        if gt_b:
            _draw_box(ax_rgb, gt_b, "#00FF00", label="GT")
        if pr_b:
            _draw_box(ax_rgb, pr_b, "#FF0000", label="Pred")

        # Depth
        ax_d = fig.add_subplot(gs[row_idx, 1])
        try:
            depth = np.array(Image.open(depth_path))
            dmin, dmax = depth.min(), depth.max()
            depth_norm = (depth - dmin) / (dmax - dmin) if dmax > dmin else depth
            ax_d.imshow(depth_norm, cmap="plasma")
        except Exception as e:
            ax_d.text(0.5, 0.5, f"Failed to load Depth\n{e}", ha="center", va="center")
        ax_d.set_title(f"{frame_label} - Depth", fontsize=10)
        ax_d.axis("off")
        if gt_b:
            _draw_box(ax_d, gt_b, "#00FF00")
        if pr_b:
            _draw_box(ax_d, pr_b, "#FF0000")

    plots_dir = _ensure_plots_dir(out_dir)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in seq_name)

    if rank is not None:
        out_path = plots_dir / f"{filename_prefix}_{rank:02d}_{safe_name}.png"
    else:
        out_path = plots_dir / f"{filename_prefix}_{safe_name}.png"

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ----------------------------- Generate multiple high-quality qualitative plots -----------------------------

def generate_top_qualitative_plots(
    per_seq: List[Dict[str, Any]],
    dataset_root: Path,
    predictions_root: Path,
    out_dir: Path,
    top_k: int = 5,
) -> List[Path]:
    """
    Automatically select the top-K sequences with best mean_iou and generate
    beautiful qualitative (RGB + Depth) comparison figures for each.
    This is the recommended way to get 5 nice visualization examples.
    """
    if not HAS_VIZ_DEPS:
        print("[visualize] matplotlib/pillow/numpy not installed → skipping qualitative plots")
        return []

    if not per_seq:
        return []

    # Sort by mean_iou descending (best first)
    sorted_seqs = sorted(per_seq, key=lambda r: r.get("mean_iou", 0), reverse=True)
    top_seqs = sorted_seqs[:top_k]

    saved = []
    print(f"[visualize] Generating {len(top_seqs)} qualitative comparison figures for the best performing sequences...")

    for rank, row in enumerate(top_seqs, 1):
        seq_name = row["sequence"]
        try:
            p = plot_qualitative_comparison(
                seq_name=seq_name,
                dataset_root=dataset_root,
                predictions_root=predictions_root,
                per_seq_row=row,
                out_dir=out_dir,
                frames=None,   # auto evenly spaced
                rank=rank,
                title_prefix="Best performing",
                filename_prefix="04_best_qualitative"
            )
            if p:
                saved.append(p)
                print(f"  ✓ Saved rank #{rank}: {seq_name} (mean_iou={row.get('mean_iou', 0):.4f})")
        except Exception as e:
            print(f"  ✗ Failed for {seq_name}: {e}")

    return saved


def generate_bottom_qualitative_plots(
    per_seq: List[Dict[str, Any]],
    dataset_root: Path,
    predictions_root: Path,
    out_dir: Path,
    bottom_k: int = 5,
) -> List[Path]:
    """
    Automatically select the BOTTOM-K (worst) sequences by mean_iou and generate
    qualitative comparison figures for them.
    Useful to visually analyze failure cases (the red bars in the extremes plot).
    """
    if not HAS_VIZ_DEPS:
        print("[visualize] matplotlib/pillow/numpy not installed → skipping worst qualitative plots")
        return []

    if not per_seq:
        return []

    # Sort by mean_iou ascending (worst first)
    sorted_seqs = sorted(per_seq, key=lambda r: r.get("mean_iou", 0))
    worst_seqs = sorted_seqs[:bottom_k]

    saved = []
    print(f"[visualize] Generating {len(worst_seqs)} qualitative comparison figures for the WORST performing sequences (lowest mean_iou)...")

    for rank, row in enumerate(worst_seqs, 1):
        seq_name = row["sequence"]
        try:
            p = plot_qualitative_comparison(
                seq_name=seq_name,
                dataset_root=dataset_root,
                predictions_root=predictions_root,
                per_seq_row=row,
                out_dir=out_dir,
                frames=None,
                rank=rank,
                title_prefix="Worst performing",
                filename_prefix="05_worst_qualitative"
            )
            if p:
                saved.append(p)
                print(f"  ✓ Saved worst #{rank}: {seq_name} (mean_iou={row.get('mean_iou', 0):.4f})")
        except Exception as e:
            print(f"  ✗ Failed for {seq_name}: {e}")

    return saved


# ----------------------------- Main entry (updated) -----------------------------

def generate_all_visualizations(
    per_seq: List[Dict[str, Any]],
    dataset_root: Path,
    predictions_root: Path,
    out_dir: Path,
    top_qualitative_k: int = 5,
    bottom_qualitative_k: int = 5,
) -> List[Path]:
    """
    Generate:
    - 3 summary plots (extremes bar, length-vs-IoU, precision-success bubble)
    - Top-K best qualitative figures (best performing sequences)
    - Bottom-K worst qualitative figures (lowest mean_iou sequences, red bars in the chart)

    This gives you both success cases and clear failure cases for analysis.
    """
    if not HAS_VIZ_DEPS:
        warnings.warn(
            "Visualization dependencies missing. Run: pip install matplotlib pillow numpy",
            UserWarning
        )
        return []

    saved = []
    try:
        p1 = plot_iou_extremes(per_seq, out_dir)
        if p1:
            saved.append(p1)
    except Exception as e:
        print(f"[visualize] plot 1 failed: {e}")

    try:
        p2 = plot_length_vs_iou(per_seq, out_dir)
        if p2:
            saved.append(p2)
    except Exception as e:
        print(f"[visualize] plot 2 failed: {e}")

    try:
        p3 = plot_precision_success_bubble(per_seq, out_dir)
        if p3:
            saved.append(p3)
    except Exception as e:
        print(f"[visualize] plot 3 failed: {e}")

    # Best performing sequences (green / high IoU side of the bar chart)
    try:
        qual_saved = generate_top_qualitative_plots(
            per_seq=per_seq,
            dataset_root=dataset_root,
            predictions_root=predictions_root,
            out_dir=out_dir,
            top_k=top_qualitative_k,
        )
        saved.extend(qual_saved)
    except Exception as e:
        print(f"[visualize] best qualitative plots failed: {e}")

    # Worst performing sequences (red / low IoU side of the bar chart) - NEW
    try:
        worst_saved = generate_bottom_qualitative_plots(
            per_seq=per_seq,
            dataset_root=dataset_root,
            predictions_root=predictions_root,
            out_dir=out_dir,
            bottom_k=bottom_qualitative_k,
        )
        saved.extend(worst_saved)
    except Exception as e:
        print(f"[visualize] worst qualitative plots failed: {e}")

    if saved:
        print(f"[visualize] Saved {len(saved)} figures under {out_dir / 'plots'}")
    return saved
