"""Discover sequences and subsample frames while preserving on-disk layout."""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path


def pick_gt_path(seq_dir: Path) -> Path | None:
    for name in ("groundtruth.txt", "groundtruth_rect.txt"):
        p = seq_dir / name
        if p.is_file():
            return p
    return None


def read_gt_lines(gt_path: Path) -> list[str]:
    lines = []
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.rstrip("\n\r")
            if s.strip():
                lines.append(s)
    return lines


def sequence_frame_count(seq_dir: Path, gt_path: Path) -> int:
    """Align GT rows with color/depth 8-digit 1-based filenames."""
    lines = read_gt_lines(gt_path)
    t = len(lines)
    while t > 0:
        jpg = seq_dir / "color" / f"{t:08d}.jpg"
        if not jpg.is_file():
            jpg = seq_dir / "color" / f"{t:08d}.jpeg"
        png = seq_dir / "depth" / f"{t:08d}.png"
        if jpg.is_file() and png.is_file():
            return t
        t -= 1
    return 0


def is_valid_sequence(seq_dir: Path) -> bool:
    if not seq_dir.is_dir():
        return False
    gt = pick_gt_path(seq_dir)
    if gt is None:
        return False
    cdir = seq_dir / "color"
    ddir = seq_dir / "depth"
    if not cdir.is_dir() or not ddir.is_dir():
        return False
    if not (seq_dir / "depth" / "00000001.png").is_file():
        return False
    if (seq_dir / "color" / "00000001.jpg").is_file():
        return True
    if (seq_dir / "color" / "00000001.jpeg").is_file():
        return True
    return False


def list_minor_classes(major_dir: Path) -> list[Path]:
    out = []
    for p in sorted(major_dir.iterdir()):
        if p.is_dir() and is_valid_sequence(p):
            out.append(p)
    return out


def list_major_categories(source_root: Path) -> list[Path]:
    majors = []
    for p in sorted(source_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.lower() in ("__pycache__", ".git"):
            continue
        if list_minor_classes(p):
            majors.append(p)
    return majors


def frame_index_to_names(orig_idx_zero_based: int) -> tuple[str, str]:
    n = orig_idx_zero_based + 1
    stem = f"{n:08d}"
    return stem + ".jpg", stem + ".png"


def _resolve_color_src(seq_dir: Path, old_i: int) -> Path:
    stem = f"{old_i+1:08d}"
    for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
        p = seq_dir / "color" / (stem + ext)
        if p.is_file():
            return p
    raise FileNotFoundError(f"Missing color frame {stem} under {seq_dir}")


def subsample_sequence(
    seq_src: Path,
    seq_dst: Path,
    rng: random.Random,
) -> dict:
    """
    Copy a random 100–200 frame subset (or fewer if source shorter), renumbered 1..K.
    Writes groundtruth.txt; copies groundtruth_rect.txt subset if present; copies nlp.txt.
    """
    gt_primary = seq_src / "groundtruth.txt"
    gt_rect = seq_src / "groundtruth_rect.txt"
    gt_read = gt_primary if gt_primary.is_file() else gt_rect
    if not gt_read.is_file():
        raise FileNotFoundError(f"No groundtruth in {seq_src}")

    lines = read_gt_lines(gt_read)
    t = sequence_frame_count(seq_src, gt_read)
    if t < 1 or len(lines) < t:
        t = min(t, len(lines))
    if t < 1:
        raise ValueError(f"Empty sequence: {seq_src}")

    k_req = rng.randint(100, 200)
    k = min(k_req, t)
    indices = sorted(rng.sample(range(t), k))

    seq_dst.mkdir(parents=True, exist_ok=True)
    (seq_dst / "color").mkdir(exist_ok=True)
    (seq_dst / "depth").mkdir(exist_ok=True)

    for new_i, old_i in enumerate(indices):
        jpg_name, png_name = frame_index_to_names(new_i)
        old_jpg = _resolve_color_src(seq_src, old_i)
        old_png = seq_src / "depth" / f"{old_i+1:08d}.png"
        shutil.copy2(old_jpg, seq_dst / "color" / jpg_name)
        shutil.copy2(old_png, seq_dst / "depth" / png_name)

    new_lines = [lines[i] for i in indices]
    with open(seq_dst / "groundtruth.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(new_lines) + "\n")

    if gt_rect.is_file():
        rect_lines = read_gt_lines(gt_rect)
        if len(rect_lines) >= t:
            new_rect = [rect_lines[i] for i in indices]
            with open(seq_dst / "groundtruth_rect.txt", "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(new_rect) + "\n")

    nlp_src = seq_src / "nlp.txt"
    if nlp_src.is_file():
        shutil.copy2(nlp_src, seq_dst / "nlp.txt")

    return {
        "source": str(seq_src),
        "frames_source": t,
        "frames_requested": k_req,
        "frames_written": k,
        "indices_min": min(indices),
        "indices_max": max(indices),
    }


def run_split(
    source_root: Path,
    output_root: Path,
    rng: random.Random,
    max_minors_per_major: int = 10,
    random_seed: int | None = None,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)

    skip_names = {"__pycache__", ".git"}
    all_major_dirs = [
        p for p in sorted(source_root.iterdir()) if p.is_dir() and p.name not in skip_names
    ]

    majors = list_major_categories(source_root)
    list_lines: list[str] = []
    report: dict = {
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "max_minors_per_major": max_minors_per_major,
        "random_seed": random_seed,
        "majors": [],
    }

    for major in majors:
        minors = list_minor_classes(major)
        if not minors:
            continue
        take = minors if len(minors) <= max_minors_per_major else rng.sample(minors, max_minors_per_major)
        take_sorted = sorted(take, key=lambda p: p.name.lower())
        major_entry = {"name": major.name, "minors_total": len(minors), "minors_selected": len(take_sorted), "sequences": []}
        print(f"[dataset_split] {major.name}: {len(take_sorted)} / {len(minors)} sequences", file=sys.stderr, flush=True)

        for seq_src in take_sorted:
            rel = f"{major.name}/{seq_src.name}".replace("\\", "/")
            seq_dst = output_root / major.name / seq_src.name
            meta = subsample_sequence(seq_src, seq_dst, rng)
            list_lines.append(rel)
            major_entry["sequences"].append({"relative": rel, **meta})

        report["majors"].append(major_entry)

    list_path = output_root / "list.txt"
    with open(list_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(sorted(list_lines)) + "\n")

    meta_path = output_root / "split_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    report["list_txt"] = str(list_path)
    report["split_meta_json"] = str(meta_path)
    processed_names = {m["name"] for m in report["majors"]}
    report["majors_all_top_level"] = len(all_major_dirs)
    report["majors_with_output_sequences"] = len(processed_names)
    report["majors_no_valid_sequence_under_source"] = sorted(
        p.name for p in all_major_dirs if p.name not in processed_names
    )
    return report
