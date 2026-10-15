#!/usr/bin/env python3
"""
Part 1: Dataset + Hand Detection audit (SLS landmark-geometry baseline)

Checks:
  1. Class labels      : folder names, A-Z coverage, images per class, exact
                         duplicates across different letters (label errors)
  2. Image quality     : unreadable files, low resolution, blur, brightness
  3. Hand detection    : failures, and detected-hand-count vs expected
                         (C = 1 hand, every other letter = 2 hands)
  4. Handedness        : how MediaPipe labels Left/Right, whether labels agree
                         with hand position in the image
  5. Duplicates        : exact (MD5) + near-duplicate (dHash) inside a letter,
                         important for a fair reference/test split later



"""

import argparse
import csv
import hashlib
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXPECTED_LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


# ----------------------------------------------------------------- helpers
def parse_args():
    p = argparse.ArgumentParser(description="SLS Part 1: dataset + hand detection audit")
    p.add_argument("--input", type=Path, default=Path("data/SLS_Dataset"))
    p.add_argument("--output", type=Path, default=Path("data/Cleaned_Dataset"))
    p.add_argument("--logs", type=Path, default=Path("data/audit_logs"))
    p.add_argument("--debug-dir", type=Path, default=None,
                   help="Save annotated images for detection failures / mismatches")
    p.add_argument("--min-conf", type=float, default=0.5)
    p.add_argument("--single-hand-letters", default="C",
                   help="Letters expected to use ONE hand (default: C)")
    p.add_argument("--expected-per-class", type=int, default=80)
    p.add_argument("--min-side", type=int, default=200, help="Min image side in px")
    p.add_argument("--blur-thresh", type=float, default=50.0,
                   help="Variance of Laplacian below this = blurry")
    p.add_argument("--dark-thresh", type=float, default=40.0)
    p.add_argument("--bright-thresh", type=float, default=220.0)
    p.add_argument("--dup-hamming", type=int, default=4,
                   help="dHash Hamming distance <= this = near duplicate")
    p.add_argument("--keep-any-detection", action="store_true",
                   help="Copy every detected image, even if hand count != expected")
    p.add_argument("--clean", action="store_true", help="Wipe output folder first")
    return p.parse_args()


def load_bgr(path: Path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except Exception:
        return None


def md5_file(path: Path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(gray):
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def pct(a, b):
    return round(a / b * 100, 1) if b else 0.0


# -------------------------------------------------------------------- main
def main():
    args = parse_args()
    single_letters = {c.strip().upper() for c in args.single_hand_letters.split(",") if c.strip()}

    if not args.input.is_dir():
        sys.exit(f"[ERROR] Input folder not found: {args.input.resolve()}")
    if args.clean and args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    args.logs.mkdir(parents=True, exist_ok=True)
    if args.debug_dir:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. class label check (folders) ----
    folders = sorted(d for d in args.input.iterdir() if d.is_dir())
    folder_names = [d.name for d in folders]
    missing = [c for c in EXPECTED_LETTERS if c not in folder_names]
    extra = [n for n in folder_names if n not in EXPECTED_LETTERS]

    records = []
    hash_to_files = defaultdict(list)  # exact-duplicate detection across ALL letters
    letter_hashes = defaultdict(list)  # (dhash, filename) per letter

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                        min_detection_confidence=args.min_conf) as hands:
        for folder in folders:
            letter = folder.name
            expected = 1 if letter.upper() in single_letters else 2
            files = sorted(f for f in folder.iterdir()
                           if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
            print(f"Auditing '{letter}' ({len(files)} images, expecting {expected} hand(s))...")

            for path in files:
                rec = dict(letter=letter, filename=path.name, readable=True, width=0, height=0,
                           blur=0.0, brightness=0.0, quality_flags="", detected=False,
                           num_hands=0, expected_hands=expected, count_ok=False,
                           labels="", scores="", label_order="", wrist_x="", status="")

                md5 = md5_file(path)
                hash_to_files[md5].append((letter, path.name))

                bgr = load_bgr(path)
                if bgr is None:
                    rec.update(readable=False, status="UNREADABLE")
                    records.append(rec)
                    continue

                h, w = bgr.shape[:2]
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                bright = float(gray.mean())
                flags = []
                if min(h, w) < args.min_side:
                    flags.append("low_res")
                if blur < args.blur_thresh:
                    flags.append("blurry")
                if bright < args.dark_thresh:
                    flags.append("dark")
                if bright > args.bright_thresh:
                    flags.append("overexposed")
                rec.update(width=w, height=h, blur=round(blur, 1),
                           brightness=round(bright, 1), quality_flags="|".join(flags))
                letter_hashes[letter].append((dhash(gray), path.name))

                # ---- 3 + 4. detection + handedness (image fed as-is, NOT flipped) ----
                result = hands.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                if result.multi_hand_landmarks:
                    info = []
                    for lm, hd in zip(result.multi_hand_landmarks, result.multi_handedness):
                        c = hd.classification[0]
                        info.append((lm.landmark[0].x, c.label, c.score))
                    info.sort(key=lambda t: t[0])  # left-to-right by wrist x
                    n = len(info)
                    labels = [i[1] for i in info]
                    if n == 2:
                        if labels == ["Left", "Right"]:
                            order = "consistent"       # Left label on image-left
                        elif labels == ["Right", "Left"]:
                            order = "crossed"          # Left label on image-right
                        else:
                            order = "same_label"       # both hands got same label
                    else:
                        order = "n/a"
                    rec.update(detected=True, num_hands=n, count_ok=(n == expected),
                               labels="|".join(labels),
                               scores="|".join(f"{i[2]:.2f}" for i in info),
                               label_order=order,
                               wrist_x="|".join(f"{i[0]:.2f}" for i in info))
                    rec["status"] = "OK" if n == expected else f"COUNT_MISMATCH(got {n}, want {expected})"
                else:
                    rec["status"] = "NO_HAND"

                # debug images
                if args.debug_dir and rec["status"] != "OK":
                    vis = bgr.copy()
                    if result.multi_hand_landmarks:
                        for lm in result.multi_hand_landmarks:
                            mp_draw.draw_landmarks(vis, lm, mp_hands.HAND_CONNECTIONS)
                    cv2.putText(vis, rec["status"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2)
                    out = args.debug_dir / letter
                    out.mkdir(exist_ok=True)
                    ok, buf = cv2.imencode(".jpg", vis)
                    if ok:
                        buf.tofile(str(out / f"{path.stem}_debug.jpg"))

                # copy to clean dataset
                keep = rec["detected"] and (rec["count_ok"] or args.keep_any_detection)
                if keep:
                    dst = args.output / letter
                    dst.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dst / path.name)

                records.append(rec)

    # ---- 5. duplicates ----
    exact_cross = [v for v in hash_to_files.values() if len({l for l, _ in v}) > 1]
    exact_same = [v for v in hash_to_files.values()
                  if len(v) > 1 and len({l for l, _ in v}) == 1]

    near_dups = defaultdict(list)
    for letter, items in letter_hashes.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                d = bin(items[i][0] ^ items[j][0]).count("1")
                if d <= args.dup_hamming:
                    near_dups[letter].append((items[i][1], items[j][1], d))

    # ---- per-letter summary ----
    summary = []
    for letter in folder_names:
        rs = [r for r in records if r["letter"] == letter]
        tot = len(rs)
        det = sum(r["detected"] for r in rs)
        ok = sum(r["count_ok"] for r in rs)
        lab = Counter(l for r in rs for l in r["labels"].split("|") if l)
        order = Counter(r["label_order"] for r in rs if r["label_order"] not in ("", "n/a"))
        summary.append(dict(
            letter=letter, total=tot,
            unreadable=sum(not r["readable"] for r in rs),
            detected=det, no_hand=sum(r["status"] == "NO_HAND" for r in rs),
            detect_pct=pct(det, tot),
            expected_hands=rs[0]["expected_hands"] if rs else "",
            hand_count_ok=ok, hand_count_ok_pct=pct(ok, tot),
            one_hand=sum(r["num_hands"] == 1 for r in rs),
            two_hands=sum(r["num_hands"] == 2 for r in rs),
            low_quality=sum(bool(r["quality_flags"]) for r in rs),
            near_dup_pairs=len(near_dups.get(letter, [])),
            label_Left=lab.get("Left", 0), label_Right=lab.get("Right", 0),
            order_consistent=order.get("consistent", 0),
            order_crossed=order.get("crossed", 0),
            order_same_label=order.get("same_label", 0),
        ))

    # ---- write CSVs ----
    with open(args.logs / "detection_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    with open(args.logs / "summary_by_letter.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # ---- console table ----
    print("\n" + "=" * 96)
    print(f"{'Let':<5}{'Total':>6}{'Detect':>8}{'Det%':>7}{'Want':>6}{'CountOK':>9}"
          f"{'OK%':>7}{'LowQ':>6}{'NearDup':>9}{'Left':>6}{'Right':>7}{'Crossed':>9}")
    print("-" * 96)
    for s in summary:
        print(f"{s['letter']:<5}{s['total']:>6}{s['detected']:>8}{s['detect_pct']:>7}"
              f"{s['expected_hands']:>6}{s['hand_count_ok']:>9}{s['hand_count_ok_pct']:>7}"
              f"{s['low_quality']:>6}{s['near_dup_pairs']:>9}{s['label_Left']:>6}"
              f"{s['label_Right']:>7}{s['order_crossed']:>9}")
    T = len(records)
    D = sum(r["detected"] for r in records)
    K = sum(r["count_ok"] for r in records)
    print("-" * 96)
    print(f"TOTAL images: {T} | detected: {D} ({pct(D, T)}%) | "
          f"hand count matches expected: {K} ({pct(K, T)}%)")
    print("=" * 96)

    # ---- auto-generated issues + markdown report ----
    issues = []
    if missing:
        issues.append(f"Missing letter folders: {', '.join(missing)}")
    if extra:
        issues.append(f"Unexpected folders (not A-Z): {', '.join(extra)}")
    for s in summary:
        if s["total"] != args.expected_per_class:
            issues.append(f"{s['letter']}: has {s['total']} images (expected {args.expected_per_class})")
        if s["unreadable"]:
            issues.append(f"{s['letter']}: {s['unreadable']} unreadable/corrupt file(s)")
        if s["detect_pct"] < 90:
            issues.append(f"{s['letter']}: low detection rate ({s['detect_pct']}%)")
        if s["hand_count_ok_pct"] < 90:
            issues.append(f"{s['letter']}: hand count differs from expected ({s['expected_hands']}) "
                          f"in {100 - s['hand_count_ok_pct']:.1f}% of images")
        if s["order_crossed"] or s["order_same_label"]:
            issues.append(f"{s['letter']}: handedness label oddities "
                          f"(crossed={s['order_crossed']}, same_label={s['order_same_label']})")
        if s["near_dup_pairs"]:
            issues.append(f"{s['letter']}: {s['near_dup_pairs']} near-duplicate pair(s)")
    if exact_cross:
        issues.append(f"{len(exact_cross)} identical image(s) appear in DIFFERENT letter folders "
                      f"(possible label error): {exact_cross[:5]}")
    if exact_same:
        issues.append(f"{len(exact_same)} exact duplicate group(s) inside the same letter")
    lowq_total = sum(s["low_quality"] for s in summary)
    if lowq_total:
        issues.append(f"{lowq_total} image(s) flagged for quality (see quality_flags column)")

    lab_all = Counter(l for r in records for l in r["labels"].split("|") if l)
    ord_all = Counter(r["label_order"] for r in records if r["label_order"] not in ("", "n/a"))
    c_labels = Counter(l for r in records if r["letter"].upper() in single_letters
                       for l in r["labels"].split("|") if l)

    with open(args.logs / "audit_report.md", "w", encoding="utf-8") as f:
        f.write("# Part 1 Audit Report: Dataset + Hand Detection\n\n")
        f.write(f"- Images audited: **{T}** across {len(folder_names)} folders\n")
        f.write(f"- Hand detected: **{D}** ({pct(D, T)}%)\n")
        f.write(f"- Detected hand count matches expectation (C=1, others=2): **{K}** ({pct(K, T)}%)\n")
        f.write(f"- MediaPipe settings: static_image_mode=True, max_num_hands=2, "
                f"min_detection_confidence={args.min_conf}\n\n")
        f.write("## MediaPipe handedness (images fed unflipped)\n\n")
        f.write(f"- Label totals: {dict(lab_all)}\n")
        f.write(f"- Labels for single-hand letter(s) {sorted(single_letters)}: {dict(c_labels)}\n")
        f.write(f"- Two-hand images, left-to-right label order: {dict(ord_all)}\n")
        f.write("  - consistent = 'Left' label on the image-left hand, 'Right' on the image-right\n")
        f.write("  - crossed / same_label = labels unreliable for those images\n\n")
        f.write("## Issues found\n\n")
        f.write("\n".join(f"- {i}" for i in issues) if issues else "- None")
        f.write("\n\n## Files\n\n- detection_log.csv (per image)\n- summary_by_letter.csv (per letter)\n")

    print(f"\nIssues found: {len(issues)} (listed in audit_report.md)")
    print(f"Report      : {(args.logs / 'audit_report.md').resolve()}")
    print(f"Clean copy  : {args.output.resolve()}")


if __name__ == "__main__":
    main()