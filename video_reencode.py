#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

VIDEO_EXTS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".m4v",
    ".wmv",
    ".flv",
    ".webm",
    ".ts",
}
SUFFIX = ".compressed"


def format_bytes(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size) < 1024.0:
            return f"{size:3.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def get_media_info(path: Path) -> dict:
    """Returns size and duration using ffprobe."""
    size = path.stat().st_size
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,bit_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        data = json.loads(res.stdout).get("format", {})
        duration = float(data.get("duration", 0.0))
        bitrate = float(data.get("bit_rate", 0.0))
    except Exception:
        duration = 0.0
        bitrate = 0.0

    if bitrate <= 0 and duration > 0:
        bitrate = (size * 8) / duration

    return {"path": path, "size": size, "duration": duration, "bitrate": bitrate}


def discover_files(paths: list[str]) -> list[Path]:
    candidates = []
    for p_str in paths:
        p = Path(p_str)
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            if SUFFIX not in p.name:
                candidates.append(p)
        elif p.is_dir():
            for root, _, files in os.walk(p):
                for f in files:
                    fp = Path(root) / f
                    if fp.suffix.lower() in VIDEO_EXTS and SUFFIX not in fp.name:
                        candidates.append(fp)
    return candidates


def run_encode(items: list[dict], crf: int, preset: str, dry_run: bool = False):
    print(f"\nProcessing {len(items)} files (CRF={crf}, Preset={preset})...\n")
    for idx, item in enumerate(items, start=1):
        orig: Path = item["path"]
        out_path = orig.with_name(f"{orig.stem}{SUFFIX}.mp4")

        print(f"[{idx}/{len(items)}] {orig.name}")
        print(
            f"       Original size: {format_bytes(item['size'])} | Bitrate: {item['bitrate']/1e6:.2f} Mbps"
        )

        if out_path.exists():
            print(f"       -> Skipped: {out_path.name} already exists.\n")
            continue

        if dry_run:
            print(f"       -> [Dry Run] Would output to: {out_path.name}\n")
            continue

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-stats",
            "-i",
            str(orig),
            "-map",
            "0",
            "-c:v",
            "libx265",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-c:s",
            "copy",
            str(out_path),
        ]

        try:
            subprocess.run(cmd, check=True)
            new_size = out_path.stat().st_size
            pct = ((new_size - item["size"]) / item["size"]) * 100
            print(
                f"       -> Done: {out_path.name} ({format_bytes(new_size)}, {pct:+.1f}%)\n"
            )
        except subprocess.CalledProcessError:
            print(f"       -> [ERROR] Failed to transcode {orig.name}\n")


def run_review(paths: list[str], output_log: str | None = None):
    """Finds existing pairs, prints a comparison log, and optionally exports a TSV."""
    print("Scanning for existing (Original + Compressed) pairs...\n")
    pairs = []

    for p_str in paths:
        target = Path(p_str)
        search_dirs = [target] if target.is_dir() else [target.parent]
        for sdir in search_dirs:
            for root, _, files in os.walk(sdir):
                for f in files:
                    if SUFFIX in f:
                        comp_file = Path(root) / f
                        base_stem = comp_file.name.replace(
                            f"{SUFFIX}{comp_file.suffix}", ""
                        )
                        for cand in (
                            Path(root) / f"{base_stem}{ext}" for ext in VIDEO_EXTS
                        ):
                            if cand.exists() and cand != comp_file:
                                pairs.append((cand, comp_file))
                                break

    if not pairs:
        print("No duplicate original/compressed pairs found.")
        return

    total_orig = sum(orig.stat().st_size for orig, _ in pairs)
    total_comp = sum(comp.stat().st_size for _, comp in pairs)
    total_saved = total_orig - total_comp

    print(
        f"{'Original':<35} | {'Orig Size':<10} | {'Comp Size':<10} | {'Savings':<15}"
    )
    print("-" * 80)

    rows = []
    for orig, comp in pairs:
        orig_sz = orig.stat().st_size
        comp_sz = comp.stat().st_size
        diff = orig_sz - comp_sz
        pct = (diff / orig_sz) * 100 if orig_sz > 0 else 0

        print(
            f"{orig.name[:35]:<35} | {format_bytes(orig_sz):<10} | {format_bytes(comp_sz):<10} | {format_bytes(diff)} ({pct:4.1f}%)"
        )
        rows.append(
            {
                "original_path": str(orig.resolve()),
                "compressed_path": str(comp.resolve()),
                "orig_bytes": orig_sz,
                "comp_bytes": comp_sz,
                "saved_bytes": diff,
                "saved_percent": f"{pct:.1f}%",
            }
        )

    print("-" * 80)
    print(
        f"Total pairs: {len(pairs)} | Total potential reclaimable space: {format_bytes(total_saved)}"
    )

    if output_log:
        log_path = Path(output_log)
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nReport written to: {log_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Rank, batch-compress, and report video size comparisons (strictly non-destructive)."
    )
    parser.add_argument("paths", nargs="+", help="Files or directories to process")
    parser.add_argument(
        "--review",
        action="store_true",
        help="Scan and display a status report of existing original vs compressed pairs",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to write review results as a TSV file",
    )
    parser.add_argument(
        "--sort",
        choices=["bitrate", "size"],
        default="bitrate",
        help="Sort order: 'bitrate' (density/bloat) or 'size' (raw GBs)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only top N files after sorting",
    )
    parser.add_argument(
        "--min-mb",
        type=float,
        default=50.0,
        help="Skip files smaller than this size in MB (default: 50)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=24,
        help="x265 CRF quality setting (default: 24)",
    )
    parser.add_argument(
        "--preset",
        default="fast",
        help="x265 preset (ultrafast, fast, medium, slow)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matched files and metrics without encoding",
    )

    args = parser.parse_args()

    if args.review:
        run_review(args.paths, output_log=args.log_file)
        return

    raw_files = discover_files(args.paths)
    if not raw_files:
        print("No eligible video files found.")
        return

    print(f"Inspecting {len(raw_files)} files with ffprobe...")
    metadata = []
    min_bytes = args.min_mb * 1024 * 1024

    for f in raw_files:
        info = get_media_info(f)
        if info["size"] >= min_bytes:
            metadata.append(info)

    if args.sort == "bitrate":
        metadata.sort(key=lambda x: x["bitrate"], reverse=True)
    else:
        metadata.sort(key=lambda x: x["size"], reverse=True)

    if args.limit:
        metadata = metadata[: args.limit]

    run_encode(metadata, crf=args.crf, preset=args.preset, dry_run=args.dry_run)


if __name__ == "__main__":
    main()