#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from tqdm import tqdm

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
CACHE_FILENAME = "video_filesize_cache.json"
AUDIO_BITRATE_BPS = 160_000  # 160k aac


def format_bytes(size: int | float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size) < 1024.0:
            return f"{size:3.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def estimate_encoded_size(
    orig_size: int, duration: float, crf: int, width: int = 0, height: int = 0
) -> int:
    """Estimates output filesize for x265 encoding based on duration/resolution or CRF ratio heuristic."""
    # CRF 28 is x265 default. Each step of ~6 roughly halves/doubles visual bitrate.
    # Base estimated visual bitrate at 1080p CRF 28 is ~1.5 Mbps.
    crf_factor = 2.0 ** ((28 - crf) / 6.0)

    if duration > 0 and width > 0 and height > 0:
        pixels = width * height
        base_video_bps = 1.5e6 * (pixels / (1920 * 1080)) * crf_factor
        total_bps = base_video_bps + AUDIO_BITRATE_BPS
        estimated_bytes = int((total_bps * duration) / 8)
        # Re-encoding shouldn't realistically be assumed larger than original unless original was extremely low bitrate
        return max(1, min(estimated_bytes, orig_size))

    if duration > 0:
        # Fallback if dimensions unavailable: assume typical 1080p target
        total_bps = (1.5e6 * crf_factor) + AUDIO_BITRATE_BPS
        estimated_bytes = int((total_bps * duration) / 8)
        return max(1, min(estimated_bytes, orig_size))

    # Fallback if duration is unknown: heuristic CRF compression ratio against original size
    # Assuming typical input is H.264 high-bitrate; CRF 24 is ~40% of original
    ratio = max(0.1, min(0.95, 0.40 * crf_factor))
    return int(orig_size * ratio)


def get_media_info(path: Path) -> dict:
    """Returns size, duration, and dimensions using ffprobe."""
    size = path.stat().st_size
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,bit_rate:stream=width,height",
        "-of",
        "json",
        str(path),
    ]
    duration = 0.0
    bitrate = 0.0
    width = 0
    height = 0
    try:
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        parsed = json.loads(res.stdout)
        fmt = parsed.get("format", {})
        duration = float(fmt.get("duration", 0.0))
        bitrate = float(fmt.get("bit_rate", 0.0))
        for stream in parsed.get("streams", []):
            if "width" in stream and "height" in stream:
                width = int(stream["width"])
                height = int(stream["height"])
                break
    except Exception:
        pass

    if bitrate <= 0 and duration > 0:
        bitrate = (size * 8) / duration

    return {
        "path": path,
        "size": size,
        "duration": duration,
        "bitrate": bitrate,
        "width": width,
        "height": height,
    }


def load_cache_file(cache_path: Path) -> dict[str, dict]:
    """Loads cache file mapping file path string to media info."""
    if not cache_path.is_file():
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_cache_file(cache_path: Path, data: dict[str, dict]):
    """Saves cache dictionary to file."""
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def discover_files(paths: list[str]) -> list[Path]:
    candidates = []
    with tqdm(desc="Discovering files", unit="dir") as pbar:
        for p_str in paths:
            p = Path(p_str)
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                if SUFFIX not in p.name:
                    candidates.append(p)
                    pbar.set_postfix({"found": len(candidates)})
            elif p.is_dir():
                for root, _, files in os.walk(p):
                    pbar.update(1)
                    for f in files:
                        fp = Path(root) / f
                        if fp.suffix.lower() in VIDEO_EXTS and SUFFIX not in fp.name:
                            candidates.append(fp)
                    pbar.set_postfix({"found": len(candidates)})
    return candidates


def run_encode(items: list[dict], crf: int, preset: str, dry_run: bool = False):
    print(f"\nProcessing {len(items)} files (CRF={crf}, Preset={preset})...\n")
    total_in = 0
    total_out = 0
    processed_count = 0

    for item in tqdm(items, desc="Encoding videos", unit="video"):
        orig: Path = item["path"]
        out_path = orig.with_name(f"{orig.stem}{SUFFIX}.mp4")
        orig_sz = item["size"]

        tqdm.write(
            f"\n--> {orig.name} | Original: {format_bytes(orig_sz)} | Bitrate: {item['bitrate']/1e6:.2f} Mbps"
        )

        if out_path.exists():
            tqdm.write(f"    -> Skipped: {out_path.name} already exists.")
            continue

        if dry_run:
            expected_sz = estimate_encoded_size(
                orig_size=orig_sz,
                duration=item.get("duration", 0.0),
                crf=crf,
                width=item.get("width", 0),
                height=item.get("height", 0),
            )
            est_diff = orig_sz - expected_sz
            est_pct = (est_diff / orig_sz) * 100 if orig_sz > 0 else 0
            tqdm.write(
                f"    -> [Dry Run] Would output to: {out_path.name} "
                f"(Expected: ~{format_bytes(expected_sz)}, Saving: {format_bytes(est_diff)} / {est_pct:.1f}%)"
            )
            total_in += orig_sz
            total_out += expected_sz
            processed_count += 1
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
            pct = ((new_size - orig_sz) / orig_sz) * 100 if orig_sz > 0 else 0
            tqdm.write(
                f"    -> Done: {out_path.name} ({format_bytes(new_size)}, {pct:+.1f}%)"
            )
            total_in += orig_sz
            total_out += new_size
            processed_count += 1
        except subprocess.CalledProcessError:
            tqdm.write(f"    -> [ERROR] Failed to transcode {orig.name}")

    # Summary
    if processed_count > 0:
        saved_bytes = total_in - total_out
        pct_saved = (saved_bytes / total_in) * 100 if total_in > 0 else 0
        tag = "[Dry Run Estimate]" if dry_run else "[Summary]"
        print("\n" + "=" * 65)
        print(f"{tag} Compression Results ({processed_count} files)")
        print(f"  Total input size:    {format_bytes(total_in)}")
        print(f"  Total output size:   {format_bytes(total_out)}")
        print(f"  Absolute saving:     {format_bytes(saved_bytes)}")
        print(f"  Compression ratio:   {pct_saved:.1f}% space saved")
        print("=" * 65 + "\n")


def run_review(paths: list[str], output_log: str | None = None):
    """Finds existing pairs, prints a comparison log, and optionally exports a TSV."""
    pairs = []

    with tqdm(desc="Discovering pairs", unit="dir") as pbar:
        for p_str in paths:
            target = Path(p_str)
            search_dirs = [target] if target.is_dir() else [target.parent]
            for sdir in search_dirs:
                for root, _, files in os.walk(sdir):
                    pbar.update(1)
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
                    pbar.set_postfix({"pairs": len(pairs)})

    if not pairs:
        print("No duplicate original/compressed pairs found.")
        return

    total_orig = sum(orig.stat().st_size for orig, _ in pairs)
    total_comp = sum(comp.stat().st_size for _, comp in pairs)
    total_saved = total_orig - total_comp

    print(f"{'Original':<35} | {'Orig Size':<10} | {'Comp Size':<10} | {'Savings':<15}")
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
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Read/write probed metadata to video_filesize_cache.json in the input folders to avoid re-probing",
    )

    args = parser.parse_args()

    if args.review:
        run_review(args.paths, output_log=args.log_file)
        return

    raw_files = discover_files(args.paths)
    if not raw_files:
        print("No eligible video files found.")
        return

    # Handle caching per folder
    caches: dict[Path, dict] = {}
    if args.use_cache:
        target_dirs = set()
        for p_str in args.paths:
            p = Path(p_str)
            target_dirs.add(p if p.is_dir() else p.parent)
        for d in target_dirs:
            cfile = d / CACHE_FILENAME
            caches[cfile] = load_cache_file(cfile)

    print(f"Inspecting {len(raw_files)} files with ffprobe...")
    metadata = []
    min_bytes = args.min_mb * 1024 * 1024
    cache_dirty = set()

    for f in tqdm(raw_files, desc="Probing metadata", unit="file"):
        info = None
        cfile = None

        if args.use_cache:
            # Match cache file by parent directory
            cfile = f.parent / CACHE_FILENAME
            if cfile not in caches:
                caches[cfile] = load_cache_file(cfile)
            cache_data = caches[cfile]
            cached_entry = cache_data.get(f.name)
            current_stat = f.stat()

            if (
                cached_entry
                and cached_entry.get("size") == current_stat.st_size
                and cached_entry.get("mtime") == current_stat.st_mtime
            ):
                info = {
                    "path": f,
                    "size": cached_entry["size"],
                    "duration": cached_entry.get("duration", 0.0),
                    "bitrate": cached_entry.get("bitrate", 0.0),
                    "width": cached_entry.get("width", 0),
                    "height": cached_entry.get("height", 0),
                }

        if info is None:
            info = get_media_info(f)
            if args.use_cache and cfile is not None:
                current_stat = f.stat()
                caches[cfile][f.name] = {
                    "size": info["size"],
                    "duration": info["duration"],
                    "bitrate": info["bitrate"],
                    "width": info.get("width", 0),
                    "height": info.get("height", 0),
                    "mtime": current_stat.st_mtime,
                }
                cache_dirty.add(cfile)

        if info["size"] >= min_bytes:
            metadata.append(info)

    if args.use_cache:
        for cfile in cache_dirty:
            save_cache_file(cfile, caches[cfile])

    if args.sort == "bitrate":
        metadata.sort(key=lambda x: x["bitrate"], reverse=True)
    else:
        metadata.sort(key=lambda x: x["size"], reverse=True)

    if args.limit:
        metadata = metadata[: args.limit]

    run_encode(metadata, crf=args.crf, preset=args.preset, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
