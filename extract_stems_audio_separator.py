"""
split song into open format stems and package up in .stem.mp4 format

Uses audio-separator (supporting RoFormer models or 4-stem Demucs models)
+ MP4Box + stempeg

Requires:
  - ffmpeg, MP4Box installed and on PATH
  - pip install audio-separator[gpu]  (or audio-separator for CPU)
  - pip install stempeg mutagen pydub
"""

import json
import subprocess
from pathlib import Path

import mutagen.easyid3
import pydub
import stempeg
from audio_separator.separator import Separator

FFMPEG_EXE = "ffmpeg.exe"
MP4BOX_EXE = "MP4Box.exe"
OUT_FOLDER = Path(r"D:\music_prod\stems_openfmt")

# Standard 4-stem model via audio-separator:
# - "htdemucs_ft.yaml" or "htdemucs.yaml" produces Vocals, Drums, Bass, Other.
# - 2-stem vocal RoFormer models (e.g. "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
#   only produce Vocals and Instrumental.
DEFAULT_MODEL = "htdemucs_ft.yaml"


def ffmpeg_encode(input_wav: Path, output_mp4: Path):
    subprocess.run(
        [
            FFMPEG_EXE,
            "-y",
            "-i",
            str(input_wav),
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            str(output_mp4),
        ],
        check=True,
    )


def match_stem(stem_files: list[str], keyword: str) -> Path:
    keyword_lower = keyword.lower()
    for name in stem_files:
        if keyword_lower in Path(name).stem.lower():
            return Path(name)
    raise RuntimeError(
        f"Could not find stem matching '{keyword}' in output files: {stem_files}"
    )


input_path = Path(input("input_path: ").strip().strip('"'))
temp_dir = OUT_FOLDER / "temp"
temp_dir.mkdir(exist_ok=True, parents=True)

audio = mutagen.easyid3.EasyID3(str(input_path))
title = audio.get("title", [input_path.stem])[0]
artist = audio.get("artist", ["Unknown Artist"])[0]
album = audio.get("album", ["Unknown Album"])[0]
print(f"{artist=} - {title=} ({album=})")

print("Initializing audio-separator...")
separator = Separator(
    output_dir=str(temp_dir),
    output_format="WAV",
)

print(f"Loading model '{DEFAULT_MODEL}'...")
separator.load_model(model_filename=DEFAULT_MODEL)

print("Running stem separation...")
output_stem_names = separator.separate(str(input_path))
print(f"Separated files: {output_stem_names}")

# audio-separator returns filenames relative to output_dir or full paths
stem_files = [
    temp_dir / f if not Path(f).is_absolute() else Path(f) for f in output_stem_names
]

stem_paths = {
    "vocals": match_stem([str(p) for p in stem_files], "vocal"),
    "drums": match_stem([str(p) for p in stem_files], "drum"),
    "bass": match_stem([str(p) for p in stem_files], "bass"),
    "other": match_stem([str(p) for p in stem_files], "other"),
}

print("Encoding stems to AAC MP4...")
stem_mp4s = {}
for stem, path_wav in stem_paths.items():
    mp4_path = path_wav.with_suffix(".mp4")
    ffmpeg_encode(path_wav, mp4_path)
    stem_mp4s[stem] = mp4_path

combined = None
for path in stem_paths.values():
    segment = pydub.AudioSegment.from_file(path)
    combined = segment if combined is None else combined.overlay(segment)
master_path = temp_dir / "temp_master.wav"
combined.export(master_path, format="wav")
master_mp4_path = master_path.with_suffix(".mp4")
ffmpeg_encode(master_path, master_mp4_path)

temp_no_meta_stem = temp_dir / f"{input_path.stem}_nometa.stem.mp4"
print("Running MP4Box...")
subprocess.run(
    [
        MP4BOX_EXE,
        "-new",
        str(temp_no_meta_stem),
        "-add",
        f"{master_mp4_path}:name=Master:group=1",
        "-add",
        f"{stem_mp4s['vocals']}:name=Vocals:group=2",
        "-add",
        f"{stem_mp4s['drums']}:name=Drums:group=2",
        "-add",
        f"{stem_mp4s['bass']}:name=Bass:group=2",
        "-add",
        f"{stem_mp4s['other']}:name=Other:group=2",
    ],
    check=True,
)

stem_metadata = {
    "version": "1.0",
    "title": title,
    "artist": artist,
    "album": album,
    "stems": [
        {"id": 1, "name": "Vocals", "color": "#FF00FF"},
        {"id": 2, "name": "Drums", "color": "#FF0000"},
        {"id": 3, "name": "Bass", "color": "#FFFF00"},
        {"id": 4, "name": "Other", "color": "#00FFFF"},
    ],
}
json_path = master_mp4_path.with_name("stem_metadata.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(stem_metadata, f, indent=2)

final_stem_out = OUT_FOLDER / f"{input_path.stem}.stem.mp4"
stempeg.inject(
    filename=str(temp_no_meta_stem),
    metadata=str(stem_metadata),
    out_filename=str(final_stem_out),
)
print("done!")

print("clearing temp...")
master_path.unlink(missing_ok=True)
master_mp4_path.unlink(missing_ok=True)
json_path.unlink(missing_ok=True)
temp_no_meta_stem.unlink(missing_ok=True)
for fp in stem_paths.values():
    fp.unlink(missing_ok=True)
for fp in stem_mp4s.values():
    fp.unlink(missing_ok=True)
