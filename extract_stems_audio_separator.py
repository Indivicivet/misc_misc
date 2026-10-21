"""
split song into open format stems and package up in .stem.mp4 format

Uses audio-separator (defaulting to BS-Roformer-SW 6-stem model, with 4-stem packaging)
+ MP4Box + stempeg

Requires:
  - ffmpeg, MP4Box installed and on PATH
  - pip install "audio-separator[gpu]"
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

# BS-Roformer-SW is a multi-stem model (separates Vocals, Drums, Bass, Guitar, Piano, Other)
# audio-separator automatically downloads the .ckpt and its .yaml config on first run.
DEFAULT_MODEL = "BS-Rofo-SW-Fixed.ckpt"


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


def match_stem(stem_files: list[Path], keyword: str) -> Path:
    keyword_lower = keyword.lower()
    for p in stem_files:
        if keyword_lower in p.stem.lower():
            return p
    raise RuntimeError(
        f"Could not find stem matching '{keyword}' in output files: {[str(p) for p in stem_files]}"
    )


input_path = Path(input("input_path: ").strip().strip('"'))
temp_dir = OUT_FOLDER / "temp"
temp_dir.mkdir(exist_ok=True, parents=True)

audio = mutagen.easyid3.EasyID3(str(input_path))
title = audio.get("title", [input_path.stem])[0]
artist = audio.get("artist", ["Unknown Artist"])[0]
album = audio.get("album", ["Unknown Album"])[0]
print(f"\nProcessing: {artist} - {title} ({album})")

# Low-VRAM configuration (4GB GPU friendly):
# - chunk_duration=30 limits continuous self-attention sequences into 30s segments
# - sample_rate=44100 / WAV output
print(
    f"Initializing audio-separator with 4GB VRAM safety settings (chunk_duration=30)..."
)
separator = Separator(
    output_dir=str(temp_dir),
    output_format="WAV",
    chunk_duration=30,
)

print(f"Loading '{DEFAULT_MODEL}'...")
separator.load_model(model_filename=DEFAULT_MODEL)

print("Running BS-Roformer separation...")
output_stem_names = separator.separate(str(input_path))
print(f"Separated files: {output_stem_names}")

stem_files = [
    temp_dir / f if not Path(f).is_absolute() else Path(f) for f in output_stem_names
]

# Native Instruments .stem.mp4 standard format defines 4 stems: Vocals, Drums, Bass, Other.
# If BS-Roformer-SW outputs extra stems (guitar, piano), combine them into the 'other' stem.
vocal_path = match_stem(stem_files, "vocal")
drum_path = match_stem(stem_files, "drum")
bass_path = match_stem(stem_files, "bass")

other_candidates = [
    p for p in stem_files if p not in (vocal_path, drum_path, bass_path)
]

if len(other_candidates) == 1:
    other_path = other_candidates[0]
else:
    # Blend guitar, piano, and other into the 4th stem
    print(
        "Combining remaining instrument stems (guitar/piano/other) into standard 4th 'Other' stem..."
    )
    other_combined = None
    for p in other_candidates:
        seg = pydub.AudioSegment.from_file(p)
        other_combined = seg if other_combined is None else other_combined.overlay(seg)
    other_path = temp_dir / f"{input_path.stem}_combined_other.wav"
    other_combined.export(other_path, format="wav")

stem_paths = {
    "vocals": vocal_path,
    "drums": drum_path,
    "bass": bass_path,
    "other": other_path,
}

print("Encoding stems to AAC MP4...")
stem_mp4s = {}
for stem, path_wav in stem_paths.items():
    mp4_path = temp_dir / f"{stem}.mp4"
    ffmpeg_encode(path_wav, mp4_path)
    stem_mp4s[stem] = mp4_path

# Create combined master mix
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
json_path = temp_dir / "stem_metadata.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(stem_metadata, f, indent=2)

final_stem_out = OUT_FOLDER / f"{input_path.stem}.stem.mp4"
stempeg.inject(
    filename=str(temp_no_meta_stem),
    metadata=str(stem_metadata),
    out_filename=str(final_stem_out),
)
print(f"Done! Created: {final_stem_out}")

print("Cleaning temp files...")
master_path.unlink(missing_ok=True)
master_mp4_path.unlink(missing_ok=True)
json_path.unlink(missing_ok=True)
temp_no_meta_stem.unlink(missing_ok=True)
if other_path != other_candidates[0]:
    other_path.unlink(missing_ok=True)
for fp in stem_files:
    fp.unlink(missing_ok=True)
for fp in stem_mp4s.values():
    fp.unlink(missing_ok=True)
