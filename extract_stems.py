"""
split song into open format stems and package up in .stem.mp4 format

equivalent to running demucs + Stem Creator with fewer steps

requires ffmpeg, mp4box installed
+ pip install demucs
"""

import subprocess
from pathlib import Path

import mutagen.easyid3
import pydub


FFMPEG_EXE = "ffmpeg.exe"  # obvs ensure in path
MP4BOX_EXE = "MP4Box.exe"  # obvs ensure in path

OUT_FOLDER = Path(r"D:\music_prod\stems_openfmt")


def ffmpeg_encode(input_wav: Path, output_mp4: Path):
    subprocess.run(
        [
            FFMPEG_EXE,
            "-y",
            "-i",
            str(input_wav),
            "-c:a",
            "libvo_aacenc", # older ver? - new vers can use aac?
            "-b:a",
            "256k",
            str(output_mp4),
        ],
    )


input_path = Path(input("input_path: ").strip().strip('"'))
temp_dir = OUT_FOLDER / "temp"
temp_dir.mkdir(exist_ok=True, parents=True)

audio = mutagen.easyid3.EasyID3(str(input_path))
title = audio.get("title", [input_path.stem])[0]
artist = audio.get("artist", ["Unknown Artist"])[0]
album = audio.get("album", ["Unknown Album"])[0]
print(f"{artist=} - {title=} ({album=})")

print("running demucs")
try:
    # easy way to run demucs...?
    subprocess.run(
        [
            "python",
            "-m",
            "demucs",
            "-n",
            "htdemucs",
            "--mp3",  # something wrong with WAV on windows...
            "-o",
            str(temp_dir),
            str(input_path),
        ]
    )
except FileNotFoundError as e:
    raise RuntimeError("couldn't find demucs") from e

# this is just where demucs puts them:
demucs_out_dir = temp_dir / "htdemucs" / input_path.stem
print(f"stems directory: {demucs_out_dir}")
stem_mp3s = {
    "vocals": demucs_out_dir / "vocals.mp3",
    "drums": demucs_out_dir / "drums.mp3",
    "bass": demucs_out_dir / "bass.mp3",
    "other": demucs_out_dir / "other.mp3",
}
stem_mp4s = {}
for stem, path_mp3 in stem_mp3s.items():
    mp4_path = path_mp3.with_suffix(".mp4")
    ffmpeg_encode(path_mp3, mp4_path)
    stem_mp4s[stem] = mp4_path

# todo :: combined == combined demucs; maybe it's redundant and just use the input?
combined = None
for name, path in stem_mp3s.items():
    segment = pydub.AudioSegment.from_file(path)
    combined = segment if combined is None else combined.overlay(segment)
master_path = temp_dir / "temp_master.wav"
combined.export(master_path, format="wav")
master_mp4_path = master_path.with_suffix(".mp4")
ffmpeg_encode(master_path, master_mp4_path)

stem_output = OUT_FOLDER / f"{input_path.stem}.stem.mp4"
print(f"running mp4box")
subprocess.run(
    [
        MP4BOX_EXE,
        "-new",
        str(stem_output),
        f"-add",
        f"{master_mp4_path}:name=Master:group=1",
        f"-add",
        f"{stem_mp4s['vocals']}:name=Vocals:group=2",
        f"-add",
        f"{stem_mp4s['drums']}:name=Drums:group=2",
        f"-add",
        f"{stem_mp4s['bass']}:name=Bass:group=2",
        f"-add",
        f"{stem_mp4s['other']}:name=Other:group=2",
    ]
)

print("clearing temp")
master_path.unlink()
master_mp4_path.unlink()
for fp in stem_mp3s.values():
    fp.unlink()
for fp in stem_mp4s.values():
    fp.unlink()
