from pathlib import Path

for fn in Path.cwd().glob("*.txt"):
    if "jpwords" in fn.name:
        continue
    Path(fn.with_name(f"{fn.stem}_jpwords.txt")).write_text(
        " // ".join(
            line.split()[0]
            for line in fn.read_text(encoding="utf-8").splitlines()
            if line[0] not in "#1qwertyuiopasdfghjklzxcvbnm"
        ),
        encoding="utf-8",
    )
