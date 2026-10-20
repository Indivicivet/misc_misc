"""Quick render test script to generate and visually inspect test_photo.png and test_both.png."""

import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from geometry import generate_full_ring_geometry
from renderer import render_to_image


def main():
    app = QApplication(sys.argv)
    geom = generate_full_ring_geometry(
        compression=0.85, seed_morphology=0.0, random_seed=101
    )

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    img_photo = render_to_image(geom, mode="photorealistic", width=2048, height=2048)
    img_photo.save(str(out_dir / "test_photo.png"))

    img_both = render_to_image(geom, mode="both", width=3840, height=1920)
    img_both.save(str(out_dir / "test_both.png"))

    print("Full 360 ring test renders successfully written to output/")


if __name__ == "__main__":
    main()
