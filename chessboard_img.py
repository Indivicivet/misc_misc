import numpy as np
from PIL import Image


def make_chessboard(
    width=8,
    height=8,
    square_w=256,
    border_thick=8,
) -> Image.Image:
    arr = np.zeros(
        (square_w * height, square_w * width),
        dtype=np.uint8,
    )
    for j in range(height):
        for i in range(width):
            if (i + j) % 2:
                continue
            arr[
                j * square_w : (j + 1) * square_w,
                i * square_w : (i + 1) * square_w,
            ] = 255
    padded = np.pad(arr, border_thick)
    return Image.fromarray(padded, mode="L").convert("RGB")


if __name__ == "__main__":
    for w, h in [(8, 8), (10, 8), (12, 8)]:
        make_chessboard(width=w, height=h).save(f"chessboard_{w=}_{h=}.png")
