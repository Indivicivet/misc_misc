import time
import numpy as np
import scipy.ndimage


def shift_custom(arr, shift_yx, cval=0):
    dy, dx = int(round(shift_yx[0])), int(round(shift_yx[1]))
    h, w = arr.shape[:2]
    if dy == 0 and dx == 0:
        return arr.copy()
    out = np.full_like(arr, cval)
    if dy > 0:
        src_y_start, src_y_end = 0, h - dy
        dst_y_start, dst_y_end = dy, h
    elif dy < 0:
        src_y_start, src_y_end = -dy, h
        dst_y_start, dst_y_end = 0, h + dy
    else:
        src_y_start, src_y_end = 0, h
        dst_y_start, dst_y_end = 0, h
    if dx > 0:
        src_x_start, src_x_end = 0, w - dx
        dst_x_start, dst_x_end = dx, w
    elif dx < 0:
        src_x_start, src_x_end = -dx, w
        dst_x_start, dst_x_end = 0, w + dx
    else:
        src_x_start, src_x_end = 0, w
        dst_x_start, dst_x_end = 0, w
    if src_y_start >= src_y_end or src_x_start >= src_x_end:
        return out
    out[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = arr[
        src_y_start:src_y_end, src_x_start:src_x_end
    ]
    return out


def shift_roll(arr, shift_yx, cval=0):
    dy, dx = int(round(shift_yx[0])), int(round(shift_yx[1]))
    h, w = arr.shape[:2]
    if abs(dy) >= h or abs(dx) >= w:
        return np.full_like(arr, cval)
    out = np.roll(arr, (dy, dx), axis=(0, 1))
    if dy > 0:
        out[:dy, :] = cval
    elif dy < 0:
        out[dy:, :] = cval
    if dx > 0:
        out[:, :dx] = cval
    elif dx < 0:
        out[:, dx:] = cval
    return out


def shift_scipy(arr, shift_yx, cval=0, order=0):
    return scipy.ndimage.shift(arr, shift_yx, cval=cval, order=order)


def verify_and_benchmark():
    # Define test setups
    test_cases = [
        # (size, dtype, shift_yx)
        ((100, 100), np.float32, (5, 10)),
        ((100, 100), np.float32, (0, 0)),
        ((1000, 1000), np.float64, (50, -30)),
        ((1000, 1000), np.uint8, (2, 2)),
        ((4000, 4000), np.float32, (20, 20)),
    ]

    print("=== Verification ===")
    for shape, dtype, shift_yx in test_cases:
        arr = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        res_custom = shift_custom(arr, shift_yx, cval=99)
        res_roll = shift_roll(arr, shift_yx, cval=99)
        res_scipy_o0 = shift_scipy(arr, shift_yx, cval=99, order=0)

        # Verify outputs match
        match_roll = np.array_equal(res_custom, res_roll)
        match_scipy = np.array_equal(res_custom, res_scipy_o0)

        print(f"Shape: {shape}, dtype: {dtype.__name__}, shift: {shift_yx}")
        print(f"  Custom vs Roll match: {match_roll}")
        print(f"  Custom vs SciPy (order=0) match: {match_scipy}")

    print("\n=== Benchmarking ===")
    print(
        f"{'Shape':<15} | {'Dtype':<8} | {'Shift':<10} | {'Custom (ms)':<12} | {'Roll (ms)':<10} | {'SciPy o0 (ms)':<14} | {'SciPy o3 (ms)':<14}"
    )
    print("-" * 90)

    for shape, dtype, shift_yx in test_cases:
        arr = np.arange(np.prod(shape), dtype=dtype).reshape(shape)

        # Determine number of iterations based on size
        n_iters = 1000 if shape[0] <= 100 else (100 if shape[0] <= 1000 else 10)

        # Custom
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = shift_custom(arr, shift_yx, cval=99)
        t_custom = (time.perf_counter() - t0) / n_iters * 1000

        # Roll
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = shift_roll(arr, shift_yx, cval=99)
        t_roll = (time.perf_counter() - t0) / n_iters * 1000

        # SciPy order=0
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = shift_scipy(arr, shift_yx, cval=99, order=0)
        t_scipy_o0 = (time.perf_counter() - t0) / n_iters * 1000

        # SciPy order=3 (only run for small/medium to save time,
        # otherwise it is extremely slow)
        if shape[0] <= 1000:
            t0 = time.perf_counter()
            for _ in range(n_iters):
                _ = shift_scipy(arr, shift_yx, cval=99, order=3)
            t_scipy_o3 = (time.perf_counter() - t0) / n_iters * 1000
            scipy_o3_str = f"{t_scipy_o3:12.4f}"
        else:
            scipy_o3_str = f"{'N/A':>12}"

        print(
            f"{str(shape):<15} | {dtype.__name__:<8}"
            f" | {str(shift_yx):<10} | {t_custom:12.4f}"
            f" | {t_roll:10.4f} | {t_scipy_o0:14.4f} | {scipy_o3_str:>14}"
        )


if __name__ == "__main__":
    verify_and_benchmark()
