import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def tet_n(x, n):
    if n == 0:
        return x ** 0
    n -= 1
    y = x.copy()
    while n:
        y = x ** y
        n -= 1
    return y


SAMPLES = 10000
MAX_N = 100

xx = xr.DataArray(
    data=[np.linspace(0, 1, SAMPLES)],
    coords={"x": np.linspace(0, 1, SAMPLES)},
    dims=["n", "x"],
)
tet_ns = [xx ** 0, xx]
for _ in range(MAX_N - 1):
    tet_ns.append(xx ** tet_ns[-1])

tetrated_n = xr.concat(tet_ns, dim="n")
# tetrated_n.plot()  # 2D plot
tetrated_n.plot.line(x="x")
plt.show()
