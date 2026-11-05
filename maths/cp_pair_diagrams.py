from dataclasses import dataclass

import numpy as np
from matplotlib import colormaps
from matplotlib import pyplot as plt


@dataclass
class Atom:
    p: int = 2
    offs: int = 0


@dataclass
class CurveData:
    p: int
    x: np.ndarray
    y: np.ndarray
    col: np.ndarray = None

    def plot(self):
        if self.col is None:
            plt.plot(self.x, self.y)
            return
        plt.plot(self.x, self.y, color=self.col)


@dataclass
class CurveGenerator:
    n_thetas: int = 1080
    exp_dropoff: float = 10
    min_amp: float = 1
    max_base_amp: float = 2
    _thetas: np.ndarray = None
    _cos: np.ndarray = None
    _sin: np.ndarray = None

    def __post_init__(self):
        self._thetas = np.linspace(0, 2 * np.pi, self.n_thetas + 1)
        # 90deg offset so 0 is top not right:
        self._cos = -np.sin(self._thetas)
        self._sin = np.cos(self._thetas)

    def generate_curve(self, atom, idx=0):
        amps = (
            self.min_amp
            + (self.max_base_amp - self.min_amp)
            * np.tanh((atom.p - 2) * 0.6 + idx * 0.2 * atom.p)
            + np.exp(
                -self.exp_dropoff
                * np.abs(
                    np.sin((self._thetas * atom.p) / 2 + atom.p * atom.offs * np.pi)
                )
            )
        )
        return CurveData(
            p=atom.p,
            x=amps * self._cos,
            y=amps * self._sin,
            col=colormaps["hsv"]((atom.p * 0.15 + idx * 0.05) % 1),
        )

    def generate_all_curves(self, *atoms):
        try:
            _ = atoms[0]
        except IndexError as e:
            raise ValueError("atoms must be nonempty") from e
        if not isinstance(atoms[0], Atom):
            atoms = [Atom(*a) for a in atoms]
        return [
            self.generate_curve(atom, idx=i_this_prime)
            for prime in {a.p for a in atoms}
            for i_this_prime, atom in enumerate(a for a in atoms if a.p == prime)
        ]


if __name__ == "__main__":
    plt.style.use("dark_background")
    plt.figure(figsize=(10, 10))
    for curve in CurveGenerator().generate_all_curves(
        # [2, 0], [2, 1 / 4], [4, 0]  # 4 isn't prime but it's an easy example
        [5, 0],
        [5, 1 / 10],
        *[[2, i / 10] for i in range(5)]
    ):
        curve.plot()
    plt.axis("scaled")
    plt.gca().set_axis_off()
    plt.show()
