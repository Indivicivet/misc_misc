"""Interactive GUI for the 1D Frank-Oseen nematic cell simulator.

Run from the `physics` directory with:
    python -m liquid_crystal.lc_gui
"""

import sys
import time
from collections import deque

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from liquid_crystal import frank_model as fm

FRAME_MS = 30
MAX_STEPS_PER_FRAME = 4000
HISTORY_LENGTH = 800

N_ROD_LAYERS = 13
ROD_LENGTH = 0.075
COLUMN_SPACING = 0.1
VIEW_HALF_WIDTH = 0.16

# Illustrative material values, roughly in the range of E7 (positive delta_eps)
# and MLC-6608 (negative delta_eps). They are approximate, not datasheet values.
_E7_LIKE = {
    "k11": 11.1,
    "k22": 6.5,
    "k33": 17.1,
    "eps_perp": 5.2,
    "delta_eps": 13.8,
    "gamma1": 0.15,
    "n_o": 1.52,
    "delta_n": 0.22,
}
_MLC6608_LIKE = {
    "k11": 16.7,
    "k22": 7.0,
    "k33": 18.1,
    "eps_perp": 7.8,
    "delta_eps": -4.2,
    "gamma1": 0.186,
    "n_o": 1.476,
    "delta_n": 0.083,
}
PRESETS = {
    "TN 90° (transmissive)": {
        **_E7_LIKE,
        "pitch": 0.0,
        "thickness": 5.0,
        "twist": 90.0,
        "pretilt_bottom": 2.0,
        "pretilt_top": 2.0,
        "azimuth_bottom": 0.0,
        "polarizer": 0.0,
        "optics_mode": "transmissive_crossed",
        "voltage": 0.0,
        "vt_max": 6.0,
    },
    "VAN (LCoS, reflective)": {
        **_MLC6608_LIKE,
        "pitch": 0.0,
        "thickness": 2.5,
        "twist": 0.0,
        "pretilt_bottom": 88.0,
        "pretilt_top": 88.0,
        "azimuth_bottom": 0.0,
        "polarizer": 45.0,
        "optics_mode": "reflective_crossed",
        "voltage": 0.0,
        "vt_max": 8.0,
    },
    "Planar ECB (transmissive)": {
        **_E7_LIKE,
        "pitch": 0.0,
        "thickness": 3.0,
        "twist": 0.0,
        "pretilt_bottom": 1.0,
        "pretilt_top": 1.0,
        "azimuth_bottom": 0.0,
        "polarizer": 45.0,
        "optics_mode": "transmissive_crossed",
        "voltage": 0.0,
        "vt_max": 5.0,
    },
    "Hybrid HAN (transmissive)": {
        **_E7_LIKE,
        "pitch": 0.0,
        "thickness": 3.0,
        "twist": 0.0,
        "pretilt_bottom": 2.0,
        "pretilt_top": 90.0,
        "azimuth_bottom": 0.0,
        "polarizer": 45.0,
        "optics_mode": "transmissive_crossed",
        "voltage": 0.0,
        "vt_max": 5.0,
    },
}

OPTICS_MODE_LABELS = {
    "transmissive_crossed": "Transmissive, crossed polarisers",
    "transmissive_parallel": "Transmissive, parallel polarisers",
    "reflective_crossed": "Reflective (LCoS mirror + PBS)",
}


class NoWheelSlider(QSlider):
    """Slider that ignores the mouse wheel so scrolling the panel is safe."""

    def wheelEvent(self, event):
        event.ignore()


class SliderSpin(QWidget):
    """Horizontal slider paired with a synced spin box. Optional log mapping."""

    valueChanged = Signal(float)

    def __init__(
        self, minimum, maximum, value, decimals=2, suffix="", log=False, steps=1000
    ):
        super().__init__()
        self._minimum = minimum
        self._maximum = maximum
        self._log = log
        self._steps = steps

        self.slider = NoWheelSlider(Qt.Horizontal)
        self.slider.setRange(0, steps)
        self.slider.setMinimumWidth(140)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(decimals)
        self.spin.setSuffix(suffix)
        self.spin.setKeyboardTracking(False)
        self.spin.setSingleStep(10 ** (-decimals) if log else (maximum - minimum) / 200)
        self.spin.setMinimumWidth(100)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)
        self.setValue(value)

    def _to_slider(self, value):
        if self._log:
            fraction = np.log(value / self._minimum) / np.log(
                self._maximum / self._minimum
            )
        else:
            fraction = (value - self._minimum) / (self._maximum - self._minimum)
        return int(round(fraction * self._steps))

    def _from_slider(self, position):
        fraction = position / self._steps
        if self._log:
            return self._minimum * (self._maximum / self._minimum) ** fraction
        return self._minimum + fraction * (self._maximum - self._minimum)

    def _on_slider(self, position):
        self.spin.setValue(self._from_slider(position))

    def _on_spin(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_slider(value))
        self.slider.blockSignals(False)
        self.valueChanged.emit(value)

    def value(self):
        return self.spin.value()

    def setValue(self, value):
        self.spin.setValue(value)
        # Keep the slider in sync even if the spin value did not change.
        self._on_spin(self.spin.value())


def _spin(minimum, maximum, value, decimals=2, suffix="", integer=False):
    spin = QSpinBox() if integer else QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    if not integer:
        spin.setDecimals(decimals)
        spin.setSingleStep(10 ** (-decimals + 1))
    spin.setSuffix(suffix)
    spin.setKeyboardTracking(False)
    spin.setMinimumWidth(110)
    spin.setValue(value)
    return spin


class LCWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nematic LC cell: Frank-Oseen simulator")
        self.w = {}
        self._suspend_updates = True
        self.vt_curve = None
        self.history = deque(maxlen=HISTORY_LENGTH)
        self.sim_time = 0.0
        self.last_steps = 0
        self.speed_limited = False

        controls = self._build_controls()
        scroll = QScrollArea()
        scroll.setWidget(controls)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(430)

        self.canvas_3d = FigureCanvasQTAgg(Figure(figsize=(4.5, 7)))
        self.canvas_2d = FigureCanvasQTAgg(Figure(figsize=(8, 7)))
        self._last_2d_draw = 0.0
        # Constrained layout is more than half the cost of a 2D redraw, so it only
        # runs after resizes or preset changes and is frozen again once drawn.
        self.canvas_2d.mpl_connect("resize_event", lambda _: self._relayout_2d())
        self.canvas_2d.mpl_connect(
            "draw_event", lambda _: self.canvas_2d.figure.set_layout_engine("none")
        )
        self._relayout_2d()
        self._setup_3d_axes()
        self._setup_2d_axes()

        view_row = QHBoxLayout()
        for label, (elev, azim) in {
            "3D": (18, -55),
            "Side (xz)": (0, -90),
            "Side (yz)": (0, 0),
            "Top": (90, -90),
        }.items():
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, e=elev, a=azim: self._set_view(e, a))
            view_row.addWidget(button)
        left_plot = QWidget()
        left_layout = QVBoxLayout(left_plot)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.canvas_3d, stretch=1)
        left_layout.addLayout(view_row)

        plots = QSplitter(Qt.Horizontal)
        plots.addWidget(left_plot)
        plots.addWidget(self.canvas_2d)
        plots.setSizes([450, 800])

        main = QSplitter(Qt.Horizontal)
        main.addWidget(scroll)
        main.addWidget(plots)
        main.setSizes([430, 1250])
        self.setCentralWidget(main)

        self._suspend_updates = False
        self._apply_preset("TN 90° (transmissive)")

        self._last_tick = time.perf_counter()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(FRAME_MS)

    # ------------------------------------------------------------------ controls

    def _build_controls(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        presets = QGroupBox("Presets")
        preset_layout = QGridLayout(presets)
        for i, name in enumerate(PRESETS):
            button = QPushButton(name)
            button.clicked.connect(lambda _=False, n=name: self._apply_preset(n))
            preset_layout.addWidget(button, i // 2, i % 2)
        layout.addWidget(presets)

        drive = QGroupBox("Drive")
        drive_form = QFormLayout(drive)
        self.w["voltage"] = SliderSpin(0.0, 20.0, 0.0, decimals=2, suffix=" V")
        self.w["voltage"].valueChanged.connect(self._on_voltage_changed)
        drive_form.addRow("Voltage (RMS)", self.w["voltage"])

        self.mode_group = QButtonGroup(self)
        mode_row = QHBoxLayout()
        for mode_id, label in enumerate(("Live dynamics", "Static equilibrium")):
            radio = QRadioButton(label)
            self.mode_group.addButton(radio, mode_id)
            mode_row.addWidget(radio)
        self.mode_group.button(0).setChecked(True)
        self.mode_group.idClicked.connect(self._on_mode_changed)
        drive_form.addRow("Mode", mode_row)

        self.w["speed"] = SliderSpin(1e-4, 1.0, 0.02, decimals=4, suffix=" x", log=True)
        self.w["speed"].setToolTip("Simulated seconds per real second")
        drive_form.addRow("Time scale", self.w["speed"])

        self.square_wave = QCheckBox("Square wave 0 ↔ V")
        self.w["period"] = _spin(1.0, 5000.0, 200.0, decimals=0, suffix=" ms")
        square_row = QHBoxLayout()
        square_row.addWidget(self.square_wave)
        square_row.addWidget(QLabel("period"))
        square_row.addWidget(self.w["period"])
        drive_form.addRow(square_row)

        buttons = QHBoxLayout()
        reset = QPushButton("Reset (V = 0 layout)")
        reset.clicked.connect(self._reset_state)
        snap = QPushButton("Snap to equilibrium")
        snap.clicked.connect(self._snap_to_equilibrium)
        buttons.addWidget(reset)
        buttons.addWidget(snap)
        drive_form.addRow(buttons)

        vt_row = QHBoxLayout()
        vt_button = QPushButton("Compute V-T curve")
        vt_button.clicked.connect(self._compute_vt_curve)
        self.w["vt_max"] = _spin(0.5, 50.0, 6.0, decimals=1, suffix=" V max")
        vt_row.addWidget(vt_button)
        vt_row.addWidget(self.w["vt_max"])
        drive_form.addRow(vt_row)
        layout.addWidget(drive)

        material = QGroupBox("Material")
        material_form = QFormLayout(material)
        for key, label, minimum, maximum, decimals, suffix in (
            ("k11", "K11 (splay)", 0.1, 100.0, 2, " pN"),
            ("k22", "K22 (twist)", 0.1, 100.0, 2, " pN"),
            ("k33", "K33 (bend)", 0.1, 100.0, 2, " pN"),
            ("eps_perp", "ε⊥", 1.0, 50.0, 2, ""),
            ("delta_eps", "Δε", -30.0, 50.0, 2, ""),
            ("gamma1", "γ1 (rot. viscosity)", 0.001, 5.0, 3, " Pa·s"),
            ("pitch", "Chiral pitch (0 = none)", -500.0, 500.0, 2, " µm"),
        ):
            self.w[key] = _spin(minimum, maximum, 1.0, decimals, suffix)
            self.w[key].valueChanged.connect(self._on_cell_changed)
            material_form.addRow(label, self.w[key])
        layout.addWidget(material)

        cell = QGroupBox("Cell geometry and anchoring")
        cell_form = QFormLayout(cell)
        self.w["thickness"] = _spin(0.5, 50.0, 5.0, 2, " µm")
        self.w["thickness"].valueChanged.connect(self._on_cell_changed)
        cell_form.addRow("Thickness d", self.w["thickness"])
        for key, label, minimum, maximum in (
            ("twist", "Twist angle", -360.0, 360.0),
            ("pretilt_bottom", "Pretilt bottom", -90.0, 90.0),
            ("pretilt_top", "Pretilt top", -90.0, 90.0),
            ("azimuth_bottom", "Rubbing azimuth (bottom)", -180.0, 180.0),
        ):
            self.w[key] = SliderSpin(minimum, maximum, 0.0, decimals=1, suffix="°")
            self.w[key].valueChanged.connect(self._on_cell_changed)
            cell_form.addRow(label, self.w[key])
        self.w["n_nodes"] = _spin(11, 201, 41, integer=True)
        self.w["n_nodes"].valueChanged.connect(self._on_cell_changed)
        cell_form.addRow("Grid nodes", self.w["n_nodes"])
        layout.addWidget(cell)

        optics = QGroupBox("Optics (normal incidence, ideal polarisers)")
        optics_form = QFormLayout(optics)
        self.optics_group = QButtonGroup(self)
        optics_column = QVBoxLayout()
        for mode_id, mode in enumerate(fm.OPTICS_MODES):
            radio = QRadioButton(OPTICS_MODE_LABELS[mode])
            self.optics_group.addButton(radio, mode_id)
            optics_column.addWidget(radio)
        self.optics_group.button(0).setChecked(True)
        self.optics_group.idClicked.connect(self._on_optics_changed)
        optics_form.addRow(optics_column)
        self.w["polarizer"] = SliderSpin(-90.0, 90.0, 0.0, decimals=1, suffix="°")
        self.w["polarizer"].valueChanged.connect(self._on_optics_changed)
        optics_form.addRow("Polariser azimuth", self.w["polarizer"])
        for key, label, minimum, maximum, value, decimals, suffix in (
            ("wavelength", "Wavelength", 380.0, 800.0, 550.0, 0, " nm"),
            ("n_o", "n_o", 1.0, 2.5, 1.52, 3, ""),
            ("delta_n", "Δn", -0.5, 0.8, 0.22, 3, ""),
        ):
            self.w[key] = _spin(minimum, maximum, value, decimals, suffix)
            self.w[key].valueChanged.connect(self._on_optics_changed)
            optics_form.addRow(label, self.w[key])
        layout.addWidget(optics)

        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.info)
        layout.addStretch(1)
        return panel

    def _read_cell_params(self):
        w = self.w
        return fm.CellParams(
            k11=w["k11"].value() * 1e-12,
            k22=w["k22"].value() * 1e-12,
            k33=w["k33"].value() * 1e-12,
            eps_perp=w["eps_perp"].value(),
            delta_eps=w["delta_eps"].value(),
            gamma1=w["gamma1"].value(),
            thickness=w["thickness"].value() * 1e-6,
            pitch=w["pitch"].value() * 1e-6,
            twist=np.radians(w["twist"].value()),
            pretilt_bottom=np.radians(w["pretilt_bottom"].value()),
            pretilt_top=np.radians(w["pretilt_top"].value()),
            azimuth_bottom=np.radians(w["azimuth_bottom"].value()),
            n_nodes=int(w["n_nodes"].value()),
        )

    def _read_optics_params(self):
        return fm.OpticsParams(
            wavelength=self.w["wavelength"].value() * 1e-9,
            n_o=self.w["n_o"].value(),
            delta_n=self.w["delta_n"].value(),
            polarizer_angle=np.radians(self.w["polarizer"].value()),
            mode=fm.OPTICS_MODES[self.optics_group.checkedId()],
        )

    @property
    def is_live(self):
        return self.mode_group.checkedId() == 0

    # ----------------------------------------------------------------- callbacks

    def _apply_preset(self, name):
        preset = PRESETS[name]
        self._suspend_updates = True
        for key, value in preset.items():
            if key == "optics_mode":
                self.optics_group.button(fm.OPTICS_MODES.index(value)).setChecked(True)
            else:
                self.w[key].setValue(value)
        self._suspend_updates = False
        self.params = self._read_cell_params()
        self.optics = self._read_optics_params()
        self._reset_state()
        self._compute_vt_curve()
        self._relayout_2d()

    def _on_cell_changed(self, *_):
        if self._suspend_updates:
            return
        self.params = self._read_cell_params()
        self.theta, self.phi = fm.resample_state(self.theta, self.phi, self.params)
        self.vt_curve = None
        if not self.is_live:
            self._snap_to_equilibrium()
        self._redraw()

    def _on_optics_changed(self, *_):
        if self._suspend_updates:
            return
        self.optics = self._read_optics_params()
        self.vt_curve = None
        self._redraw()

    def _on_voltage_changed(self, *_):
        if self._suspend_updates:
            return
        if not self.is_live:
            self._snap_to_equilibrium()

    def _on_mode_changed(self, *_):
        if not self.is_live:
            self._snap_to_equilibrium()

    def _reset_state(self):
        self.theta, self.phi = fm.initial_state(self.params)
        self.theta, self.phi = fm.equilibrium(self.theta, self.phi, self.params, 0.0)
        self.sim_time = 0.0
        self.history.clear()
        if not self.is_live:
            self._snap_to_equilibrium()
        self._redraw()

    def _snap_to_equilibrium(self):
        self.theta, self.phi = fm.equilibrium(
            self.theta, self.phi, self.params, self._applied_voltage()
        )
        self._redraw()

    def _compute_vt_curve(self):
        QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            voltages = np.linspace(0.0, self.w["vt_max"].value(), 81)
            self.vt_curve = (
                voltages,
                fm.voltage_sweep(self.params, self.optics, voltages),
            )
        finally:
            QGuiApplication.restoreOverrideCursor()
        self._redraw()

    def _set_view(self, elev, azim):
        self.ax_3d.view_init(elev=elev, azim=azim)
        self.canvas_3d.draw_idle()

    def _relayout_2d(self):
        self.canvas_2d.figure.set_layout_engine("constrained")
        self.canvas_2d.draw_idle()

    # ------------------------------------------------------------------ dynamics

    def _drive_segment(self, remaining):
        """Return (voltage, segment duration) for the drive at the current time."""
        voltage = self.w["voltage"].value()
        if not self.square_wave.isChecked():
            return voltage, remaining
        period = self.w["period"].value() * 1e-3
        phase = self.sim_time % period
        if phase < period / 2:
            until_edge = period / 2 - phase
            level = voltage
        else:
            until_edge = period - phase
            level = 0.0
        # Avoid zero-length segments from floating-point residue at an edge.
        return level, min(remaining, max(until_edge, 1e-9 * period))

    def _applied_voltage(self):
        return self._drive_segment(1.0)[0]

    def _tick(self):
        now = time.perf_counter()
        real_dt = min(now - self._last_tick, 0.1)
        self._last_tick = now
        if not self.is_live:
            return
        remaining = self.w["speed"].value() * real_dt
        steps_total = 0
        while remaining > 0 and steps_total < MAX_STEPS_PER_FRAME:
            voltage, segment = self._drive_segment(remaining)
            self.theta, self.phi, advanced, steps = fm.relax_dynamics(
                self.theta,
                self.phi,
                self.params,
                voltage,
                segment,
                MAX_STEPS_PER_FRAME - steps_total,
            )
            self.sim_time += advanced
            remaining -= advanced
            steps_total += steps
            if steps == 0:
                break
        self.last_steps = steps_total
        self.speed_limited = remaining > 1e-12
        self.history.append(
            (
                self.sim_time,
                self._applied_voltage(),
                fm.optical_output(self.theta, self.phi, self.params, self.optics),
            )
        )
        self._redraw()

    # ------------------------------------------------------------------ plotting

    def _setup_3d_axes(self):
        fig = self.canvas_3d.figure
        fig.subplots_adjust(left=0.0, right=0.88, bottom=0.0, top=0.95)
        ax = fig.add_subplot(projection="3d")
        self.ax_3d = ax
        ax.set_xlim(-VIEW_HALF_WIDTH, VIEW_HALF_WIDTH)
        ax.set_ylim(-VIEW_HALF_WIDTH, VIEW_HALF_WIDTH)
        ax.set_zlim(-0.04, 1.04)
        # Equal data aspect so that drawn rod angles are the true director angles.
        ax.set_box_aspect((2 * VIEW_HALF_WIDTH, 2 * VIEW_HALF_WIDTH, 1.08))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([0, 0.5, 1])
        ax.set_zticklabels(["0", "d/2", "d"])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.view_init(elev=18, azim=-55)

        corners = VIEW_HALF_WIDTH * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        for z in (0.0, 1.0):
            ax.add_collection3d(
                Poly3DCollection(
                    [np.column_stack([corners, np.full(4, z)])],
                    facecolor="0.6",
                    alpha=0.15,
                    edgecolor="0.4",
                )
            )
        (self.rub_bottom,) = ax.plot([], [], [], color="k", lw=2)
        (self.rub_top,) = ax.plot([], [], [], color="k", lw=2)

        self.tilt_cmap = matplotlib.colormaps["coolwarm"]
        self.tilt_norm = Normalize(0, 90)
        # Placeholder segment: add_collection3d cannot handle an empty collection.
        self.rods = Line3DCollection([np.zeros((2, 3))], linewidths=5, capstyle="round")
        ax.add_collection3d(self.rods, autolim=False)
        fig.colorbar(
            ScalarMappable(norm=self.tilt_norm, cmap=self.tilt_cmap),
            ax=ax,
            shrink=0.5,
            label="|tilt| from substrate (deg)",
        )
        offsets = COLUMN_SPACING * np.array([-1, 0, 1])
        grid_x, grid_y = np.meshgrid(offsets, offsets)
        self.column_offsets = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    def _setup_2d_axes(self):
        fig = self.canvas_2d.figure
        (ax_angles, ax_energy), (ax_vt, ax_time) = fig.subplots(2, 2)
        self.ax_angles, self.ax_energy, self.ax_vt, self.ax_time = (
            ax_angles,
            ax_energy,
            ax_vt,
            ax_time,
        )

        (self.line_theta,) = ax_angles.plot([], [], label="tilt θ")
        (self.line_phi,) = ax_angles.plot([], [], label="azimuth φ")
        ax_angles.set_xlabel("z (µm)")
        ax_angles.set_ylabel("angle (deg)")
        ax_angles.set_title("Director profile")
        ax_angles.legend(loc="best")
        ax_angles.grid(alpha=0.3)

        self.energy_lines = {
            key: ax_energy.plot([], [], label=key)[0]
            for key in ("splay", "twist", "bend")
        }
        ax_energy.set_xlabel("z (µm)")
        ax_energy.set_ylabel("J/m³")
        ax_energy.set_title("Frank distortion energy density")
        ax_energy.legend(loc="upper right")
        ax_energy.grid(alpha=0.3)

        (self.line_vt,) = ax_vt.plot([], [], color="C0")
        self.vt_marker = ax_vt.axvline(0.0, color="C3", lw=1)
        (self.vt_point,) = ax_vt.plot([], [], "o", color="C3")
        ax_vt.set_xlabel("RMS voltage (V)")
        ax_vt.set_ylabel("optical output")
        ax_vt.set_title("Quasi-static V-T curve")
        ax_vt.set_ylim(-0.02, 1.02)
        ax_vt.grid(alpha=0.3)

        (self.line_output,) = ax_time.plot([], [], color="C0", label="output")
        ax_time.set_ylim(-0.02, 1.02)
        ax_time.set_xlabel("simulated time (ms)")
        ax_time.set_ylabel("optical output", color="C0")
        ax_time.set_title("Switching dynamics")
        ax_time.grid(alpha=0.3)
        self.ax_time_voltage = ax_time.twinx()
        (self.line_voltage,) = self.ax_time_voltage.plot(
            [], [], color="C1", lw=1, drawstyle="steps-post"
        )
        self.ax_time_voltage.set_ylabel("applied V", color="C1")

    def _director_vectors(self, theta, phi):
        return np.column_stack(
            [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)]
        )

    def _redraw(self, throttle_2d=False):
        """Update all plots. `throttle_2d` caps the slow 2D canvas to ~8 fps."""
        p = self.params
        theta, phi = self.theta, self.phi
        z_um = p.z * 1e6
        voltage = self._applied_voltage()
        output = fm.optical_output(theta, phi, p, self.optics)

        # 3D rods: resample the profile onto a fixed number of drawn layers.
        s_layers = np.linspace(0.0, 1.0, N_ROD_LAYERS)
        s_nodes = np.linspace(0.0, 1.0, p.n_nodes)
        theta_layers = np.interp(s_layers, s_nodes, theta)
        directors = self._director_vectors(
            theta_layers, np.interp(s_layers, s_nodes, phi)
        )
        centres = np.concatenate(
            [
                np.broadcast_to(
                    self.column_offsets[:, None, :],
                    (len(self.column_offsets), N_ROD_LAYERS, 2),
                ),
                np.broadcast_to(
                    s_layers[None, :, None], (len(self.column_offsets), N_ROD_LAYERS, 1)
                ),
            ],
            axis=-1,
        )
        half_rod = 0.5 * ROD_LENGTH * directors[None, :, :]
        self.rods.set_segments(
            np.stack([centres - half_rod, centres + half_rod], axis=2).reshape(-1, 2, 3)
        )
        self.rods.set_color(
            np.tile(
                self.tilt_cmap(self.tilt_norm(np.degrees(np.abs(theta_layers)))),
                (len(self.column_offsets), 1),
            )
        )
        arrow = 1.6 * COLUMN_SPACING
        for line, azimuth, z in (
            (self.rub_bottom, p.azimuth_bottom, 0.0),
            (self.rub_top, p.azimuth_bottom + p.twist, 1.0),
        ):
            line.set_data_3d(
                [-arrow * np.cos(azimuth), arrow * np.cos(azimuth)],
                [-arrow * np.sin(azimuth), arrow * np.sin(azimuth)],
                [z, z],
            )
        self.ax_3d.set_title(f"Director field, V = {voltage:.2f} V (rubbing: black)")
        self.canvas_3d.draw_idle()

        # Profiles and energy densities.
        self.line_theta.set_data(z_um, np.degrees(theta))
        self.line_phi.set_data(z_um, np.degrees(phi))
        densities = fm.energy_densities(theta, phi, p, voltage)
        for key, line in self.energy_lines.items():
            line.set_data(densities["z_links"] * 1e6, densities[key])
        for ax in (self.ax_angles, self.ax_energy):
            ax.relim()
            ax.autoscale_view()

        if self.vt_curve is None:
            self.line_vt.set_data([], [])
        else:
            self.line_vt.set_data(*self.vt_curve)
            self.ax_vt.set_xlim(0.0, self.vt_curve[0][-1])
        self.vt_marker.set_xdata([voltage, voltage])
        self.vt_point.set_data([voltage], [output])

        if self.history:
            times, voltages, outputs = np.array(self.history).T
            times_ms = times * 1e3
            self.line_output.set_data(times_ms, outputs)
            self.line_voltage.set_data(times_ms, voltages)
            self.ax_time.set_xlim(times_ms[0], max(times_ms[-1], times_ms[0] + 1e-6))
            self.ax_time_voltage.set_ylim(-0.05, max(voltages.max(), 0.1) * 1.1)
        else:
            self.line_output.set_data([], [])
            self.line_voltage.set_data([], [])
        self.canvas_2d.draw_idle()

        self._update_info(voltage, output, densities)
        now = time.perf_counter()
        if throttle_2d and now - self._last_2d_draw < 0.12:
            return
        self._last_2d_draw = now

    def _update_info(self, voltage, output, densities):
        p = self.params
        threshold = fm.freedericksz_threshold(p)
        threshold_text = (
            f"{threshold:.3f} V"
            if threshold is not None
            else "n/a for this geometry / sign of Δε"
        )
        h = p.h
        totals = {
            key: h * densities[key].sum() * 1e6 for key in ("splay", "twist", "bend")
        }
        speed_note = " (limited by step cap)" if self.speed_limited else ""
        mid = p.n_nodes // 2
        self.info.setText(
            f"<b>Analytic Fréedericksz threshold</b> (zero pretilt): {threshold_text}"
            f"<br><b>Optical output</b>: {output:.4f}"
            f"<br><b>Mid-plane</b>: θ = {np.degrees(self.theta[mid]):.2f}°, "
            f"φ = {np.degrees(self.phi[mid]):.2f}°"
            f"<br><b>E_z range</b>: {densities['e_z'].min() * 1e-6:.3f} to "
            f"{densities['e_z'].max() * 1e-6:.3f} V/µm"
            f"<br><b>Elastic energy</b> (µJ/m²): splay {totals['splay']:.3f}, "
            f"twist {totals['twist']:.3f}, bend {totals['bend']:.3f}"
            f"<br><b>Sim time</b>: {self.sim_time * 1e3:.2f} ms, "
            f"{self.last_steps} steps/frame{speed_note}"
        )


def main():
    app = QApplication(sys.argv)
    window = LCWindow()
    window.resize(1700, 950)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
