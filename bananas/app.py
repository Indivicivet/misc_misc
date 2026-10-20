"""Interactive PySide6 Desktop Application for the Banana Hand / Ring Simulator.

Controls:
- Inner ring finger count slider (6 to 12)
- Outer ring finger count slider (8 to 18)
- Packing compression slider (modulating facet flattening vs rounded 3-sided relaxation)
- Seed morphology slider (commercial aborted ovules to wild-type seeds)
- Randomize variance & jiggle
- Synchronized side-by-side photorealistic and structural viewports
- High-resolution 2K/4K PNG image export
"""

import sys
from pathlib import Path
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QImage, QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QPushButton,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QMessageBox,
    QFrame,
)

from geometry import FullRingGeometry, generate_full_ring_geometry
from renderer import (
    draw_photorealistic_slice,
    draw_structural_diagram,
    draw_structural_legend,
    render_to_image,
)


class BananaViewport(QWidget):
    """Custom canvas widget rendering either photorealistic, structural, or side-by-side."""

    def __init__(self, mode: str = "both", parent=None):
        super().__init__(parent)
        self.mode = mode
        self.geom: FullRingGeometry = None
        self.setMinimumSize(500, 500)

    def set_geometry(self, geom: FullRingGeometry) -> None:
        self.geom = geom
        self.update()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        w = self.width()
        h = self.height()
        painter.fillRect(0, 0, w, h, QColor(10, 12, 16))

        if self.geom is None:
            painter.end()
            return

        bounds_r = max(130.0, 15.0 * max(self.geom.n_inner, self.geom.n_outer / 1.5))
        model_dim = bounds_r * 2.0
        view_rect = QRectF(-bounds_r, -bounds_r, model_dim, model_dim)

        if self.mode == "both":
            half_w = w / 2.0

            # Left: Photorealistic
            painter.save()
            painter.setClipRect(QRectF(0, 0, half_w, h))
            scale = min(half_w / model_dim, h / model_dim) * 0.92
            painter.translate(half_w / 2.0, h / 2.0)
            painter.scale(scale, scale)
            draw_photorealistic_slice(painter, self.geom, view_rect)
            painter.restore()

            # Right: Structural
            painter.save()
            painter.setClipRect(QRectF(half_w, 0, half_w, h))
            scale = min(half_w / model_dim, h / model_dim) * 0.92
            painter.translate(half_w + half_w / 2.0, h / 2.0)
            painter.scale(scale, scale)
            draw_structural_diagram(painter, self.geom, view_rect)
            painter.restore()

            draw_structural_legend(painter, QRectF(half_w, 0, half_w, h))

            painter.setPen(QColor(50, 68, 90))
            painter.drawLine(int(half_w), 0, int(half_w), h)

            painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            painter.setPen(QColor(240, 245, 255))
            painter.drawText(
                20,
                28,
                f"Transverse Slice ({self.geom.n_inner} Inner / {self.geom.n_outer} Outer)",
            )
            painter.drawText(int(half_w) + 310, 28, "Facet Chords & Rib Vertices")

        elif self.mode == "photorealistic":
            scale = min(w / model_dim, h / model_dim) * 0.94
            painter.save()
            painter.translate(w / 2.0, h / 2.0)
            painter.scale(scale, scale)
            draw_photorealistic_slice(painter, self.geom, view_rect)
            painter.restore()

        elif self.mode == "structural":
            scale = min(w / model_dim, h / model_dim) * 0.94
            painter.save()
            painter.translate(w / 2.0, h / 2.0)
            painter.scale(scale, scale)
            draw_structural_diagram(painter, self.geom, view_rect)
            painter.restore()
            draw_structural_legend(painter, QRectF(0, 0, w, h))

        painter.end()


class MainWindow(QMainWindow):
    """Main application window with interactive parameters and high-res export."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Musa Banana Ring Simulator - Botanical Transverse Slice & Morphology"
        )
        self.resize(1400, 860)

        self.n_inner = 8
        self.n_outer = 12
        self.current_compression = 0.90
        self.current_seed_morph = 0.0
        self.random_seed = 101

        self._build_ui()
        self.update_simulation()

    def _build_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Left Viewport
        self.viewport = BananaViewport(mode="both")
        main_layout.addWidget(self.viewport, stretch=4)

        # Right Control Sidebar
        sidebar = QFrame()
        sidebar.setFixedWidth(340)
        sidebar.setStyleSheet(
            "QFrame { background-color: #171b22; border-radius: 8px; }"
            "QLabel { color: #e6ebf2; font-family: 'Segoe UI'; }"
            "QGroupBox { color: #8ec07c; font-weight: bold; border: 1px solid #2e3846; border-radius: 6px; margin-top: 10px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
            "QPushButton { background-color: #243346; color: #ffffff; border-radius: 5px; padding: 8px; font-weight: bold; }"
            "QPushButton:hover { background-color: #354a65; }"
            "QPushButton:pressed { background-color: #1a2533; }"
            "QComboBox { background-color: #1f2732; color: #ffffff; border: 1px solid #364456; border-radius: 4px; padding: 4px 8px; }"
        )
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(14, 14, 14, 14)
        side_layout.setSpacing(12)

        # 1. View Display Mode
        view_box = QGroupBox("View Display Mode")
        v_layout = QVBoxLayout(view_box)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            [
                "Side-by-Side (Both)",
                "Semi-Photorealistic Only",
                "Structural Diagram Only",
            ]
        )
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        v_layout.addWidget(self.mode_combo)
        side_layout.addWidget(view_box)

        # 2. Ring Composition (Inner & Outer Finger Counts)
        count_box = QGroupBox("Ring Finger Count Controls")
        c_layout = QVBoxLayout(count_box)
        c_layout.setSpacing(8)

        # Inner fingers slider
        self.inner_count_label = QLabel(f"Inner Ring Fingers: {self.n_inner}")
        c_layout.addWidget(self.inner_count_label)
        self.inner_slider = QSlider(Qt.Orientation.Horizontal)
        self.inner_slider.setRange(6, 12)
        self.inner_slider.setValue(self.n_inner)
        self.inner_slider.valueChanged.connect(self._on_inner_count_changed)
        c_layout.addWidget(self.inner_slider)

        # Outer fingers slider
        self.outer_count_label = QLabel(f"Outer Ring Fingers: {self.n_outer}")
        c_layout.addWidget(self.outer_count_label)
        self.outer_slider = QSlider(Qt.Orientation.Horizontal)
        self.outer_slider.setRange(8, 18)
        self.outer_slider.setValue(self.n_outer)
        self.outer_slider.valueChanged.connect(self._on_outer_count_changed)
        c_layout.addWidget(self.outer_slider)

        side_layout.addWidget(count_box)

        # 3. Physics & Morphology Controls
        morph_box = QGroupBox("Botany & Physical Squeeze")
        m_layout = QVBoxLayout(morph_box)
        m_layout.setSpacing(8)

        # Compression Slider
        self.comp_label = QLabel(f"Packing Compression: {self.current_compression:.2f}")
        m_layout.addWidget(self.comp_label)
        self.comp_slider = QSlider(Qt.Orientation.Horizontal)
        self.comp_slider.setRange(0, 100)
        self.comp_slider.setValue(int(self.current_compression * 100))
        self.comp_slider.valueChanged.connect(self._on_compression_changed)
        m_layout.addWidget(self.comp_slider)

        comp_desc = QLabel(
            "<small style='color: #8a99ad;'>0.0 = Unpressured 3-sided | 1.0 = Strong 4~5 facets</small>"
        )
        m_layout.addWidget(comp_desc)

        # Seed Slider
        self.seed_label = QLabel("Seed Morphology: Commercial Parthenocarpic")
        m_layout.addWidget(self.seed_label)
        self.seed_slider = QSlider(Qt.Orientation.Horizontal)
        self.seed_slider.setRange(0, 100)
        self.seed_slider.setValue(int(self.current_seed_morph * 100))
        self.seed_slider.valueChanged.connect(self._on_seed_changed)
        m_layout.addWidget(self.seed_slider)

        # Randomize Button
        self.rand_btn = QPushButton("🎲 Jiggle & Re-seed Variance")
        self.rand_btn.clicked.connect(self._on_randomize)
        m_layout.addWidget(self.rand_btn)

        side_layout.addWidget(morph_box)

        # 4. Botanical Statistics
        stats_box = QGroupBox("Live Botanical Statistics")
        s_layout = QVBoxLayout(stats_box)
        s_layout.setSpacing(4)
        self.stat_inner = QLabel()
        self.stat_outer = QLabel()
        self.stat_total = QLabel()
        s_layout.addWidget(self.stat_inner)
        s_layout.addWidget(self.stat_outer)
        s_layout.addWidget(self.stat_total)
        side_layout.addWidget(stats_box)

        # 5. High-Res Export
        export_box = QGroupBox("High-Resolution Export")
        e_layout = QVBoxLayout(export_box)
        e_layout.setSpacing(8)

        self.res_combo = QComboBox()
        self.res_combo.addItems(
            ["4K UHD (3840 x 2160)", "2K QHD (2560 x 1440)", "Full HD (1920 x 1080)"]
        )
        e_layout.addWidget(self.res_combo)

        self.export_mode_combo = QComboBox()
        self.export_mode_combo.addItems(
            [
                "Both Side-by-Side",
                "Photorealistic Slice Only",
                "Structural Diagram Only",
            ]
        )
        e_layout.addWidget(self.export_mode_combo)

        self.export_btn = QPushButton("💾 Export Image (PNG)...")
        self.export_btn.setStyleSheet(
            "QPushButton { background-color: #2b6142; color: #ffffff; padding: 9px; font-size: 13px; }"
            "QPushButton:hover { background-color: #387c55; }"
        )
        self.export_btn.clicked.connect(self._on_export)
        e_layout.addWidget(self.export_btn)

        side_layout.addWidget(export_box)
        side_layout.addStretch()

        main_layout.addWidget(sidebar, stretch=1)

    def update_simulation(self) -> None:
        geom = generate_full_ring_geometry(
            n_inner=self.n_inner,
            n_outer=self.n_outer,
            compression=self.current_compression,
            seed_morphology=self.current_seed_morph,
            random_seed=self.random_seed,
        )
        self.viewport.set_geometry(geom)

        # Update stats
        inner_facets = [f.facet_count for f in geom.fingers if f.tier == "inner"]
        outer_facets = [f.facet_count for f in geom.fingers if f.tier == "outer"]
        avg_in = sum(inner_facets) / max(len(inner_facets), 1)
        avg_out = sum(outer_facets) / max(len(outer_facets), 1)

        self.stat_inner.setText(
            f"• Inner Ring: {len(inner_facets)} fingers (avg {avg_in:.1f} facets)"
        )
        self.stat_outer.setText(
            f"• Outer Ring: {len(outer_facets)} fingers (avg {avg_out:.1f} facets)"
        )
        self.stat_total.setText(f"• Total: {len(geom.fingers)} bananas in 360° ring")

    def _on_mode_changed(self, idx: int) -> None:
        modes = ["both", "photorealistic", "structural"]
        self.viewport.set_mode(modes[idx])

    def _on_inner_count_changed(self, val: int) -> None:
        self.n_inner = val
        self.inner_count_label.setText(f"Inner Ring Fingers: {self.n_inner}")
        self.update_simulation()

    def _on_outer_count_changed(self, val: int) -> None:
        self.n_outer = val
        self.outer_count_label.setText(f"Outer Ring Fingers: {self.n_outer}")
        self.update_simulation()

    def _on_compression_changed(self, val: int) -> None:
        self.current_compression = val / 100.0
        self.comp_label.setText(f"Packing Compression: {self.current_compression:.2f}")
        self.update_simulation()

    def _on_seed_changed(self, val: int) -> None:
        self.current_seed_morph = val / 100.0
        if self.current_seed_morph < 0.15:
            text = "Seed Morphology: Commercial Parthenocarpic"
        elif self.current_seed_morph > 0.75:
            text = "Seed Morphology: Wild Musa (Hard Seeds)"
        else:
            text = (
                f"Seed Morphology: Intermediate ({int(self.current_seed_morph * 100)}%)"
            )
        self.seed_label.setText(text)
        self.update_simulation()

    def _on_randomize(self) -> None:
        import random

        self.random_seed = random.randint(1000, 999999)
        self.update_simulation()

    def _on_export(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Banana Slice Render",
            str(Path.cwd() / "output" / "banana_simulation.png"),
            "PNG Images (*.png);;All Files (*)",
        )
        if not file_path:
            return

        res_idx = self.res_combo.currentIndex()
        if res_idx == 0:
            w, h = 3840, 2160
        elif res_idx == 1:
            w, h = 2560, 1440
        else:
            w, h = 1920, 1080

        mode_idx = self.export_mode_combo.currentIndex()
        modes = ["both", "photorealistic", "structural"]
        export_mode = modes[mode_idx]

        if export_mode != "both":
            w, h = min(w, h), min(w, h)

        img = render_to_image(self.viewport.geom, mode=export_mode, width=w, height=h)
        success = img.save(file_path)

        if success:
            QMessageBox.information(
                self,
                "Export Successful",
                f"Image saved successfully to:\n{file_path}\n({w} x {h} px)",
            )
        else:
            QMessageBox.warning(self, "Export Failed", "Could not save image to disk.")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
