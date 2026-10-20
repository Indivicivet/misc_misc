"""Rendering engine for banana cross-sections.

Supports:
- High-fidelity semi-photorealistic 2D slice rendering through a full 360-degree
  two-layer ring of bananas tightly packed around the central stem.
- Professional botanical & geometric analysis diagram:
  - Exact facet polygon chords connecting verified rib vertices.
  - Authentic side counts (3, 4, 5 sides) based on trilocular ovary ground state + contact planes.
  - Active contact tangents showing exactly which faces are flattened by neighbor compression.
  - Clean, un-sloppy technical typography and legend.
- High-resolution rendering to QImage / PNG.
"""

from typing import Tuple
import math
import numpy as np

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import (
    QPainter,
    QPainterPath,
    QColor,
    QBrush,
    QPen,
    QRadialGradient,
    QLinearGradient,
    QFont,
    QImage,
)

from geometry import FullRingGeometry, BananaFinger, CentralStem


def create_path_from_points(pts: np.ndarray) -> QPainterPath:
    """Build a closed QPainterPath from an (N, 2) array of coordinates."""
    path = QPainterPath()
    if len(pts) == 0:
        return path
    path.moveTo(float(pts[0, 0]), float(pts[0, 1]))
    for i in range(1, len(pts)):
        path.lineTo(float(pts[i, 0]), float(pts[i, 1]))
    path.closeSubpath()
    return path


def draw_photorealistic_slice(
    painter: QPainter,
    geom: FullRingGeometry,
    view_rect: QRectF,
) -> None:
    """Render the semi-photorealistic botanical slice of the full 360-degree ring."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    # 1. Dark slate background
    bg_grad = QRadialGradient(
        view_rect.center(),
        max(view_rect.width(), view_rect.height()) * 0.70,
    )
    bg_grad.setColorAt(0.0, QColor(24, 28, 34))
    bg_grad.setColorAt(0.65, QColor(14, 16, 20))
    bg_grad.setColorAt(1.0, QColor(8, 10, 12))
    painter.fillRect(view_rect, bg_grad)

    painter.setPen(QPen(QColor(255, 255, 255, 5), 1.0))
    grid_step = 25.0
    x_curr = view_rect.left()
    while x_curr < view_rect.right():
        painter.drawLine(
            QPointF(x_curr, view_rect.top()), QPointF(x_curr, view_rect.bottom())
        )
        x_curr += grid_step
    y_curr = view_rect.top()
    while y_curr < view_rect.bottom():
        painter.drawLine(
            QPointF(view_rect.left(), y_curr), QPointF(view_rect.right(), y_curr)
        )
        y_curr += grid_step

    # 2. Central Rachis / Stem
    stem = geom.stem
    stem_path = create_path_from_points(stem.boundary_pts)

    painter.save()
    painter.translate(2.5, 3.5)
    painter.fillPath(stem_path, QColor(0, 0, 0, 110))
    painter.restore()

    stem_grad = QRadialGradient(
        QPointF(stem.center[0], stem.center[1]),
        stem.radius * 1.05,
    )
    stem_grad.setColorAt(0.0, QColor(228, 238, 205))
    stem_grad.setColorAt(0.70, QColor(212, 226, 182))
    stem_grad.setColorAt(0.92, QColor(155, 185, 115))
    stem_grad.setColorAt(1.0, QColor(80, 110, 50))
    painter.fillPath(stem_path, stem_grad)
    painter.strokePath(stem_path, QPen(QColor(55, 80, 35), 2.0))

    # Stem monocot vascular bundles
    for v_pt in stem.vascular_bundles:
        painter.setBrush(QBrush(QColor(68, 95, 45, 215)))
        painter.setPen(QPen(QColor(45, 68, 30, 235), 0.6))
        painter.drawEllipse(QPointF(float(v_pt[0]), float(v_pt[1])), 1.3, 1.3)
        painter.setBrush(QBrush(QColor(238, 246, 218, 210)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(float(v_pt[0]), float(v_pt[1])), 0.5, 0.5)

    # 3. Banana Fingers
    sorted_fingers = sorted(
        geom.fingers, key=lambda f: (0 if f.tier == "inner" else 1, f.id)
    )

    for finger in sorted_fingers:
        outer_path = create_path_from_points(finger.peel_outer_pts)
        painter.save()
        painter.translate(3.0, 4.0)
        painter.fillPath(outer_path, QColor(0, 0, 0, 100))
        painter.restore()

    for finger in sorted_fingers:
        outer_path = create_path_from_points(finger.peel_outer_pts)
        inner_path = create_path_from_points(finger.peel_inner_pts)

        # 3a. Peel (Pericarp)
        peel_ring = outer_path.subtracted(inner_path)

        peel_grad = QRadialGradient(
            QPointF(finger.center[0], finger.center[1]),
            finger.radius * 1.06,
        )
        peel_grad.setColorAt(0.66, QColor(252, 250, 238))
        peel_grad.setColorAt(0.78, QColor(254, 246, 192))
        peel_grad.setColorAt(0.89, QColor(255, 226, 44))
        peel_grad.setColorAt(0.97, QColor(244, 206, 24))
        peel_grad.setColorAt(1.00, QColor(165, 162, 40))

        painter.fillPath(peel_ring, peel_grad)
        painter.strokePath(outer_path, QPen(QColor(118, 112, 28), 1.2))

        # 3b. Phloem Vascular Bundles
        for v_pt in finger.vascular_bundles:
            painter.setBrush(QBrush(QColor(248, 244, 222)))
            painter.setPen(QPen(QColor(190, 178, 128), 0.6))
            painter.drawEllipse(QPointF(float(v_pt[0]), float(v_pt[1])), 0.75, 0.75)

        # 3c. Endocarp Pulp
        pulp_grad = QRadialGradient(
            QPointF(finger.center[0] - 1.2, finger.center[1] - 1.2),
            finger.radius * 0.82,
        )
        pulp_grad.setColorAt(0.0, QColor(255, 254, 246))
        pulp_grad.setColorAt(0.35, QColor(254, 251, 238))
        pulp_grad.setColorAt(0.75, QColor(249, 244, 222))
        pulp_grad.setColorAt(0.96, QColor(242, 234, 202))
        pulp_grad.setColorAt(1.00, QColor(230, 220, 184))

        painter.fillPath(inner_path, pulp_grad)
        painter.strokePath(inner_path, QPen(QColor(216, 206, 170), 0.85))

        # 3d. Radial Starchy Parenchyma Grain
        painter.setPen(QPen(QColor(218, 208, 178, 32), 0.75))
        n_rays = 48
        r_inner = 1.8
        r_outer = finger.radius * 0.70
        for ray_i in range(n_rays):
            r_ang = ray_i * (2 * math.pi / n_rays) + finger.rotation
            p1 = QPointF(
                finger.center[0] + r_inner * math.cos(r_ang),
                finger.center[1] + r_inner * math.sin(r_ang),
            )
            p2 = QPointF(
                finger.center[0] + r_outer * math.cos(r_ang),
                finger.center[1] + r_outer * math.sin(r_ang),
            )
            painter.drawLine(p1, p2)

        # 3e. Trilocular Carpel Sutures
        painter.setPen(QPen(QColor(196, 184, 146, 110), 1.1))
        for loc_ang in finger.locule_angles:
            suture_end = QPointF(
                finger.center[0] + (finger.radius * 0.72) * math.cos(loc_ang),
                finger.center[1] + (finger.radius * 0.72) * math.sin(loc_ang),
            )
            painter.drawLine(
                QPointF(finger.center[0], finger.center[1]),
                suture_end,
            )

        # 3f. Central Placenta Core
        painter.setBrush(QBrush(QColor(212, 196, 160, 85)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(finger.center[0], finger.center[1]), 1.6, 1.6)

        # 3g. Aborted Ovules vs Wild Seeds
        for s_pos, s_r in zip(finger.ovule_positions, finger.ovule_radii):
            s_pt = QPointF(float(s_pos[0]), float(s_pos[1]))
            if geom.seed_morphology < 0.15:
                painter.setBrush(QBrush(QColor(48, 24, 10, 220)))
                painter.setPen(QPen(QColor(26, 12, 5, 150), 0.35))
                painter.drawEllipse(s_pt, float(s_r), float(s_r))
            else:
                s_grad = QRadialGradient(s_pt, float(s_r) * 1.3)
                s_grad.setColorAt(0.0, QColor(64, 38, 22))
                s_grad.setColorAt(0.65, QColor(28, 15, 7))
                s_grad.setColorAt(1.0, QColor(10, 5, 2))
                painter.setBrush(s_grad)
                painter.setPen(QPen(QColor(6, 3, 1), 0.8))
                painter.drawEllipse(s_pt, float(s_r), float(s_r * 0.88))
                if float(s_r) > 1.0:
                    painter.setBrush(QBrush(QColor(170, 145, 110, 180)))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawEllipse(
                        QPointF(s_pt.x() - 0.3 * s_r, s_pt.y() - 0.2 * s_r),
                        0.35 * s_r,
                        0.35 * s_r,
                    )

        # 3h. Cut Surface Specular Sheen
        sheen_grad = QLinearGradient(
            QPointF(finger.center[0] - 5, finger.center[1] - 5),
            QPointF(finger.center[0] + 5, finger.center[1] + 5),
        )
        sheen_grad.setColorAt(0.0, QColor(255, 255, 255, 50))
        sheen_grad.setColorAt(0.5, QColor(255, 255, 255, 10))
        sheen_grad.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.setBrush(sheen_grad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(
            QPointF(finger.center[0] - 1.2, finger.center[1] - 1.2),
            finger.radius * 0.42,
            finger.radius * 0.32,
        )

    painter.restore()


def draw_structural_diagram(
    painter: QPainter,
    geom: FullRingGeometry,
    view_rect: QRectF,
) -> None:
    """Render technical scientific / botanical analysis diagram."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    # 1. Clean technical dark blueprint background
    painter.fillRect(view_rect, QColor(14, 18, 24))

    # Engineering grid
    painter.setPen(QPen(QColor(36, 48, 64, 80), 1.0))
    grid_step = 25.0
    x_curr = view_rect.left()
    while x_curr < view_rect.right():
        painter.drawLine(
            QPointF(x_curr, view_rect.top()), QPointF(x_curr, view_rect.bottom())
        )
        x_curr += grid_step
    y_curr = view_rect.top()
    while y_curr < view_rect.bottom():
        painter.drawLine(
            QPointF(view_rect.left(), y_curr), QPointF(view_rect.right(), y_curr)
        )
        y_curr += grid_step

    # 2. Central Rachis Stem
    stem = geom.stem
    stem_path = create_path_from_points(stem.boundary_pts)
    painter.strokePath(
        stem_path,
        QPen(QColor(75, 160, 100, 180), 1.5, Qt.PenStyle.DashDotLine),
    )

    painter.setPen(QColor(115, 195, 135))
    stem_font = QFont("Consolas")
    stem_font.setPointSizeF(4.0)
    stem_font.setBold(True)
    painter.setFont(stem_font)
    painter.drawText(
        QRectF(stem.center[0] - 25, stem.center[1] - 5, 50, 10),
        Qt.AlignmentFlag.AlignCenter,
        "RACHIS AXIS",
    )

    # 3. Banana fingers structural analysis
    for finger in geom.fingers:
        outer_pts = finger.peel_outer_pts
        n_pts = len(outer_pts)
        rib_idx = finger.rib_vertex_indices

        # 3a. Draw the true smooth organic contour
        outer_path = create_path_from_points(outer_pts)
        contour_color = (
            QColor(50, 140, 215, 160)
            if finger.tier == "outer"
            else QColor(165, 95, 215, 160)
        )
        painter.strokePath(outer_path, QPen(contour_color, 1.4))

        # 3b. Draw the Facet Chords connecting verified rib vertices
        # This makes the polygon edges completely unmistakable: 3 edges, 4 edges, or 5 edges!
        if len(rib_idx) >= 3:
            chord_path = QPainterPath()
            p0 = QPointF(
                float(outer_pts[rib_idx[0], 0]), float(outer_pts[rib_idx[0], 1])
            )
            chord_path.moveTo(p0)
            for k in range(1, len(rib_idx)):
                pk = QPointF(
                    float(outer_pts[rib_idx[k], 0]), float(outer_pts[rib_idx[k], 1])
                )
                chord_path.lineTo(pk)
            chord_path.closeSubpath()
            painter.strokePath(
                chord_path,
                QPen(QColor(255, 255, 255, 80), 1.0, Qt.PenStyle.DashLine),
            )

        # 3c. Trilocular Carpel Axes (3-carpel botanical symmetry at 120 deg)
        painter.setPen(QPen(QColor(80, 200, 240, 120), 1.0, Qt.PenStyle.DotLine))
        for loc_ang in finger.locule_angles:
            s_end = QPointF(
                finger.center[0] + (finger.radius * 0.70) * math.cos(loc_ang),
                finger.center[1] + (finger.radius * 0.70) * math.sin(loc_ang),
            )
            painter.drawLine(
                QPointF(finger.center[0], finger.center[1]),
                s_end,
            )

        # 3d. Rib Vertex Markers (sharp crosshairs / dots on the verified corners)
        for r_i in rib_idx:
            r_pt = QPointF(float(outer_pts[r_i, 0]), float(outer_pts[r_i, 1]))
            painter.setBrush(QBrush(QColor(255, 65, 50)))
            painter.setPen(QPen(QColor(255, 255, 255), 1.0))
            painter.drawEllipse(r_pt, 2.8, 2.8)

        # 3e. Clean Technical Typography: Facet Count
        painter.setPen(QColor(245, 248, 255))
        font = QFont("Consolas")
        font.setPointSizeF(3.6)
        font.setBold(True)
        painter.setFont(font)
        poly_name = f"{finger.facet_count}-gon"
        painter.drawText(
            QRectF(finger.center[0] - 12, finger.center[1] - 4, 24, 8),
            Qt.AlignmentFlag.AlignCenter,
            poly_name,
        )

    painter.restore()


def draw_structural_legend(painter: QPainter, target_rect: QRectF) -> None:
    """Render scientific technical legend in screen space."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    legend_w = 285.0
    legend_h = 165.0
    box = QRectF(
        target_rect.left() + 20,
        target_rect.top() + 20,
        legend_w,
        legend_h,
    )
    painter.setBrush(QBrush(QColor(14, 18, 24, 235)))
    painter.setPen(QPen(QColor(50, 68, 90), 1.5))
    painter.drawRoundedRect(box, 6.0, 6.0)

    painter.setPen(QColor(240, 246, 255))
    painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
    painter.drawText(
        QRectF(box.left() + 14, box.top() + 10, 255, 20),
        Qt.AlignmentFlag.AlignLeft,
        "Musa Fruit Morphology Analysis",
    )

    painter.setFont(QFont("Segoe UI", 8))

    # 3-carpel baseline explanation
    painter.setPen(QColor(100, 210, 255))
    painter.drawText(
        QPointF(box.left() + 14, box.top() + 44),
        "• Ground State: 3 Carpels (Trilocular syncarpous)",
    )

    painter.setPen(QColor(210, 220, 235))
    painter.drawText(
        QPointF(box.left() + 14, box.top() + 64),
        "• Pressure Compaction: flattens faces into 4~5 sides",
    )

    # Red dot: rib vertex
    painter.setBrush(QBrush(QColor(255, 65, 50)))
    painter.setPen(QPen(QColor(255, 255, 255), 1.0))
    painter.drawEllipse(QPointF(box.left() + 20, box.top() + 86), 3.5, 3.5)
    painter.setPen(QColor(210, 220, 235))
    painter.drawText(
        QPointF(box.left() + 32, box.top() + 90),
        "Verified Rib Vertices (corners)",
    )

    # Dashed white line: facet chord
    painter.setPen(QPen(QColor(255, 255, 255, 120), 1.5, Qt.PenStyle.DashLine))
    painter.drawLine(
        QPointF(box.left() + 14, box.top() + 112),
        QPointF(box.left() + 28, box.top() + 112),
    )
    painter.setPen(QColor(210, 220, 235))
    painter.drawText(
        QPointF(box.left() + 32, box.top() + 116),
        "Facet Chords (flat contact planes)",
    )

    # Cyan dotted rays: 3 locule sutures
    painter.setPen(QPen(QColor(80, 200, 240, 160), 1.5, Qt.PenStyle.DotLine))
    painter.drawLine(
        QPointF(box.left() + 14, box.top() + 138),
        QPointF(box.left() + 28, box.top() + 138),
    )
    painter.setPen(QColor(210, 220, 235))
    painter.drawText(
        QPointF(box.left() + 32, box.top() + 142),
        "120° Carpel Locule Septa",
    )

    painter.restore()


def render_to_image(
    geom: FullRingGeometry,
    mode: str = "both",
    width: int = 3840,
    height: int = 2160,
) -> QImage:
    """Render high-resolution image for export."""
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor(10, 12, 16))

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    bounds_radius = max(130.0, 15.0 * max(geom.n_inner, geom.n_outer / 1.5))
    model_dim = bounds_radius * 2.0
    view_rect = QRectF(-bounds_radius, -bounds_radius, model_dim, model_dim)

    if mode == "both":
        half_w = width / 2.0
        # Left viewport: Photorealistic
        painter.save()
        painter.setClipRect(QRectF(0, 0, half_w, height))
        scale = min(half_w / model_dim, height / model_dim) * 0.92
        painter.translate(half_w / 2.0, height / 2.0)
        painter.scale(scale, scale)
        draw_photorealistic_slice(painter, geom, view_rect)
        painter.restore()

        # Right viewport: Structural
        painter.save()
        painter.setClipRect(QRectF(half_w, 0, half_w, height))
        scale = min(half_w / model_dim, height / model_dim) * 0.92
        painter.translate(half_w + half_w / 2.0, height / 2.0)
        painter.scale(scale, scale)
        draw_structural_diagram(painter, geom, view_rect)
        painter.restore()

        draw_structural_legend(painter, QRectF(half_w, 0, half_w, height))

        painter.setPen(QPen(QColor(50, 68, 90), 2.0))
        painter.drawLine(QPointF(half_w, 0), QPointF(half_w, height))

        painter.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        painter.setPen(QColor(240, 245, 255))
        painter.drawText(
            QRectF(30, 25, 450, 35),
            Qt.AlignmentFlag.AlignLeft,
            f"Transverse Slice ({geom.n_inner} In / {geom.n_outer} Out)",
        )
        painter.drawText(
            QRectF(half_w + 310, 25, 450, 35),
            Qt.AlignmentFlag.AlignLeft,
            "Facet Chords & Rib Vertices",
        )

    elif mode == "photorealistic":
        scale = min(width / model_dim, height / model_dim) * 0.94
        painter.translate(width / 2.0, height / 2.0)
        painter.scale(scale, scale)
        draw_photorealistic_slice(painter, geom, view_rect)

    elif mode == "structural":
        scale = min(width / model_dim, height / model_dim) * 0.94
        painter.translate(width / 2.0, height / 2.0)
        painter.scale(scale, scale)
        draw_structural_diagram(painter, geom, view_rect)
        draw_structural_legend(painter, QRectF(0, 0, width, height))

    painter.end()
    return image
