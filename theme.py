from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QPolygon, QFont
from PySide6.QtCore import QPoint

BG = "#0a0a0a"
PANEL = "#141414"
PANEL2 = "#1d1d1d"
TEXT = "#f5f5f5"
DIM = "#9a9a9a"
ACCENT = "#ff2a5f"   # hot pink-red
ACCENT2 = "#ff0000"  # blood red
WARN = "#ffcc00"
OK_GREEN = "#39ff6a"
HOT_RED = "#ff2222"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: "Segoe UI", "Arial Black", sans-serif;
    font-size: 10pt;
}}
QGroupBox {{
    background-color: {PANEL};
    border: 2px solid {ACCENT};
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 10px;
    font-weight: 900;
    letter-spacing: 1px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: {ACCENT};
    background-color: {BG};
}}
QLabel {{ color: {TEXT}; }}
QLabel#femTitle {{
    color: {ACCENT};
    font-size: 22pt;
    font-weight: 900;
    letter-spacing: 3px;
}}
QLabel#femSub {{
    color: {DIM};
    font-size: 9pt;
    letter-spacing: 2px;
}}
QSlider::groove:horizontal {{
    background: {PANEL2};
    height: 8px;
    border-radius: 4px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    border: 2px solid white;
    width: 18px;
    margin: -6px 0;
    border-radius: 4px;
}}
QSlider::groove:vertical {{
    background: {PANEL2};
    width: 8px;
    border-radius: 4px;
}}
QSlider::handle:vertical {{
    background: {ACCENT};
    border: 2px solid white;
    height: 18px;
    margin: 0 -6px;
    border-radius: 4px;
}}
QDoubleSpinBox, QComboBox {{
    background-color: {PANEL2};
    color: {TEXT};
    border: 1px solid {ACCENT};
    border-radius: 4px;
    padding: 4px;
}}
QPushButton {{
    background-color: {ACCENT};
    color: white;
    font-weight: 900;
    border: none;
    border-radius: 4px;
    padding: 8px 12px;
}}
QPushButton:hover {{ background-color: {ACCENT2}; }}
QPushButton:checked {{ background-color: white; color: black; }}
QPushButton#presetBtn {{
    background-color: {PANEL2};
    border: 2px solid {ACCENT};
    color: {TEXT};
    padding: 6px 8px;
}}
QPushButton#presetBtn:checked {{
    background-color: {ACCENT};
    color: white;
}}
QCheckBox {{ color: {TEXT}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 2px solid {ACCENT};
    border-radius: 3px;
    background: {PANEL2};
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; }}
QStatusBar {{ background-color: {PANEL}; color: {DIM}; }}
QMenu {{ background-color: {PANEL}; color: {TEXT}; }}
"""


def apply_theme(app):
    app.setStyleSheet(STYLESHEET)


def title_font():
    f = QFont("Arial Black", 22)
    f.setBold(True)
    f.setLetterSpacing(QFont.PercentageSpacing, 110)
    return f


def make_icon():
    pm = QPixmap(64, 64)
    pm.fill(QColor(BG))
    p = QPainter(pm)
    # blood-red backdrop slash
    p.setPen(QColor(ACCENT2))
    p.setBrush(QColor(ACCENT2))
    p.drawRect(4, 50, 56, 6)
    # white speaker box + pink spike (cute-brutal)
    p.setBrush(QColor("white"))
    p.setPen(QColor("white"))
    p.drawRect(10, 24, 15, 16)
    p.drawPolygon(QPolygon([QPoint(25, 24), QPoint(42, 10), QPoint(42, 54), QPoint(25, 40)]))
    p.setPen(QColor(ACCENT))
    p.setBrush(QColor(ACCENT))
    p.drawPolygon(QPolygon([QPoint(42, 10), QPoint(52, 4), QPoint(46, 22)]))
    p.end()
    return QIcon(pm)
