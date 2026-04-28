"""
NightWalk Image Visualizer
===========================
Usage:
    python map_visualizer.py urban-mosaic/washington-square.csv

    The `image` column in the CSV contains paths like:
        0/20160628/device/session/image.jpg
    These are resolved relative to the CSV's parent folder, i.e.:
        urban-mosaic/washington-square/0/20160628/.../image.jpg

    Override the image root with --image-root if needed.

Controls:
    • Click any dot on the map → shows the image + metadata on the right
    • Layer control (top-left of map) → switch basemap

Dependencies:
    pip install PyQt5 PyQtWebEngine folium pandas
"""

import sys, os, json, tempfile, argparse
from pathlib import Path

import pandas as pd
import folium

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QLabel, QFrame, QSizePolicy
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import Qt, QUrl, QObject, pyqtSlot
from PyQt5.QtGui import QPixmap, QFont, QColor, QPalette

# ── Config ────────────────────────────────────────────────────────────────────

MAX_DOTS_ON_MAP = 5000   # subsample for browser performance

# ── Map builder ───────────────────────────────────────────────────────────────

def build_map_html(df):
    center_lat = df.lat.mean()
    center_lon = df.lon.mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=17,
        tiles="CartoDB dark_matter",
    )
    folium.TileLayer("CartoDB positron", name="Light").add_to(m)
    folium.TileLayer("OpenStreetMap", name="OSM").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # Subsample for rendering performance
    dot_df = df.sample(min(MAX_DOTS_ON_MAP, len(df)), random_state=42)

    points_data = []
    for _, row in dot_df.iterrows():
        points_data.append({
            "id":            int(row["id"]),
            "lat":           float(row["lat"]),
            "lon":           float(row["lon"]),
            "image":         str(row.get("image", "")).strip(),
            "heading":       float(row.get("heading", row.get("azimuth", 0))),
            "period":        str(row.get("period", "")),
            "neighbourhood": str(row.get("neighbourhood", "")),
            "taken_on":      str(row.get("taken_on", "")),
        })

    # Write folium base map to string
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w")
    m.save(tmp.name)
    tmp.close()
    with open(tmp.name, "r") as f:
        html = f.read()
    os.unlink(tmp.name)

    bridge_script = f"""
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<script>
var POINTS = {json.dumps(points_data)};
var _bridge = null;

new QWebChannel(qt.webChannelTransport, function(ch) {{
    _bridge = ch.objects.bridge;
}});

document.addEventListener('DOMContentLoaded', function() {{
    setTimeout(function() {{
        var lmap = null;
        for (var k in window) {{
            try {{
                if (window[k] && window[k].getCenter && window[k].addLayer) {{
                    lmap = window[k]; break;
                }}
            }} catch(e) {{}}
        }}
        if (!lmap) return;

        POINTS.forEach(function(pt) {{
            L.circleMarker([pt.lat, pt.lon], {{
                radius: 5,
                color: "#378ADD",
                fillColor: "#378ADD",
                fillOpacity: 0.85,
                weight: 1,
            }}).addTo(lmap).on('click', function() {{
                if (_bridge) _bridge.pointClicked(JSON.stringify(pt));
            }});
        }});
    }}, 900);
}});
</script>
"""
    html = html.replace("</body>", bridge_script + "\n</body>")
    return html

# ── Qt Bridge ─────────────────────────────────────────────────────────────────

class Bridge(QObject):
    def __init__(self, window):
        super().__init__()
        self._win = window

    @pyqtSlot(str)
    def pointClicked(self, data):
        self._win.on_point_clicked(json.loads(data))

# ── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, df, image_root):
        super().__init__()
        self.df         = df
        self.image_root = Path(image_root)

        self.setWindowTitle("NightWalk — Image Visualizer")
        self.resize(1440, 880)

        root = QSplitter(Qt.Horizontal)
        self.setCentralWidget(root)

        # ── Map ───────────────────────────────────────────────────────────
        self.web = QWebEngineView()
        self.channel = QWebChannel()
        self.bridge = Bridge(self)
        self.channel.registerObject("bridge", self.bridge)
        self.web.page().setWebChannel(self.channel)
        root.addWidget(self.web)

        # ── Right panel: image + metadata ─────────────────────────────────
        right = QWidget()
        right.setFixedWidth(340)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(12, 12, 12, 12)
        rl.setSpacing(10)

        title = QLabel("Image Preview")
        title.setFont(QFont("", 13, QFont.Bold))
        rl.addWidget(title)

        self.img_label = QLabel("Click a dot\non the map")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setFixedHeight(260)
        self.img_label.setStyleSheet(
            "background: #0d0d0d; border-radius: 6px; color: #555; font-size: 13px;"
        )
        rl.addWidget(self.img_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #333;")
        rl.addWidget(sep)

        self.meta = QLabel()
        self.meta.setWordWrap(True)
        self.meta.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.meta.setStyleSheet("font-size: 13px; line-height: 1.6;")
        self.meta.setTextFormat(Qt.RichText)
        rl.addWidget(self.meta)

        stats = QLabel(f"{len(df):,} images total · showing {min(MAX_DOTS_ON_MAP, len(df)):,} on map")
        stats.setStyleSheet("color: #555; font-size: 11px;")
        stats.setAlignment(Qt.AlignBottom)
        rl.addWidget(stats)
        rl.addStretch()

        root.addWidget(right)
        root.setSizes([1080, 340])

        # ── Load map ──────────────────────────────────────────────────────
        print("Building map…")
        html = build_map_html(df)
        self._tmp = tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w", encoding="utf-8"
        )
        self._tmp.write(html)
        self._tmp.close()
        self.web.load(QUrl.fromLocalFile(self._tmp.name))
        print("Map loaded.")

    def on_point_clicked(self, pt):
        # Metadata
        self.meta.setText(
            f"<b>ID:</b> {pt['id']}<br>"
            f"<b>Period:</b> {pt['period']}<br>"
            f"<b>Taken:</b> {pt['taken_on'][:19]}<br>"
            f"<b>Area:</b> {pt['neighbourhood']}<br>"
            f"<b>Heading:</b> {pt['heading']:.1f}°<br>"
            f"<b>Lat/Lon:</b> {pt['lat']:.6f}, {pt['lon']:.6f}<br>"
            f"<br><span style='color:#666; font-size:11px'>{pt['image']}</span>"
        )

        # Resolve image path: image_root / image_column_value
        img_path = self.image_root / pt["image"]
        abs_path = img_path.resolve()
        print(f"Looking for image: {abs_path}")

        if abs_path.exists():
            pix = QPixmap(str(abs_path))
            if not pix.isNull():
                pix = pix.scaled(
                    self.img_label.width() - 4,
                    self.img_label.height() - 4,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self.img_label.setPixmap(pix)
                return

        self.img_label.setPixmap(QPixmap())
        self.img_label.setText(f"Not found:\n{abs_path}")

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to image CSV")
    parser.add_argument(
        "--image-root", default=None,
        help="Root folder prepended to image paths in CSV. "
             "Default: same folder as the CSV file."
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    # Default image root = folder named after the CSV file (sibling folder)
    # e.g. urban-mosaic/washington-square.csv → urban-mosaic/washington-square/
    image_root = Path(args.image_root).resolve() if args.image_root else csv_path.parent / csv_path.stem

    print(f"CSV:        {csv_path}")
    print(f"Image root: {image_root}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows")

    for col in ["lat", "lon", "heading", "azimuth"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "heading" not in df.columns and "azimuth" in df.columns:
        df["heading"] = df["azimuth"]

    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(28, 28, 32))
    palette.setColor(QPalette.WindowText,      QColor(220, 220, 220))
    palette.setColor(QPalette.Base,            QColor(20, 20, 24))
    palette.setColor(QPalette.Text,            QColor(220, 220, 220))
    palette.setColor(QPalette.Button,          QColor(42, 42, 48))
    palette.setColor(QPalette.ButtonText,      QColor(220, 220, 220))
    palette.setColor(QPalette.Highlight,       QColor(0, 119, 182))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    win = MainWindow(df, image_root)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()