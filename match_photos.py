"""
NightWalk — Night/Day Photo Matcher
=====================================
Usage:
    python match_photos.py night-photos/ urban-mosaic/washington-square.csv

Arguments:
    night_dir   Folder containing exported iPhone night photos (.JPG / .HEIC)
    csv         The daytime image CSV (e.g. washington-square.csv)

    --image-root   Root folder for daytime image paths (default: csv_dir/csv_stem)
                   e.g. urban-mosaic/washington-square/
    --output       Output CSV path (default: matches.csv)
    --candidates   Number of nearest daytime candidates to show (default: 4)

Output CSV columns:
    night_photo      filename of night photo (relative to night_dir)
    night_lat        GPS lat from night photo EXIF
    night_lon        GPS lon from night photo EXIF
    night_taken      timestamp from night photo EXIF
    day_image        matched daytime image path (same format as original CSV)
    day_id           matched daytime image ID
    day_lat          daytime image lat
    day_lon          daytime image lon
    day_heading      daytime image heading
    distance_m       distance between night and day GPS points
    skipped          True if user skipped this photo

Dependencies:
    pip install PyQt5 PyQtWebEngine Pillow pandas
"""

import sys, os, math, csv, argparse, subprocess
from pathlib import Path

import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QProgressBar, QSizePolicy, QScrollArea, QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont, QColor, QPalette

# ── Geo ───────────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ── EXIF extraction ───────────────────────────────────────────────────────────

def get_exif_gps(path):
    """Extract (lat, lon, timestamp, heading) from iPhone photo EXIF/XMP.
    heading is None if not available. Returns None if no GPS found."""
    try:
        img = Image.open(path)
        exif_raw = img._getexif()
        if not exif_raw:
            return None

        gps_info = {}
        timestamp = None

        for tag_id, val in exif_raw.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                for k, v in val.items():
                    gps_info[GPSTAGS.get(k, k)] = v
            elif tag == "DateTimeOriginal":
                timestamp = val

        if "GPSLatitude" not in gps_info or "GPSLongitude" not in gps_info:
            return None

        def to_deg(vals):
            d, m, s = vals
            return float(d) + float(m) / 60 + float(s) / 3600

        lat = to_deg(gps_info["GPSLatitude"])
        lon = to_deg(gps_info["GPSLongitude"])
        if gps_info.get("GPSLatitudeRef") == "S":
            lat = -lat
        if gps_info.get("GPSLongitudeRef") == "W":
            lon = -lon

        # GPSImgDirection = camera heading in degrees (True north)
        heading = None
        raw_dir = gps_info.get("GPSImgDirection")
        if raw_dir is not None:
            try:
                heading = float(raw_dir)
            except Exception:
                pass

        return lat, lon, timestamp or "", heading

    except Exception as e:
        print(f"  EXIF error for {path}: {e}")
        return None


def get_gps_mdls(path):
    """Fallback: use macOS mdls to extract GPS (works for HEIC too)."""
    try:
        result = subprocess.run(
            ["mdls", "-name", "kMDItemLatitude",
                     "-name", "kMDItemLongitude",
                     "-name", "kMDItemContentCreationDate", str(path)],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split("\n")
        data = {}
        for line in lines:
            if "=" in line:
                k, v = line.split("=", 1)
                data[k.strip()] = v.strip()
        lat = data.get("kMDItemLatitude", "(null)")
        lon = data.get("kMDItemLongitude", "(null)")
        ts  = data.get("kMDItemContentCreationDate", "")
        if lat == "(null)" or lon == "(null)":
            return None
        return float(lat), float(lon), ts, None  # mdls doesn't expose heading
    except Exception as e:
        print(f"  mdls error for {path}: {e}")
        return None


def extract_gps(path):
    """Try Pillow first, fall back to mdls. Returns (lat, lon, ts, heading) or None."""
    result = get_exif_gps(path)
    if result:
        return result
    return get_gps_mdls(path)

# ── Candidate finder ──────────────────────────────────────────────────────────

def heading_diff(h1, h2):
    """Absolute angular difference between two headings, 0-180."""
    diff = abs(h1 - h2) % 360
    return diff if diff <= 180 else 360 - diff

def find_candidates(lat, lon, df, n=4, night_heading=None, pre_filter=20,
                    dist_weight=0.7, heading_weight=0.3):
    """Return the n best daytime candidates using weighted distance + heading score."""
    
    # --- ADD THIS LINE: Ensure the pool is at least twice the size of n ---
    actual_pre_filter = max(pre_filter, n * 2) 

    df = df.copy()
    df["_dist"] = df.apply(
        lambda r: haversine(lat, lon, r["lat"], r["lon"]), axis=1
    )
    
    # --- UPDATE THIS LINE to use actual_pre_filter ---
    # Stage 1: take closest pre_filter by distance
    pool = df.nsmallest(actual_pre_filter, "_dist").copy()

    if night_heading is not None:
        day_heading_col = "heading" if "heading" in pool.columns else "azimuth"
        pool["_hdiff"] = pool[day_heading_col].apply(
            lambda h: heading_diff(night_heading, h)
        )
        # Normalize both to 0-1 then apply weights
        max_dist = pool["_dist"].max() or 1
        pool["_score"] = (
            dist_weight   * (pool["_dist"] / max_dist) +
            heading_weight * (pool["_hdiff"] / 180.0)
        )
        pool = pool.nsmallest(n, "_score")
    else:
        pool = pool.head(n)

    return pool.reset_index(drop=True)

# ── UI ────────────────────────────────────────────────────────────────────────

CARD_W = 260
CARD_H = 220
IMG_H  = 160

class CandidateCard(QWidget):
    """A clickable card showing one daytime candidate image + metadata."""

    def __init__(self, row, image_root, on_select, already_matched=False):
        super().__init__()
        self._on_select = on_select
        self._row = row
        self.selected = False
        self.already_matched = already_matched

        self.setFixedSize(CARD_W, CARD_H + 60)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Image thumbnail
        self.img_label = QLabel()
        self.img_label.setFixedSize(CARD_W - 12, IMG_H)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("background: #111; border-radius: 4px;")

        img_path = image_root / str(row["image"]).strip()
        if img_path.exists():
            pix = QPixmap(str(img_path)).scaled(
                CARD_W - 12, IMG_H,
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.img_label.setPixmap(pix)
        else:
            self.img_label.setText("image\nnot found")
            self.img_label.setStyleSheet("background:#111; color:#555; border-radius:4px;")

        layout.addWidget(self.img_label)

        # Metadata
        dist_m = float(row.get("_dist", 0))
        heading = float(row.get("heading", row.get("azimuth", 0)))
        meta = QLabel(
            f"<b>{dist_m:.0f}m away</b> &nbsp;·&nbsp; {heading:.0f}° heading<br>"
            f"<span style='color:#888;font-size:11px'>{str(row.get('period',''))} &nbsp;·&nbsp; "
            f"ID {row['id']}</span>"
        )
        meta.setTextFormat(Qt.RichText)
        meta.setWordWrap(True)
        meta.setStyleSheet("font-size: 12px;")
        layout.addWidget(meta)

        if already_matched:
            warn = QLabel("⚠️ already matched")
            warn.setStyleSheet("color: #f4a261; font-size: 11px;")
            layout.addWidget(warn)

        self._update_style()

    def _update_style(self):
        if self.selected:
            self.setStyleSheet(
                "CandidateCard { background: #0a3d62; border: 2px solid #378ADD; border-radius: 8px; }"
            )
        elif self.already_matched:
            self.setStyleSheet(
                "CandidateCard { background: #2a1f0e; border: 2px solid #f4a261; border-radius: 8px; }"
                "CandidateCard:hover { border: 2px solid #fb8500; background: #2e2210; }"
            )
        else:
            self.setStyleSheet(
                "CandidateCard { background: #1e1e24; border: 2px solid #333; border-radius: 8px; }"
                "CandidateCard:hover { border: 2px solid #555; background: #26262e; }"
            )

    def mousePressEvent(self, event):
        self._on_select(self._row)

    def set_selected(self, val):
        self.selected = val
        self._update_style()


class MatcherWindow(QMainWindow):
    def __init__(self, night_photos, df, image_root, output_path, n_candidates):
        super().__init__()
        self.night_photos  = night_photos   # list of (path, lat, lon, timestamp)
        self.df            = df
        self.image_root    = image_root
        self.output_path   = Path(output_path)
        self.n_candidates  = n_candidates
        self.current_idx   = 0
        self.matches       = []             # accumulated results
        self._selected_row = None
        self._cards        = []
        self._matched_day_ids = set()

        # Resume from existing output if present
        if self.output_path.exists():
            existing = pd.read_csv(self.output_path)
            done = set(existing["night_photo"].tolist())
            self.matches = existing.to_dict("records")
            self.night_photos = [p for p in self.night_photos if p[0].name not in done]
            self._matched_day_ids = set(
                str(m["day_id"]) for m in self.matches if not m["skipped"] and m["day_id"]
            )
            print(f"Resuming: {len(done)} already matched, {len(self.night_photos)} remaining")

        # --- ADD THIS LINE ---
        # Keep track of how many matches existed before this session started
        self._session_start = len(self.matches)

        self.setWindowTitle("NightWalk — Photo Matcher")
        self.resize(1400, 760)
        self._build_ui()
        self._load_current()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(16, 16, 16, 16)
        main.setSpacing(16)

        # ── Left: night photo ─────────────────────────────────────────────
        left = QVBoxLayout()

        self.progress_label = QLabel()
        self.progress_label.setFont(QFont("", 12))
        left.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background:#222; border-radius:3px; }"
            "QProgressBar::chunk { background:#378ADD; border-radius:3px; }"
        )
        left.addWidget(self.progress_bar)

        night_title = QLabel("Night Photo")
        night_title.setFont(QFont("", 13, QFont.Bold))
        left.addWidget(night_title)

        self.night_img = QLabel()
        self.night_img.setFixedSize(480, 360)
        self.night_img.setAlignment(Qt.AlignCenter)
        self.night_img.setStyleSheet("background:#0d0d0d; border-radius:8px;")
        left.addWidget(self.night_img)

        self.night_meta = QLabel()
        self.night_meta.setWordWrap(True)
        self.night_meta.setStyleSheet("font-size:12px; color:#aaa;")
        self.night_meta.setTextFormat(Qt.RichText)
        left.addWidget(self.night_meta)

        left.addStretch()

        # Buttons
        btn_row = QHBoxLayout()
        self.back_btn = QPushButton("← Back")
        self.back_btn.setFixedHeight(38)
        self.back_btn.clicked.connect(self._go_back)
        self.back_btn.setStyleSheet("font-size:13px;")

        self.skip_btn = QPushButton("Skip")
        self.skip_btn.setFixedHeight(38)
        self.skip_btn.clicked.connect(self._skip)
        self.skip_btn.setStyleSheet("font-size:13px;")

        self.confirm_btn = QPushButton("✓  Confirm match")
        self.confirm_btn.setFixedHeight(38)
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.clicked.connect(self._confirm)
        self.confirm_btn.setStyleSheet(
            "QPushButton { background:#0a3d62; color:#E6F1FB; border-radius:6px; font-size:13px; }"
            "QPushButton:disabled { background:#222; color:#555; }"
            "QPushButton:enabled:hover { background:#185FA5; }"
        )
        btn_row.addWidget(self.back_btn)
        btn_row.addWidget(self.skip_btn)
        btn_row.addWidget(self.confirm_btn)
        left.addLayout(btn_row)

        main.addLayout(left)

        # ── Divider ───────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color:#333;")
        main.addWidget(sep)

        # ── Right: candidates ─────────────────────────────────────────────
        right = QVBoxLayout()

        day_title = QLabel("Select best matching daytime image")
        day_title.setFont(QFont("", 13, QFont.Bold))
        right.addWidget(day_title)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(12)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll.setWidget(self.grid_widget)
        right.addWidget(self.scroll, stretch=1)

        main.addLayout(right, stretch=1)

    def _load_current(self):
        if self.current_idx >= len(self.night_photos):
            self._finish()
            return

        path, lat, lon, ts, night_heading = self.night_photos[self.current_idx]
        total = len(self.night_photos)

        # Progress
        self.progress_label.setText(
            f"Photo {self.current_idx + 1} of {total}  —  {len(self.matches)} matched so far"
        )
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(self.current_idx)

        # Night image
        pix = QPixmap(str(path))
        if not pix.isNull():
            pix = pix.scaled(480, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.night_img.setPixmap(pix)
        else:
            self.night_img.setText("Cannot load image")

        heading_str = f"{night_heading:.1f}°" if night_heading is not None else "n/a"
        self.night_meta.setText(
            f"<b>{path.name}</b><br>"
            f"GPS: {lat:.6f}, {lon:.6f}<br>"
            f"Heading: {heading_str}<br>"
            f"Taken: {ts}"
        )

        # Find candidates
        candidates = find_candidates(lat, lon, self.df, self.n_candidates, night_heading)

        # Clear old cards
        for card in self._cards:
            card.deleteLater()
        self._cards = []
        self._selected_row = None
        self.confirm_btn.setEnabled(False)

        for _, row in candidates.iterrows():
            already = str(row["id"]) in self._matched_day_ids
            card = CandidateCard(row, self.image_root, self._on_card_selected, already)
            self._cards.append(card)
        self._reflow_cards()

    def _go_back(self):
        if self.current_idx == 0:
            return
        # Only pop matches made in this session
        if len(self.matches) > self._session_start:
            removed = self.matches.pop()
            if not removed["skipped"] and removed["day_id"]:
                self._matched_day_ids.discard(str(removed["day_id"]))
            self._save()
        self.current_idx -= 1
        self._load_current()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            if self.confirm_btn.isEnabled():
                self._confirm()
        elif key == Qt.Key_Right:
            if self.confirm_btn.isEnabled():
                self._confirm()
            else:
                self._skip()
        elif key == Qt.Key_Left:
            self._go_back()
        elif key == Qt.Key_X:
            self._skip()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reflow_cards()

    def _reflow_cards(self):
        if not self._cards:
            return
        # Calculate how many columns fit in the scroll area width
        available = self.scroll.viewport().width() - 12
        cols = max(1, available // (CARD_W + 12))
        # Re-add all cards to grid
        for i, card in enumerate(self._cards):
            self.grid_layout.addWidget(card, i // cols, i % cols)

    def _on_card_selected(self, row):
        self._selected_row = row
        for card in self._cards:
            card.set_selected(card._row["id"] == row["id"])
        self.confirm_btn.setEnabled(True)

    def _confirm(self):
        if self._selected_row is None:
            return
        path, lat, lon, ts, night_heading = self.night_photos[self.current_idx]
        row = self._selected_row
        self.matches.append({
            "night_photo":  path.name,
            "night_lat":    lat,
            "night_lon":    lon,
            "night_taken":  ts,
            "day_image":    str(row["image"]).strip(),   # original CSV format
            "day_id":       int(row["id"]),
            "day_lat":      float(row["lat"]),
            "day_lon":      float(row["lon"]),
            "day_heading":  float(row.get("heading", row.get("azimuth", 0))),
            "distance_m":   round(float(row["_dist"]), 2),
            "skipped":      False,
        })
        self._matched_day_ids.add(str(row["id"]))
        self._save()
        self.current_idx += 1
        self._load_current()

    def _skip(self):
        path, lat, lon, ts, night_heading = self.night_photos[self.current_idx]
        self.matches.append({
            "night_photo":  path.name,
            "night_lat":    lat,
            "night_lon":    lon,
            "night_taken":  ts,
            "day_image":    "",
            "day_id":       "",
            "day_lat":      "",
            "day_lon":      "",
            "day_heading":  "",
            "distance_m":   "",
            "skipped":      True,
        })
        self._save()
        self.current_idx += 1
        self._load_current()

    def _save(self):
        pd.DataFrame(self.matches).to_csv(self.output_path, index=False)

    def _finish(self):
        self.night_img.setText("All done!")
        self.night_meta.setText(
            f"Matched {sum(1 for m in self.matches if not m['skipped'])} photos.\n"
            f"Skipped {sum(1 for m in self.matches if m['skipped'])}.\n"
            f"Saved to {self.output_path}"
        )
        self.confirm_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)
        for card in self._cards:
            card.deleteLater()
        self._cards = []
        print(f"\nDone. Results saved to {self.output_path}")

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("night_dir", help="Folder of night photos")
    parser.add_argument("csv",       help="Daytime image CSV")
    parser.add_argument("--image-root",   default=None)
    parser.add_argument("--output",       default="matches.csv")
    parser.add_argument("--candidates",   type=int, default=30)
    args = parser.parse_args()

    night_dir = Path(args.night_dir)
    csv_path  = Path(args.csv).expanduser().absolute()
    image_root = Path(args.image_root).expanduser().absolute() if args.image_root else csv_path.parent / csv_path.stem

    print(f"Night photos: {night_dir}")
    print(f"Daytime CSV:  {csv_path}")
    print(f"Image root:   {image_root}")

    # Load night photos and extract GPS
    extensions = {".jpg", ".jpeg", ".JPG", ".JPEG", ".heic", ".HEIC"}
    photo_files = sorted([p for p in night_dir.iterdir() if p.suffix in extensions])
    print(f"\nFound {len(photo_files)} night photos, extracting GPS…")

    night_photos = []
    skipped_no_gps = []
    for p in photo_files:
        gps = extract_gps(p)
        if gps:
            night_photos.append((p, gps[0], gps[1], gps[2], gps[3]))
        else:
            skipped_no_gps.append(p.name)

    if skipped_no_gps:
        print(f"  Warning: no GPS found in {len(skipped_no_gps)} photos: {skipped_no_gps[:5]}")
    print(f"  {len(night_photos)} photos with GPS ready for matching")

    if not night_photos:
        print("No photos with GPS data found. Exiting.")
        sys.exit(1)

    # Load daytime CSV
    df = pd.read_csv(csv_path)
    for col in ["lat", "lon", "snapped_lat", "snapped_lon", "heading", "azimuth"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Use snapped coordinates where available
    if "snapped_lat" in df.columns and "snapped_lon" in df.columns:
        df["lat"] = df["snapped_lat"].fillna(df["lat"])
        df["lon"] = df["snapped_lon"].fillna(df["lon"])
    if "heading" not in df.columns and "azimuth" in df.columns:
        df["heading"] = df["azimuth"]
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    print(f"  {len(df):,} daytime images loaded")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(22, 22, 28))
    palette.setColor(QPalette.WindowText,      QColor(220, 220, 220))
    palette.setColor(QPalette.Base,            QColor(15, 15, 20))
    palette.setColor(QPalette.Text,            QColor(220, 220, 220))
    palette.setColor(QPalette.Button,          QColor(42, 42, 52))
    palette.setColor(QPalette.ButtonText,      QColor(220, 220, 220))
    palette.setColor(QPalette.Highlight,       QColor(0, 119, 182))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    win = MatcherWindow(night_photos, df, image_root, args.output, args.candidates)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()