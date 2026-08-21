"""
generate_maps.py  [v4]
======================
Generates 2 HTML map figures for a LaTeX paper:

  Figure A — map_effnet.html
    EfficientNet training images colored by grid cell.
    Shows dense, full-area spatial coverage.

  Figure B — map_train_test.html
    Train = small, muted grey circles (background)
    Test  = grid-cell colored dots on top (larger, vivid)
    Shows test set evenly samples across all collected routes.

CSV inputs (place next to this script):
  - efficientnet_train_images.csv  (cols: snapped_lat, snapped_lon, grid_cell)
  - train_split.csv                (cols: day_lat, day_lon)
  - test_split.csv                 (cols: day_lat, day_lon)

Dependencies:
    pip install pandas
"""

import os, json, random, colorsys
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EFFNET_CSV = os.path.join(SCRIPT_DIR, "efficientnet_train_images.csv")
TRAIN_CSV  = os.path.join(SCRIPT_DIR, "train_split.csv")
TEST_CSV   = os.path.join(SCRIPT_DIR, "test_split.csv")
OUT_DIR    = SCRIPT_DIR

# ── Map view (identical for both figures) ─────────────────────────────────────

CENTER_LAT = 40.7295
CENTER_LON = -73.9990
ZOOM       = 15

# ── Dot styles ────────────────────────────────────────────────────────────────

# EfficientNet / Test — vivid grid-cell colors
CELL_SATURATION   = 0.72
CELL_LIGHTNESS    = 0.42
CELL_SHUFFLE_SEED = 99

# Train (background, muted)
TRAIN_COLOR   = "#A1DEFF"   # grey
TRAIN_RADIUS  = 3
TRAIN_OPACITY = 0.45

# Test (foreground, vivid, larger)
TEST_RADIUS   = 4
TEST_OPACITY  = 0.92

# EfficientNet dots
EFFNET_RADIUS  = 3
EFFNET_OPACITY = 0.85

TILE_URL  = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
TILE_ATTR = "© OpenStreetMap © CARTO"

# ── Data loading ──────────────────────────────────────────────────────────────

def load_effnet(path):
    df = pd.read_csv(path).dropna(subset=["snapped_lat", "snapped_lon", "grid_cell"])
    print(f"  EfficientNet : {len(df):,} points, {df['grid_cell'].nunique()} cells")
    return df

def load_split(path, label):
    df = pd.read_csv(path).dropna(subset=["day_lat", "day_lon"])
    print(f"  {label:12s} : {len(df):,} points")
    return df

# ── Grid-cell coloring ────────────────────────────────────────────────────────

def build_cell_colors(df):
    unique_cells = sorted(df["grid_cell"].unique())
    shuffled = unique_cells.copy()
    random.Random(CELL_SHUFFLE_SEED).shuffle(shuffled)
    n = len(shuffled)
    return {
        cell: "#{:02x}{:02x}{:02x}".format(
            *[int(c * 255) for c in colorsys.hls_to_rgb(i / n, CELL_LIGHTNESS, CELL_SATURATION)]
        )
        for i, cell in enumerate(shuffled)
    }

# ── HTML shell ────────────────────────────────────────────────────────────────

STYLE = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f5f5f3;
  display: flex; flex-direction: column; height: 100vh; padding: 14px; gap: 10px;
}
h1 { font-size: 15px; font-weight: 600; }
.subtitle { font-size: 11px; color: #888; margin-top: 2px; }
.stats { display: flex; gap: 10px; flex-shrink: 0; }
.stat { background:#fff; border:0.5px solid #ddd; border-radius:8px; padding:6px 14px; }
.stat-label { font-size:10px; color:#999; }
.stat-value { font-size:18px; font-weight:500; }
#map { flex:1; border-radius:10px; border:0.5px solid #ddd; }
.legend {
  position:absolute; bottom:24px; right:10px; z-index:1000;
  background:rgba(255,255,255,0.93); border:0.5px solid #ccc;
  border-radius:8px; padding:8px 14px; font-size:12px;
  display:flex; flex-direction:column; gap:6px;
}
.legend-row { display:flex; align-items:center; gap:8px; }
.legend-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
.legend-sq  { width:10px; height:10px; border-radius:2px; flex-shrink:0; }
"""

def html_shell(title, subtitle, stats_html, map_js, legend_html):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>{STYLE}</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">{subtitle}</div>
<div class="stats">{stats_html}</div>
<div style="position:relative;flex:1;display:flex;flex-direction:column;">
  <div id="map" style="flex:1;border-radius:10px;border:0.5px solid #ddd;"></div>
  <div class="legend">{legend_html}</div>
</div>
<script>
const map = L.map("map").setView([{CENTER_LAT}, {CENTER_LON}], {ZOOM});
L.tileLayer("{TILE_URL}", {{attribution:"{TILE_ATTR}",subdomains:"abcd",maxZoom:19}}).addTo(map);
{map_js}
</script>
</body>
</html>"""

def stat(label, value):
    return f'<div class="stat"><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>'

def leg_circle(color, label):
    return f'<div class="legend-row"><div class="legend-dot" style="background:{color};opacity:0.85"></div><span>{label}</span></div>'

def leg_square(color, label):
    return f'<div class="legend-row"><div class="legend-sq" style="background:{color}"></div><span>{label}</span></div>'

def save(html, name):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  → {path}")

# ── Figure A: EfficientNet ────────────────────────────────────────────────────

def build_effnet(df):
    cell_colors = build_cell_colors(df)
    pts = [{"lat": r.snapped_lat, "lon": r.snapped_lon, "c": cell_colors[r.grid_cell]}
           for r in df.itertuples()]

    js = f"""
const pts = {json.dumps(pts)};
pts.forEach(p => {{
  L.circleMarker([p.lat, p.lon], {{
    radius: {EFFNET_RADIUS}, fillColor: p.c,
    color: "rgba(255,255,255,0.4)", weight: 0.5,
    fillOpacity: {EFFNET_OPACITY}
  }}).addTo(map);
}});
"""
    n_cells = df["grid_cell"].nunique()
    save(html_shell(
        title       = "EfficientNet Training Image Locations",
        subtitle    = "Downtown Manhattan · one dot per image · colored by grid cell",
        stats_html  = stat("images", f"{len(df):,}") + stat("grid cells", f"{n_cells}"),
        map_js      = js,
        legend_html = leg_circle("#aaa", f"One image ({n_cells} cells, colored)"),
    ), "map_effnet.html")

# ── Figure B: Train (grey) + Test (grid-cell colors) ─────────────────────────

def build_train_test(df_train, df_test, cell_colors):
    # Split CSVs don't have grid_cell — assign color by nearest effnet cell
    # via a simple lat/lon string key matching the grid_cell naming convention.
    # If your split CSVs DO have grid_cell, use that directly instead.
    # Here we fall back to coloring test dots by their index-based hue so they
    # still look vivid and distinct without requiring a spatial join.

    # Check if split CSVs have grid_cell
    has_cell_train = "grid_cell" in df_train.columns
    has_cell_test  = "grid_cell" in df_test.columns

    def test_color(row, i):
        if has_cell_test and row.grid_cell in cell_colors:
            return cell_colors[row.grid_cell]
        # fallback: cycle through palette by index
        hue = i / max(len(df_test), 1)
        r, g, b = colorsys.hls_to_rgb(hue, CELL_LIGHTNESS, CELL_SATURATION)
        return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

    train_pts = [{"lat": r.day_lat, "lon": r.day_lon} for r in df_train.itertuples()]
    test_pts  = [{"lat": r.day_lat, "lon": r.day_lon, "c": test_color(r, i)}
                 for i, r in enumerate(df_test.itertuples())]

    js = f"""
// Train — grey circles, drawn first (underneath)
const trainPts = {json.dumps(train_pts)};
trainPts.forEach(p => {{
  L.circleMarker([p.lat, p.lon], {{
    radius: {TRAIN_RADIUS}, fillColor: "{TRAIN_COLOR}",
    color: "#27B3FF", weight: 1.2,
    fillOpacity: {TRAIN_OPACITY}
  }}).addTo(map);
}});

// Test — vivid colored squares (rectangles via SVG), drawn on top
const testPts = {json.dumps(test_pts)};
testPts.forEach(p => {{
  const sq = L.marker([p.lat, p.lon], {{
    icon: L.divIcon({{
      className: "",
      html: `<div style="
        width:8px; height:8px;
        background:${{p.c}};
        border:1.5px solid rgba(255,255,255,0.8);
        border-radius:1px;
        margin:-4px 0 0 -4px;
      "></div>`,
      iconSize: [8, 8],
      iconAnchor: [4, 4],
    }})
  }}).addTo(map);
}});
"""

    save(html_shell(
        title       = "Train + Test Split Locations",
        subtitle    = "Downtown Manhattan · test set evenly samples collected routes",
        stats_html  = stat("train", f"{len(df_train):,}") + stat("test", f"{len(df_test):,}"),
        map_js      = js,
        legend_html = (
            leg_circle(TRAIN_COLOR, f"Train images (n={len(df_train):,})") +
            leg_square("#e06c4a",   f"Test images (n={len(df_test):,}), colored by cell")
        ),
    ), "map_train_test.html")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    df_effnet = load_effnet(EFFNET_CSV)
    df_train  = load_split(TRAIN_CSV, "Train")
    df_test   = load_split(TEST_CSV,  "Test")

    cell_colors = build_cell_colors(df_effnet)

    print("\nGenerating figures …")
    build_effnet(df_effnet)
    build_train_test(df_train, df_test, cell_colors)

    print(f"\nDone — 2 HTML files in: {OUT_DIR}")
    print("Open in browser, then screenshot or use a headless tool to export PNG for LaTeX.")

if __name__ == "__main__":
    main()