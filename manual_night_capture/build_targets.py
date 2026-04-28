"""Generate prioritized targets for manual nighttime photo capture
around Washington Square Park.

Outputs:
    manual_night_capture/targets.csv
    manual_night_capture/targets_map.html

Run from the repository root:
    python manual_night_capture/build_targets.py

See --help for tuning flags.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Approximate metres per degree at NYC latitude. Good enough for a 25 m grid
# over a few hundred metres of extent.
METRES_PER_DEG_LAT = 111_000.0
METRES_PER_DEG_LON = 84_000.0

DEFAULT_BBOX = (40.727, 40.735, -74.003, -73.992)  # lat_min, lat_max, lon_min, lon_max

WASHINGTON_SQUARE_CSV = Path("urban-mosaic/washington-square.csv")
LABELS_FINAL_CSV = Path("label_split/labels_final.csv")
OUTPUT_DIR = Path("manual_night_capture")
OUTPUT_CSV = OUTPUT_DIR / "targets.csv"
OUTPUT_HTML = OUTPUT_DIR / "targets_map.html"

REQUIRED_DAY_COLS = {"image", "lat", "lon"}
REQUIRED_LABEL_COLS = {"image", "final_label"}

CATEGORY_ORDER = ["priority", "validation", "secondary"]
CATEGORY_COLORS = {
    "priority": "#2ecc40",   # green
    "validation": "#0074d9", # blue
    "secondary": "#ffdc00",  # yellow
}
CARDINAL_ARROWS = {"N": "↑", "E": "→", "S": "↓", "W": "←"}
CARDINAL_CENTERS = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a prioritized list of cells for manual nighttime "
        "photo capture around Washington Square Park.",
    )
    p.add_argument("--grid-size", type=float, default=25.0,
                   help="Cell size in metres (default: 25).")
    p.add_argument("--min-day-count", type=int, default=3,
                   help="Minimum day photos per cell to be emitted (default: 3).")
    p.add_argument("--max-priority", type=int, default=500,
                   help="Maximum priority cells to emit (default: 500).")
    p.add_argument("--max-secondary", type=int, default=50,
                   help="Maximum secondary cells to emit (default: 50).")
    p.add_argument(
        "--bbox", type=float, nargs=4, default=list(DEFAULT_BBOX),
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="Bounding box (default: %s)." % (DEFAULT_BBOX,),
    )
    return p.parse_args()


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_day_photos(bbox: Tuple[float, float, float, float]) -> Tuple[pd.DataFrame, str]:
    """Load the day-photo dataset, validate its schema, restrict to bbox."""
    if not WASHINGTON_SQUARE_CSV.exists():
        fail(f"Missing input file: {WASHINGTON_SQUARE_CSV}")
    print(f"Loading day photos from {WASHINGTON_SQUARE_CSV}…")
    df = pd.read_csv(WASHINGTON_SQUARE_CSV, low_memory=False)

    missing = REQUIRED_DAY_COLS - set(df.columns)
    if missing:
        fail(
            f"{WASHINGTON_SQUARE_CSV} is missing expected columns: "
            f"{sorted(missing)}. Found: {sorted(df.columns)}"
        )
    # Prefer 'heading' over 'azimuth'.
    heading_col = None
    for candidate in ("heading", "azimuth"):
        if candidate in df.columns:
            heading_col = candidate
            break
    if heading_col is None:
        fail(
            f"{WASHINGTON_SQUARE_CSV} must have a 'heading' or 'azimuth' "
            f"column. Found: {sorted(df.columns)}"
        )

    df = df[["image", "lat", "lon", heading_col]].copy()
    df = df.rename(columns={heading_col: "heading_deg"})
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["heading_deg"] = pd.to_numeric(df["heading_deg"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["lat", "lon"])

    lat_min, lat_max, lon_min, lon_max = bbox
    in_bbox = (
        df["lat"].between(lat_min, lat_max)
        & df["lon"].between(lon_min, lon_max)
    )
    df = df.loc[in_bbox].reset_index(drop=True)
    print(f"  {before:,} rows total → {len(df):,} day photos in bbox "
          f"(heading source: '{heading_col}').")
    return df, heading_col


def load_night_photos(day_df: pd.DataFrame) -> pd.DataFrame:
    """Load night-labelled images and join lat/lon from the day dataset."""
    if not LABELS_FINAL_CSV.exists():
        fail(f"Missing input file: {LABELS_FINAL_CSV}")
    print(f"Loading labels from {LABELS_FINAL_CSV}…")
    labels = pd.read_csv(LABELS_FINAL_CSV, low_memory=False)

    missing = REQUIRED_LABEL_COLS - set(labels.columns)
    if missing:
        fail(
            f"{LABELS_FINAL_CSV} is missing expected columns: "
            f"{sorted(missing)}. Found: {sorted(labels.columns)}"
        )

    night_labels = labels.loc[labels["final_label"] == "night", ["image"]].copy()
    print(f"  {len(night_labels):,} night-labelled images in labels_final.csv.")

    # Join on image to pick up lat/lon. day_df is already restricted to the
    # bbox, so this naturally limits night photos to the bbox too.
    night = night_labels.merge(
        day_df[["image", "lat", "lon"]], on="image", how="inner"
    )
    print(f"  {len(night):,} night photos after inner join with day metadata "
          f"(joined on 'image').")
    return night


def build_grid(
    bbox: Tuple[float, float, float, float], grid_size_m: float
) -> dict:
    lat_min, lat_max, lon_min, lon_max = bbox
    cell_h = grid_size_m / METRES_PER_DEG_LAT
    cell_w = grid_size_m / METRES_PER_DEG_LON
    n_rows = int(np.ceil((lat_max - lat_min) / cell_h))
    n_cols = int(np.ceil((lon_max - lon_min) / cell_w))
    print(
        f"Grid: {n_rows} rows × {n_cols} cols "
        f"({grid_size_m:g} m → {cell_h:.6f}° lat × {cell_w:.6f}° lon)."
    )
    return {
        "lat_min": lat_min, "lat_max": lat_max,
        "lon_min": lon_min, "lon_max": lon_max,
        "cell_h": cell_h, "cell_w": cell_w,
        "n_rows": n_rows, "n_cols": n_cols,
    }


def assign_cells(df: pd.DataFrame, grid: dict) -> pd.DataFrame:
    """Append row, col, cell_id columns. Drops anything outside the grid."""
    if df.empty:
        out = df.copy()
        out["row"] = pd.Series(dtype=int)
        out["col"] = pd.Series(dtype=int)
        out["cell_id"] = pd.Series(dtype=str)
        return out

    row = np.floor((df["lat"].to_numpy() - grid["lat_min"]) / grid["cell_h"]).astype(int)
    col = np.floor((df["lon"].to_numpy() - grid["lon_min"]) / grid["cell_w"]).astype(int)
    keep = (
        (row >= 0) & (row < grid["n_rows"])
        & (col >= 0) & (col < grid["n_cols"])
    )
    out = df.loc[keep].copy()
    out["row"] = row[keep]
    out["col"] = col[keep]
    out["cell_id"] = [f"r{r:02d}_c{c:02d}" for r, c in zip(out["row"], out["col"])]
    return out


def cardinal_bin(heading_deg: float) -> str:
    """N (315-45), E (45-135), S (135-225), W (225-315). NaNs -> 'unknown'."""
    if pd.isna(heading_deg):
        return "unknown"
    h = float(heading_deg) % 360.0
    if h >= 315.0 or h < 45.0:
        return "N"
    if h < 135.0:
        return "E"
    if h < 225.0:
        return "S"
    return "W"


def angular_distance(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def compute_cell_stats(
    day_in_grid: pd.DataFrame,
    night_in_grid: pd.DataFrame,
    grid: dict,
    min_day_count: int,
) -> pd.DataFrame:
    """Per-cell stats for cells with day_count >= min_day_count."""
    print(f"Computing per-cell stats (min_day_count={min_day_count})…")

    day = day_in_grid.copy()
    day["cardinal"] = day["heading_deg"].apply(cardinal_bin)

    day_counts = day.groupby("cell_id").size().rename("day_count")
    night_counts = (
        night_in_grid.groupby("cell_id").size().rename("night_count")
        if not night_in_grid.empty
        else pd.Series(dtype=int, name="night_count")
    )

    cells = pd.DataFrame(day_counts).join(night_counts, how="left")
    cells["night_count"] = cells["night_count"].fillna(0).astype(int)
    cells = cells.loc[cells["day_count"] >= min_day_count].copy()
    print(f"  {len(cells):,} cells meet day_count >= {min_day_count}.")

    rows = []
    for cell_id, sub in day.loc[day["cell_id"].isin(cells.index)].groupby("cell_id"):
        bin_counts = sub["cardinal"].value_counts()
        bin_counts = bin_counts.reindex(["N", "E", "S", "W"], fill_value=0)
        modal = bin_counts.idxmax()
        modal_count = int(bin_counts.loc[modal])
        total = int(bin_counts.sum())
        confidence = "strong" if total > 0 and (modal_count / total) > 0.6 else "mixed"

        bin_members = sub.loc[sub["cardinal"] == modal].copy()
        center = CARDINAL_CENTERS[modal]
        bin_members["dist"] = bin_members["heading_deg"].apply(
            lambda h: angular_distance(h, center) if pd.notna(h) else float("inf")
        )
        # Tie-break on image filename so the result is deterministic.
        bin_members = bin_members.sort_values(["dist", "image"])
        target_image = bin_members.iloc[0]["image"]

        # Cell centre in lat/lon.
        row_idx, col_idx = sub.iloc[0][["row", "col"]]
        lat_center = grid["lat_min"] + (row_idx + 0.5) * grid["cell_h"]
        lon_center = grid["lon_min"] + (col_idx + 0.5) * grid["cell_w"]

        rows.append({
            "cell_id": cell_id,
            "lat_center": round(lat_center, 7),
            "lon_center": round(lon_center, 7),
            "day_count": int(cells.loc[cell_id, "day_count"]),
            "night_count": int(cells.loc[cell_id, "night_count"]),
            "modal_heading": modal,
            "heading_confidence": confidence,
            "target_day_image_hint": target_image,
        })

    return pd.DataFrame(rows)


def categorize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat = np.where(
        df["night_count"] >= 2, "validation",
        np.where(df["night_count"] == 1, "secondary", "priority"),
    )
    df["cell_category"] = cat
    return df


def select_and_sort(df: pd.DataFrame, max_priority: int, max_secondary: int) -> pd.DataFrame:
    parts = []
    for cat in CATEGORY_ORDER:
        sub = df.loc[df["cell_category"] == cat].copy()
        sub = sub.sort_values(["day_count", "cell_id"], ascending=[False, True])
        if cat == "priority":
            sub = sub.head(max_priority)
        elif cat == "secondary":
            sub = sub.head(max_secondary)
        # validation: keep all
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    return out[
        [
            "cell_id", "lat_center", "lon_center",
            "day_count", "night_count",
            "modal_heading", "heading_confidence",
            "target_day_image_hint", "cell_category",
        ]
    ]


def print_summary(df: pd.DataFrame) -> None:
    print("\n=== Summary ===")
    print(f"Total cells emitted: {len(df):,}")
    by_cat = df["cell_category"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
    print("By category:")
    for cat in CATEGORY_ORDER:
        print(f"  {cat:<10} {int(by_cat[cat]):>5}")

    by_heading = df["modal_heading"].value_counts().reindex(
        ["N", "E", "S", "W"], fill_value=0
    )
    print("By modal heading:")
    for h in ["N", "E", "S", "W"]:
        print(f"  {h} {CARDINAL_ARROWS[h]}  {int(by_heading[h]):>5}")

    by_conf = df["heading_confidence"].value_counts().reindex(
        ["strong", "mixed"], fill_value=0
    )
    print("By heading confidence:")
    for c in ["strong", "mixed"]:
        print(f"  {c:<7} {int(by_conf[c]):>5}")


def render_html(df: pd.DataFrame, bbox: Tuple[float, float, float, float]) -> str:
    lat_min, lat_max, lon_min, lon_max = bbox
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    records = df.to_dict(orient="records")
    payload = {
        "center": [center_lat, center_lon],
        "bbox": [[lat_min, lon_min], [lat_max, lon_max]],
        "colors": CATEGORY_COLORS,
        "arrows": CARDINAL_ARROWS,
        "headingDeg": CARDINAL_CENTERS,
        "cells": records,
    }
    data_json = json.dumps(payload, ensure_ascii=False, default=str)

    legend_rows = "".join(
        f'<div><span class="dot" style="background:{CATEGORY_COLORS[c]}"></span>'
        f'{c} ({int((df["cell_category"] == c).sum())})</div>'
        for c in CATEGORY_ORDER
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
<title>Nighttime Capture Targets — Washington Square</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body {{ height: 100%; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  #map {{ position: absolute; inset: 0; }}
  .legend {{
    position: absolute;
    bottom: 18px;
    left: 18px;
    z-index: 1000;
    background: rgba(26, 26, 26, 0.92);
    color: #eee;
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid #444;
    font-size: 13px;
    line-height: 1.6;
  }}
  .legend b {{ display: block; margin-bottom: 4px; }}
  .legend .dot {{
    display: inline-block; width: 10px; height: 10px;
    border-radius: 50%; margin-right: 6px; vertical-align: middle;
    border: 1px solid rgba(0,0,0,0.4);
  }}
  .popup {{ font-size: 13px; line-height: 1.4; }}
  .popup .cid {{ font-weight: 600; }}
  .popup .hint {{ word-break: break-all; color: #444; font-size: 11px; }}
  .arrow-icon {{ background: transparent; border: none; }}
  .arrow-icon svg {{ display: block; overflow: visible; }}
  .legend .arrow-row {{ display: flex; gap: 10px; align-items: center; margin-top: 6px; }}
  .legend .arrow-row svg {{ display: block; }}
</style>
</head>
<body>
<div id="map"></div>
<div class="legend">
  <b>Capture targets</b>
  {legend_rows}
  <div class="arrow-row">
    <svg width="22" height="22" viewBox="-11 -11 22 22" aria-hidden="true">
      <g stroke="#111" stroke-width="1" stroke-linejoin="round" fill="#888">
        <circle cx="0" cy="0" r="4"></circle>
        <polygon points="0,-9 3.4,-3.5 -3.4,-3.5"></polygon>
      </g>
    </svg>
    <span style="font-size:11px;color:#aaa;">arrow points the direction to face<br>faded = mixed heading confidence</span>
  </div>
  <div style="font-size:11px;color:#aaa;margin-top:4px;">↑ N · → E · ↓ S · ← W</div>
</div>
<script>
const DATA = {data_json};
const map = L.map('map', {{ zoomControl: true }}).setView(DATA.center, 16);
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors'
}}).addTo(map);
map.fitBounds(DATA.bbox);

function escapeHtml(s) {{
  return String(s).replace(/[&<>"']/g, c => ({{
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }})[c]);
}}

function arrowSvg(color, rotationDeg, opacity) {{
  return `
    <svg width="26" height="26" viewBox="-13 -13 26 26">
      <g transform="rotate(${{rotationDeg}})"
         stroke="#111" stroke-width="1" stroke-linejoin="round"
         fill="${{color}}" fill-opacity="${{opacity}}">
        <circle cx="0" cy="0" r="5"></circle>
        <polygon points="0,-11 4,-4 -4,-4"></polygon>
      </g>
    </svg>`;
}}

DATA.cells.forEach(cell => {{
  const color = DATA.colors[cell.cell_category];
  const arrow = DATA.arrows[cell.modal_heading] || '?';
  const rotation = DATA.headingDeg[cell.modal_heading];
  const opacity = cell.heading_confidence === 'strong' ? 1.0 : 0.55;
  const icon = L.divIcon({{
    className: 'arrow-icon',
    html: arrowSvg(color, rotation == null ? 0 : rotation, opacity),
    iconSize: [26, 26],
    iconAnchor: [13, 13],
    popupAnchor: [0, -10],
  }});
  const marker = L.marker([cell.lat_center, cell.lon_center], {{ icon }}).addTo(map);
  const html = `
    <div class="popup">
      <div><span class="cid">${{escapeHtml(cell.cell_id)}}</span>
        &nbsp;<span style="color:${{color}};">●</span> ${{escapeHtml(cell.cell_category)}}</div>
      <div>Heading: <b>${{escapeHtml(cell.modal_heading)}} ${{arrow}}</b>
        <span style="color:#888">(${{escapeHtml(cell.heading_confidence)}})</span></div>
      <div>Day photos: ${{cell.day_count}} · Night photos: ${{cell.night_count}}</div>
      <div class="hint">${{escapeHtml(cell.target_day_image_hint)}}</div>
    </div>`;
  marker.bindPopup(html);
}});
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    bbox = tuple(args.bbox)
    if not (bbox[0] < bbox[1] and bbox[2] < bbox[3]):
        fail(f"Invalid --bbox {bbox}: need LAT_MIN < LAT_MAX and LON_MIN < LON_MAX.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    day_df, _ = load_day_photos(bbox)
    night_df = load_night_photos(day_df)

    grid = build_grid(bbox, args.grid_size)
    day_grid = assign_cells(day_df, grid)
    night_grid = assign_cells(night_df, grid)

    cells = compute_cell_stats(day_grid, night_grid, grid, args.min_day_count)
    if cells.empty:
        fail("No cells met the minimum day-count threshold; nothing to emit.")

    cells = categorize(cells)
    out_df = select_and_sort(cells, args.max_priority, args.max_secondary)

    print(f"\nWriting {OUTPUT_CSV}…")
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Writing {OUTPUT_HTML}…")
    OUTPUT_HTML.write_text(render_html(out_df, bbox), encoding="utf-8")

    print_summary(out_df)
    print("\nDone.")


if __name__ == "__main__":
    main()
