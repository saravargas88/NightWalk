"""
Street Image Route Visualizer
==============================
Usage:
    python visualize_routes.py your_images.csv

Opens an interactive map in your browser showing:
  - Individual image points (click for metadata)
  - Inferred car driving routes (images grouped by vehicle + session, colored by route)
  - Route stats in the sidebar

Dependencies: pip install folium pandas scikit-learn
"""

import sys
import os
import pandas as pd
import folium
import webbrowser
import json
import math
from pathlib import Path


# ── Config ──────────────────────────────────────────────────────────────────

# Max time gap (seconds) between consecutive images on the same vehicle
# to still consider them part of the same "pass" down a block
ROUTE_TIME_GAP_SEC = 90

# Max distance (meters) between consecutive images to be on the same route segment
ROUTE_DIST_GAP_M = 150

# Minimum images to show a route polyline (filters out single stray shots)
MIN_ROUTE_LENGTH = 3

OUTPUT_FILE = "image_map.html"

# ── Helpers ──────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    """Distance in meters between two lat/lon points."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def infer_routes(df):
    """
    Group images into route segments (a single car pass down a block).
    Strategy:
      1. Sort by vehicle + timestamp
      2. Split into a new segment whenever the time gap > ROUTE_TIME_GAP_SEC
         OR the spatial gap > ROUTE_DIST_GAP_M
    Returns a list of DataFrames, one per route segment.
    """
    routes = []

    # Use android_id as vehicle identifier; fall back to vehicle_id
    id_col = "android_id" if "android_id" in df.columns else "vehicle_id"

    # Parse timestamps robustly
    df = df.copy()
    if "timestamp" in df.columns:
        df["_ts"] = pd.to_numeric(df["timestamp"], errors="coerce")
    elif "taken_on" in df.columns:
        df["_ts"] = pd.to_datetime(df["taken_on"], utc=True, errors="coerce").astype("int64") // 10**9
    else:
        df["_ts"] = 0

    df = df.sort_values([id_col, "_ts"]).reset_index(drop=True)

    for vehicle, group in df.groupby(id_col):
        group = group.reset_index(drop=True)
        if len(group) == 0:
            continue

        segment = [0]
        for i in range(1, len(group)):
            prev = group.iloc[i - 1]
            curr = group.iloc[i]
            dt = abs(float(curr["_ts"]) - float(prev["_ts"]))
            dd = haversine(prev["lat"], prev["lon"], curr["lat"], curr["lon"])
            if dt > ROUTE_TIME_GAP_SEC or dd > ROUTE_DIST_GAP_M:
                if len(segment) >= MIN_ROUTE_LENGTH:
                    routes.append(group.iloc[segment].copy())
                segment = [i]
            else:
                segment.append(i)

        if len(segment) >= MIN_ROUTE_LENGTH:
            routes.append(group.iloc[segment].copy())

    return routes


def route_color_palette(n):
    """Return n visually distinct hex colors."""
    palette = [
        "#E63946", "#2A9D8F", "#E9C46A", "#F4A261", "#264653",
        "#6A0572", "#0077B6", "#80B918", "#F72585", "#7209B7",
        "#3A86FF", "#FB8500", "#06D6A0", "#EF233C", "#8338EC",
        "#118AB2", "#FFB703", "#D62828", "#023047", "#219EBC",
    ]
    return [palette[i % len(palette)] for i in range(n)]


def heading_arrow_html(heading_deg):
    """Return a tiny SVG arrow showing camera heading."""
    rad = math.radians(heading_deg - 90)
    dx = round(math.cos(rad) * 10, 1)
    dy = round(math.sin(rad) * 10, 1)
    return (
        f'<svg width="24" height="24" viewBox="-12 -12 24 24">'
        f'<line x1="0" y1="0" x2="{dx}" y2="{dy}" stroke="#333" stroke-width="2" '
        f'marker-end="url(#arr)"/>'
        f'<defs><marker id="arr" markerWidth="4" markerHeight="4" refX="2" refY="2" orient="auto">'
        f'<path d="M0,0 L4,2 L0,4 Z" fill="#333"/></marker></defs>'
        f'</svg>'
    )


def make_popup(row, route_idx=None):
    """Build an HTML popup for a single image point."""
    img_name = str(row.get("image", "")).split("/")[-1]
    heading = float(row.get("heading", row.get("azimuth", 0)))
    arrow = heading_arrow_html(heading)
    route_label = f"Route #{route_idx + 1}" if route_idx is not None else "—"
    html = f"""
    <div style="font-family:sans-serif;font-size:12px;min-width:220px;">
      <b style="font-size:13px;">Image ID {row.get('id','?')}</b><br>
      <span style="color:#555">{img_name}</span><br><hr style="margin:4px 0">
      <table style="width:100%;border-collapse:collapse">
        <tr><td style="color:#888">Route</td><td><b>{route_label}</b></td></tr>
        <tr><td style="color:#888">Period</td><td>{row.get('period','?')}</td></tr>
        <tr><td style="color:#888">Taken on</td><td>{str(row.get('taken_on','?'))[:19]}</td></tr>
        <tr><td style="color:#888">Lat / Lon</td><td>{float(row['lat']):.6f}, {float(row['lon']):.6f}</td></tr>
        <tr><td style="color:#888">Heading</td><td>{heading:.1f}°</td></tr>
        <tr><td style="color:#888">Neighbourhood</td><td>{row.get('neighbourhood','?')}</td></tr>
        <tr><td style="color:#888">Borough</td><td>{row.get('borough','?')}</td></tr>
      </table>
      <div style="margin-top:6px;display:flex;align-items:center;gap:6px">
        <span style="color:#888;font-size:11px">Camera heading:</span>{arrow}
      </div>
    </div>
    """
    return folium.Popup(html, max_width=280)


def build_map(df, routes):
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=17,
        tiles="CartoDB dark_matter",
    )

    # ── Add a light base layer option ──────────────────────────────────────
    folium.TileLayer("CartoDB positron", name="Light basemap").add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    # ── Route polylines ─────────────────────────────────────────────────────
    colors = route_color_palette(len(routes))
    route_layer = folium.FeatureGroup(name=f"Routes ({len(routes)} segments)", show=True)

    for i, route_df in enumerate(routes):
        color = colors[i]
        coords = list(zip(route_df["lat"], route_df["lon"]))

        folium.PolyLine(
            coords,
            color=color,
            weight=4,
            opacity=0.75,
            tooltip=f"Route {i+1} — {len(route_df)} images",
        ).add_to(route_layer)

        # Start marker
        start = route_df.iloc[0]
        folium.CircleMarker(
            [start["lat"], start["lon"]],
            radius=7,
            color="white",
            fill=True,
            fill_color=color,
            fill_opacity=1.0,
            tooltip=f"Route {i+1} start",
        ).add_to(route_layer)

    route_layer.add_to(m)

    # ── Individual image dots ───────────────────────────────────────────────
    # Build a lookup: image id → route index
    id_to_route = {}
    for i, route_df in enumerate(routes):
        for rid in route_df["id"]:
            id_to_route[rid] = i

    img_layer = folium.FeatureGroup(name=f"Images ({len(df)} total)", show=True)

    for _, row in df.iterrows():
        route_idx = id_to_route.get(row["id"])
        color = colors[route_idx] if route_idx is not None else "#aaaaaa"

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=1,
            popup=make_popup(row, route_idx),
            tooltip=f"ID {row['id']} — click for details",
        ).add_to(img_layer)

    img_layer.add_to(m)

    # ── Stats panel (top-right) ─────────────────────────────────────────────
    n_orphans = len(df) - sum(len(r) for r in routes)
    route_lengths = sorted([len(r) for r in routes], reverse=True)
    stats_rows = "".join(
        f"<tr><td style='padding:2px 8px 2px 0;color:#aaa'>Route {i+1}</td>"
        f"<td style='padding:2px 0'><b>{colors[i]}</b> &nbsp; {len(routes[i])} images</td></tr>"
        for i in range(min(10, len(routes)))
    )
    if len(routes) > 10:
        stats_rows += f"<tr><td colspan=2 style='color:#aaa'>… and {len(routes)-10} more routes</td></tr>"

    panel_html = f"""
    <div style="position:fixed;top:10px;right:10px;z-index:9999;
         background:rgba(20,20,30,0.92);color:#eee;padding:14px 18px;
         border-radius:10px;font-family:sans-serif;font-size:12px;
         max-width:240px;box-shadow:0 2px 12px rgba(0,0,0,0.5)">
      <div style="font-size:15px;font-weight:600;margin-bottom:8px">
        Image Route Map
      </div>
      <table>
        <tr><td style='color:#aaa;padding-right:8px'>Total images</td><td><b>{len(df)}</b></td></tr>
        <tr><td style='color:#aaa;padding-right:8px'>Route segments</td><td><b>{len(routes)}</b></td></tr>
        <tr><td style='color:#aaa;padding-right:8px'>Orphan images</td><td><b>{n_orphans}</b></td></tr>
        <tr><td style='color:#aaa;padding-right:8px'>Longest route</td><td><b>{max(route_lengths) if route_lengths else 0} imgs</b></td></tr>
      </table>
      <hr style='border-color:#444;margin:8px 0'>
      <div style='font-size:11px;color:#888;margin-bottom:4px'>Route breakdown (top 10):</div>
      <table>{stats_rows}</table>
      <hr style='border-color:#444;margin:8px 0'>
      <div style='font-size:10px;color:#666'>Click any dot for image details.<br>
      Use layer control (top-left) to toggle routes/images.</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(panel_html))

    folium.LayerControl(collapsed=False).add_to(m)

    return m


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_routes.py <your_images.csv>")
        print("\nRunning on built-in demo data…\n")
        csv_path = None
    else:
        csv_path = sys.argv[1]
        if not os.path.exists(csv_path):
            print(f"Error: file not found — {csv_path}")
            sys.exit(1)

    if csv_path:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} images from {csv_path}")
    else:
        # Demo data matching the columns in the project CSV
        import io
        demo_csv = """id,azimuth,pitch,roll,heading,android_id,accuracy,speed,zendrive_id,vehicle_id,snapped_distance,position,orientation,taken_on,image,lat,lon,snapped_lat,snapped_lon,neighbourhood,borough,timestamp,dt,period,geometry
2069249,60.42,-3.92,-87.81,60.42,79ddade2485edbde,-1.0,6.16,36.0,13.0,8.96032,3,60,2016-06-28 10:00:46-04:00,0/20160628/79ddade2485edbde/cds-img-19.jpg,40.7286184,-73.994477,40.72855,-73.994534,West Village,Manhattan,1467122446,2016-06-28 10:00:46-04:00,morning,POINT (-73.994477 40.7286184)
2069250,60.42,-3.92,-87.81,60.42,79ddade2485edbde,-1.0,6.16,36.0,13.0,8.96032,3,60,2016-06-28 10:00:56-04:00,0/20160628/79ddade2485edbde/cds-img-20.jpg,40.7286900,-73.994200,40.72855,-73.994100,West Village,Manhattan,1467122456,2016-06-28 10:00:56-04:00,morning,POINT (-73.994200 40.7286900)
2069251,60.42,-3.92,-87.81,60.42,79ddade2485edbde,-1.0,6.16,36.0,13.0,8.96032,3,60,2016-06-28 10:01:06-04:00,0/20160628/79ddade2485edbde/cds-img-21.jpg,40.7287600,-73.993920,40.72855,-73.993800,West Village,Manhattan,1467122466,2016-06-28 10:01:06-04:00,morning,POINT (-73.993920 40.7287600)
28423755,67.08,12.89,-85.41,67.08,13a07ea2578fefb5,-1.0,7.75,47.0,1.0,1.6736,4,67,2016-08-10 16:37:58-04:00,0/20160810/13a07ea2578fefb5/cds-img-108.jpg,40.7280032,-73.9933867,40.72799,-73.993396,West Village,Manhattan,1470861478,2016-08-10 16:37:58-04:00,afternoon,POINT (-73.9933867 40.7280032)
28423756,67.08,12.89,-85.41,67.08,13a07ea2578fefb5,-1.0,7.75,47.0,1.0,1.6736,4,67,2016-08-10 16:38:08-04:00,0/20160810/13a07ea2578fefb5/cds-img-109.jpg,40.7281000,-73.993100,40.72799,-73.993100,West Village,Manhattan,1470861488,2016-08-10 16:38:08-04:00,afternoon,POINT (-73.993100 40.7281000)
28423757,67.08,12.89,-85.41,67.08,13a07ea2578fefb5,-1.0,7.75,47.0,1.0,1.6736,4,67,2016-08-10 16:38:18-04:00,0/20160810/13a07ea2578fefb5/cds-img-110.jpg,40.7282000,-73.992800,40.72799,-73.992800,West Village,Manhattan,1470861498,2016-08-10 16:38:18-04:00,afternoon,POINT (-73.992800 40.7282000)
34077750,65.1,-0.69,-88.04,65.1,b6d8aab376c7ce54,-1.0,0.02,48.0,4.0,1.3007,4,65,2016-08-31 12:18:18-04:00,0/20160831/b6d8aab376c7ce54/cds-img-17.jpg,40.7288944,-74.00059509999998,40.728884,-74.000603,West Village,Manhattan,1472660298,2016-08-31 12:18:18-04:00,morning,POINT (-74.00059509999998 40.7288944)
34077751,65.1,-0.69,-88.04,65.1,b6d8aab376c7ce54,-1.0,0.02,48.0,4.0,1.3007,4,65,2016-08-31 12:18:28-04:00,0/20160831/b6d8aab376c7ce54/cds-img-18.jpg,40.7289500,-74.000300,40.728884,-74.000300,West Village,Manhattan,1472660308,2016-08-31 12:18:28-04:00,morning,POINT (-74.000300 40.7289500)
34077752,65.1,-0.69,-88.04,65.1,b6d8aab376c7ce54,-1.0,0.02,48.0,4.0,1.3007,4,65,2016-08-31 12:18:38-04:00,0/20160831/b6d8aab376c7ce54/cds-img-19.jpg,40.7290100,-74.000000,40.728884,-74.000000,West Village,Manhattan,1472660318,2016-08-31 12:18:38-04:00,morning,POINT (-74.000000 40.7290100)
1691371,22.79,-1.34,-71.06,22.79,1ccda49b184cb0ef,-1.0,7.29,42.0,11.0,5.42247,3,23,2016-06-30 11:20:24-04:00,0/20160630/1ccda49b184cb0ef/cds-img-105.jpg,40.7322717,-74.00061590000001,40.732247,-74.000559,West Village,Manhattan,1467300024,2016-06-30 11:20:24-04:00,morning,POINT (-74.00061590000001 40.7322717)
1691372,22.79,-1.34,-71.06,22.79,1ccda49b184cb0ef,-1.0,7.29,42.0,11.0,5.42247,3,23,2016-06-30 11:20:34-04:00,0/20160630/1ccda49b184cb0ef/cds-img-106.jpg,40.7323500,-74.000400,40.732247,-74.000400,West Village,Manhattan,1467300034,2016-06-30 11:20:34-04:00,morning,POINT (-74.000400 40.7323500)
1691373,22.79,-1.34,-71.06,22.79,1ccda49b184cb0ef,-1.0,7.29,42.0,11.0,5.42247,3,23,2016-06-30 11:20:44-04:00,0/20160630/1ccda49b184cb0ef/cds-img-107.jpg,40.7324300,-74.000200,40.732247,-74.000200,West Village,Manhattan,1467300044,2016-06-30 11:20:44-04:00,morning,POINT (-74.000200 40.7324300)
35550109,65.88,2.63,-82.45,65.88,19919ff6db95782d,-1.0,0.02,58.0,17.0,11.2492,4,66,2016-08-29 12:19:36-04:00,0/20160829/19919ff6db95782d/cds-img-22.jpg,40.7277166,-73.9961558,40.727661,-73.996043,West Village,Manhattan,1472487576,2016-08-29 12:19:36-04:00,morning,POINT (-73.9961558 40.7277166)
35550110,65.88,2.63,-82.45,65.88,19919ff6db95782d,-1.0,0.02,58.0,17.0,11.2492,4,66,2016-08-29 12:19:46-04:00,0/20160829/19919ff6db95782d/cds-img-23.jpg,40.7277900,-73.995900,40.727661,-73.995900,West Village,Manhattan,1472487586,2016-08-29 12:19:46-04:00,morning,POINT (-73.995900 40.7277900)
35550111,65.88,2.63,-82.45,65.88,19919ff6db95782d,-1.0,0.02,58.0,17.0,11.2492,4,66,2016-08-29 12:19:56-04:00,0/20160829/19919ff6db95782d/cds-img-24.jpg,40.7278600,-73.995650,40.727661,-73.995650,West Village,Manhattan,1472487596,2016-08-29 12:19:56-04:00,morning,POINT (-73.995650 40.7278600)
"""
        df = pd.read_csv(io.StringIO(demo_csv))
        print(f"Using demo data: {len(df)} images across {df['android_id'].nunique()} vehicles")

    # Validate required columns
    required = ["id", "lat", "lon", "heading"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Error: CSV missing required columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["heading"] = pd.to_numeric(df["heading"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    print(f"Inferring driving routes…")
    routes = infer_routes(df)
    print(f"Found {len(routes)} route segments "
          f"(time gap ≤{ROUTE_TIME_GAP_SEC}s, dist gap ≤{ROUTE_DIST_GAP_M}m, min {MIN_ROUTE_LENGTH} images)")
    for i, r in enumerate(routes[:10]):
        print(f"  Route {i+1}: {len(r)} images, vehicle {r.iloc[0].get('android_id','?')[:8]}…")

    print(f"Building map…")
    m = build_map(df, routes)

    out = Path(OUTPUT_FILE)
    m.save(str(out))
    print(f"\nMap saved → {out.resolve()}")
    print("Opening in browser…")
    webbrowser.open(f"file://{out.resolve()}")


if __name__ == "__main__":
    main()