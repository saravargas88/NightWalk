"""heatmap.py
Generates an interactive side-by-side heatmap comparing the geographic
concentration of day vs night images, using final labels from labels_final.csv.
"""
import folium
import pandas as pd
from folium.plugins import HeatMap
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import CSV_PATH

LABELS_CSV = Path("labels_final.csv")
OUT_HTML   = Path("heatmap.html")

# ── Load & join ───────────────────────────────────────────────────────────────
labels = pd.read_csv(LABELS_CSV)[["image", "final_label"]]
meta   = pd.read_csv(CSV_PATH)[["image", "lat", "lon"]]
df     = labels.merge(meta, on="image", how="left").dropna(subset=["lat", "lon"])

night = df[df["final_label"] == "night"]
day   = df[df["final_label"] == "day"]

print(f"Night images with coords: {len(night):,}")
print(f"Day   images with coords: {len(day):,}")

center = [df["lat"].mean(), df["lon"].mean()]

# ── Heatmap parameters ────────────────────────────────────────────────────────
# Night has far fewer images so boost its radius so it's visible alongside day
HEAT_PARAMS_DAY   = dict(radius=10, blur=12, max_zoom=16, min_opacity=0.3)
HEAT_PARAMS_NIGHT = dict(radius=14, blur=16, max_zoom=16, min_opacity=0.5)

night_coords = night[["lat", "lon"]].values.tolist()
day_coords   = day[["lat", "lon"]].values.tolist()

# ── Build map with layer toggle ───────────────────────────────────────────────
m = folium.Map(location=center, zoom_start=15, tiles="CartoDB dark_matter")

day_layer = folium.FeatureGroup(name=f"☀️ Day ({len(day):,} images)", show=True)
HeatMap(day_coords, name="Day", gradient={"0.2":"#1a6b1a","0.5":"#4caf50","1.0":"#b9f6ca"},
        **HEAT_PARAMS_DAY).add_to(day_layer)
day_layer.add_to(m)

night_layer = folium.FeatureGroup(name=f"🌙 Night ({len(night):,} images)", show=True)
HeatMap(night_coords, name="Night", gradient={"0.2":"#1a1a6b","0.5":"#3f51b5","1.0":"#b3c5ff"},
        **HEAT_PARAMS_NIGHT).add_to(night_layer)
night_layer.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_html = f"""
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:#1a1a1a;
            border:1px solid #444;border-radius:8px;padding:12px 16px;font-family:sans-serif;
            font-size:13px;color:#eee;line-height:1.8;">
  <b>Image concentration</b><br>
  <span style="color:#4caf50">&#9632;</span> Day &nbsp; {len(day):,} images<br>
  <span style="color:#3f51b5">&#9632;</span> Night &nbsp; {len(night):,} images<br>
  <span style="font-size:11px;color:#888">Toggle layers top-right</span>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

m.save(str(OUT_HTML))
print(f"\nSaved {OUT_HTML}")
print("Open in browser — use the layer toggle (top right) to show/hide day and night.")
