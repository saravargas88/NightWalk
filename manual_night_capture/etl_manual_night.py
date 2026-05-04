"""ETL for manually-captured iPhone night photos around Washington Square Park.

Extracts EXIF GPS + timestamp from photos in `manual_night_capture/raw/`,
converts/resizes them to the dataset's standard 1632x1224 JPEG format, joins
them against the targets grid, builds a day-night pairs CSV against the
existing dataset, and appends `night` rows to `label_split/labels_final.csv`.

Run from repository root:

    python manual_night_capture/etl_manual_night.py
    python manual_night_capture/etl_manual_night.py --dry-run
    python manual_night_capture/etl_manual_night.py --skip-pairs

Idempotent: re-running on the same `raw/` directory will not duplicate rows
in any output CSV, will not re-process images already present in
`urban-mosaic/manual_night/`, and will never overwrite an existing
`labels_final.csv.bak`.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.spatial import cKDTree

try:
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
except ImportError:
    pillow_heif = None  # noqa: N816 — handled at runtime if HEIC is encountered


# ── Paths (relative to repo root; the script must be run from there) ──────────
INPUT_DIR      = Path("manual_night_capture/raw")
OUTPUT_IMG_DIR = Path("urban-mosaic/manual_night")
TARGETS_CSV    = Path("manual_night_capture/targets.csv")
METADATA_OUT   = Path("manual_night_capture/manual_night_metadata.csv")
PAIRS_OUT      = Path("manual_night_capture/manual_pairs.csv")
MASTER_CSV     = Path("urban-mosaic/washington-square.csv")
LABELS_CSV     = Path("label_split/labels_final.csv")
LABELS_BAK     = Path("label_split/labels_final.csv.bak")
SKIPPED_LOG    = Path("manual_night_capture/skipped.log")
DAY_IMG_DIR    = Path("urban-mosaic/washington-square")

# ── Constants ─────────────────────────────────────────────────────────────────
OUT_W, OUT_H        = 1632, 1224
JPEG_QUALITY        = 92
SUPPORTED_EXTS      = {".jpg", ".jpeg", ".heic"}  # compared case-insensitively

BBOX                = (40.727, 40.735, -74.003, -73.992)  # lat_min, lat_max, lon_min, lon_max
GRID_SIZE_M         = 25.0
METRES_PER_DEG_LAT  = 111_000.0
METRES_PER_DEG_LON  = 84_000.0
PAIR_FLAG_DIST_M    = 30.0

REL_NIGHT_PARENT    = "manual_night"  # path prefix used in output CSVs

# EXIF IFD tag IDs (use literals so we don't depend on Pillow's IFD enum
# being available across versions).
GPS_IFD_TAG  = 0x8825
EXIF_IFD_TAG = 0x8769
TAG_DATETIME_ORIGINAL = 0x9003

# Required input-CSV columns. Anything missing is a hard error.
REQUIRED_TARGET_COLS = {
    "cell_id", "lat_center", "lon_center",
    "modal_heading", "heading_confidence",
    "target_day_image_hint", "cell_category",
}
REQUIRED_MASTER_COLS = {"image", "lat", "lon"}
REQUIRED_LABEL_COLS  = {"image", "final_label"}


# ── Tiny utilities ────────────────────────────────────────────────────────────
def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest manually-captured iPhone night photos into the "
                    "NightWalk dataset.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Process and report but do not write CSVs or copy images.",
    )
    p.add_argument(
        "--skip-pairs", action="store_true",
        help="Only build metadata; skip pair matching against the day pool.",
    )
    return p.parse_args()


def dms_to_decimal(dms, ref) -> float:
    """Convert ((deg, min, sec), ref) EXIF GPS to a signed decimal degree."""
    d = float(dms[0])
    m = float(dms[1])
    s = float(dms[2])
    val = d + m / 60.0 + s / 3600.0
    ref_str = ref.decode() if isinstance(ref, bytes) else ref
    if ref_str.upper() in ("S", "W"):
        val = -val
    return val


def extract_exif(img: Image.Image) -> Tuple[Optional[float], Optional[float], Optional[datetime], Optional[str]]:
    """Returns (lat, lon, dt, error_reason). On success error_reason is None."""
    try:
        exif = img.getexif()
    except Exception as e:  # pragma: no cover — pathological file
        return None, None, None, f"getexif failed: {e}"
    if not exif:
        return None, None, None, "no EXIF block"

    gps = exif.get_ifd(GPS_IFD_TAG)
    if not gps:
        return None, None, None, "no GPS EXIF"

    lat_dms = gps.get(2)   # GPSLatitude
    lat_ref = gps.get(1)   # GPSLatitudeRef
    lon_dms = gps.get(4)   # GPSLongitude
    lon_ref = gps.get(3)   # GPSLongitudeRef
    if not (lat_dms and lat_ref and lon_dms and lon_ref):
        return None, None, None, "incomplete GPS EXIF"

    try:
        lat = dms_to_decimal(lat_dms, lat_ref)
        lon = dms_to_decimal(lon_dms, lon_ref)
    except Exception as e:
        return None, None, None, f"GPS parse failed: {e}"

    dt: Optional[datetime] = None
    exif_ifd = exif.get_ifd(EXIF_IFD_TAG)
    dto = exif_ifd.get(TAG_DATETIME_ORIGINAL) if exif_ifd else None
    if dto is None:
        # Some cameras only put DateTime in the top-level IFD0.
        dto = exif.get(0x0132)
    if dto:
        s = dto.decode() if isinstance(dto, bytes) else str(dto)
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(s.strip().split(".")[0], fmt)
                break
            except ValueError:
                continue

    return lat, lon, dt, None


def crop_to_fill(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize to fit-larger-dim, then center-crop to (target_w, target_h)."""
    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("zero-size image")
    scale = max(target_w / w, target_h / h)
    new_w = max(target_w, int(round(w * scale)))
    new_h = max(target_h, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top  = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def grey_of_pil(img: Image.Image) -> float:
    """Apply the canonical NightWalk brightness formula to an in-memory image."""
    return round(np.array(img.convert("L")).mean(), 2)


def grey_of_path(path: Path) -> float:
    """Canonical NightWalk brightness formula (verbatim from brightness_scorer.py)."""
    img = Image.open(path).convert("L")
    return round(np.array(img).mean(), 2)


def lat_lon_to_cell_id(lat: float, lon: float) -> str:
    """Project a (lat, lon) into the same grid used by build_targets.py."""
    lat_min, _, lon_min, _ = BBOX
    cell_h = GRID_SIZE_M / METRES_PER_DEG_LAT
    cell_w = GRID_SIZE_M / METRES_PER_DEG_LON
    row = int(np.floor((lat - lat_min) / cell_h))
    col = int(np.floor((lon - lon_min) / cell_w))
    return f"r{row:02d}_c{col:02d}"


# ── Loaders with schema validation ────────────────────────────────────────────
def load_targets() -> pd.DataFrame:
    if not TARGETS_CSV.exists():
        fail(f"Missing {TARGETS_CSV}. Run build_targets.py first.")
    df = pd.read_csv(TARGETS_CSV)
    missing = REQUIRED_TARGET_COLS - set(df.columns)
    if missing:
        fail(f"{TARGETS_CSV} is missing columns: {sorted(missing)}. "
             f"Found: {sorted(df.columns)}")
    return df


def load_master() -> pd.DataFrame:
    if not MASTER_CSV.exists():
        fail(f"Missing {MASTER_CSV}.")
    df = pd.read_csv(MASTER_CSV, low_memory=False)
    missing = REQUIRED_MASTER_COLS - set(df.columns)
    if missing:
        fail(f"{MASTER_CSV} is missing columns: {sorted(missing)}. "
             f"Found: {sorted(df.columns)}")
    return df


def load_labels() -> pd.DataFrame:
    if not LABELS_CSV.exists():
        fail(f"Missing {LABELS_CSV}.")
    df = pd.read_csv(LABELS_CSV, low_memory=False)
    missing = REQUIRED_LABEL_COLS - set(df.columns)
    if missing:
        fail(f"{LABELS_CSV} is missing columns: {sorted(missing)}. "
             f"Found: {sorted(df.columns)}")
    return df


# ── Per-image processing ──────────────────────────────────────────────────────
def output_filename_for(input_path: Path) -> str:
    """Output filename: keep the original stem, force lowercase .jpg."""
    return f"{input_path.stem}.jpg"


def process_image(
    input_path: Path,
    output_path: Path,
    dry_run: bool,
) -> Tuple[Optional[float], Optional[float], Optional[datetime], Optional[float], Optional[str]]:
    """Returns (lat, lon, dt, night_grey, error_reason).

    Reads EXIF, converts/resizes, optionally writes the output JPEG, and
    computes brightness on the resized output. On any error returns
    (None, None, None, None, reason).
    """
    if input_path.suffix.lower() in (".heic",) and pillow_heif is None:
        return None, None, None, None, "HEIC encountered but pillow-heif not installed"

    try:
        with Image.open(input_path) as raw:
            raw.load()
            lat, lon, dt, exif_err = extract_exif(raw)
            if exif_err is not None:
                return None, None, None, None, exif_err

            oriented = ImageOps.exif_transpose(raw).convert("RGB")
    except Exception as e:
        return None, None, None, None, f"open failed: {e}"

    try:
        cropped = crop_to_fill(oriented, OUT_W, OUT_H)
    except Exception as e:
        return None, None, None, None, f"resize failed: {e}"

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cropped.save(output_path, format="JPEG", quality=JPEG_QUALITY)
        except Exception as e:
            return None, None, None, None, f"save failed: {e}"
        night_grey = grey_of_path(output_path)
    else:
        night_grey = grey_of_pil(cropped)

    return lat, lon, dt, night_grey, None


# ── Pair building ─────────────────────────────────────────────────────────────
def build_pairs(
    new_records: list[dict],
    master_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build manual_pairs.csv content (no I/O)."""
    night_set = set(
        labels_df.loc[labels_df["final_label"] == "night", "image"].astype(str)
    )
    day_pool = master_df.loc[
        ~master_df["image"].astype(str).isin(night_set), ["image", "lat", "lon"]
    ].copy()
    day_pool["lat"] = pd.to_numeric(day_pool["lat"], errors="coerce")
    day_pool["lon"] = pd.to_numeric(day_pool["lon"], errors="coerce")
    day_pool = day_pool.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if day_pool.empty:
        fail("Day pool is empty after filtering out night-labelled images.")

    print(f"  day pool size: {len(day_pool):,} (master rows minus "
          f"{len(night_set):,} night-labelled images).")

    tree = cKDTree(day_pool[["lat", "lon"]].to_numpy())
    query_pts = np.array([[r["lat"], r["lon"]] for r in new_records])
    distances, indices = tree.query(query_pts, k=1)
    dist_m = distances * METRES_PER_DEG_LAT  # 111_000

    grey_cache: dict[str, float] = {}
    rows = []
    for rec, idx, d in zip(new_records, indices, dist_m):
        day_row = day_pool.iloc[int(idx)]
        day_image = str(day_row["image"])
        if day_image not in grey_cache:
            day_path = DAY_IMG_DIR / day_image
            if day_path.exists():
                try:
                    grey_cache[day_image] = grey_of_path(day_path)
                except Exception:
                    grey_cache[day_image] = float("nan")
            else:
                grey_cache[day_image] = float("nan")
        rows.append({
            "day_image":   day_image,
            "night_image": rec["image_rel"],
            "day_grey":    grey_cache[day_image],
            "night_grey":  rec["night_grey"],
            "dist_m":      round(float(d), 2),
            "flagged":     bool(d > PAIR_FLAG_DIST_M),
        })
    return pd.DataFrame(rows)


# ── Idempotent CSV writers ────────────────────────────────────────────────────
def upsert_csv(out_path: Path, new_df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """Replace rows in `out_path` whose `key_col` is in new_df, then append.

    If `out_path` doesn't exist, just write `new_df`. Returns the final dataframe.
    """
    if out_path.exists():
        existing = pd.read_csv(out_path)
        if key_col in existing.columns and not new_df.empty:
            existing = existing.loc[~existing[key_col].isin(new_df[key_col])].copy()
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()
    combined.to_csv(out_path, index=False)
    return combined


def append_labels(
    new_records: list[dict],
    labels_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:
    """Append night labels for any new image not already present.

    Backs up labels_final.csv -> labels_final.csv.bak the first time only.
    Returns (final_df, n_appended).
    """
    if not LABELS_BAK.exists():
        shutil.copy2(LABELS_CSV, LABELS_BAK)
        print(f"  wrote backup: {LABELS_BAK}")
    else:
        print(f"  backup already exists ({LABELS_BAK}); leaving it alone.")

    existing_images = set(labels_df["image"].astype(str))
    rows = []
    for rec in new_records:
        if rec["image_rel"] in existing_images:
            continue
        dt: Optional[datetime] = rec.get("dt")
        rows.append({
            "image":           rec["image_rel"],
            "grey":            rec["night_grey"],
            "hour":            dt.hour if dt is not None else None,
            "taken_on_short":  dt.strftime("%Y-%m-%d %H:%M:%S") if dt is not None else None,
            "period":          "night",
            "machine_label":   "night",
            "source":          "manual_capture",
            "final_label":     "night",
        })
    if not rows:
        return labels_df, 0

    new_rows = pd.DataFrame(rows)
    # Align to existing schema; pandas fills missing cols with NaN.
    combined = pd.concat([labels_df, new_rows], ignore_index=True)
    # Reorder to match original schema, keeping any extra columns at the end.
    cols = list(labels_df.columns)
    extras = [c for c in combined.columns if c not in cols]
    combined = combined[cols + extras]
    combined.to_csv(LABELS_CSV, index=False)
    return combined, len(new_rows)


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(
    n_total: int,
    accepted: list[dict],
    rejected: list[Tuple[str, str]],
    skipped_existing: list[str],
    pairs_df: Optional[pd.DataFrame],
    targets_df: pd.DataFrame,
    master_df: Optional[pd.DataFrame],
    labels_df: pd.DataFrame,
) -> None:
    print("\n=== Summary ===")
    print(f"Raw photos found:    {n_total}")
    print(f"Accepted (new):      {len(accepted)}")
    print(f"Skipped (already in OUTPUT_IMG_DIR): {len(skipped_existing)}")
    print(f"Rejected:            {len(rejected)}")
    if rejected:
        by_reason: dict[str, int] = defaultdict(int)
        for _, reason in rejected:
            by_reason[reason] += 1
        for reason, n in sorted(by_reason.items(), key=lambda x: -x[1]):
            print(f"  {n:>4}  {reason}")

    if accepted:
        greys = np.array([r["night_grey"] for r in accepted], dtype=float)
        print("\nBrightness (new photos):")
        print(f"  min={greys.min():.2f}  median={np.median(greys):.2f}  "
              f"mean={greys.mean():.2f}  max={greys.max():.2f}")

    if pairs_df is not None and not pairs_df.empty:
        d = pairs_df["dist_m"].to_numpy()
        flagged = int(pairs_df["flagged"].sum())
        print("\nPair distances:")
        print(f"  median={np.median(d):.1f} m  max={d.max():.1f} m  "
              f"flagged (>{PAIR_FLAG_DIST_M:g} m)={flagged}")

    if accepted:
        cell_to_cat = dict(zip(targets_df["cell_id"], targets_df["cell_category"]))
        cell_ids = [r["cell_id"] for r in accepted if r.get("cell_id")]
        unique_cells = set(cell_ids)
        print(f"\nCells covered: {len(unique_cells)} unique")
        cat_counts: dict[str, int] = defaultdict(int)
        for cid in unique_cells:
            cat_counts[cell_to_cat.get(cid, "unknown")] += 1
        for cat in ("priority", "validation", "secondary", "unknown"):
            if cat_counts.get(cat):
                print(f"  {cat:<11} {cat_counts[cat]}")

        # Validation-cell brightness comparison
        val_cells = sorted({
            r["cell_id"] for r in accepted
            if r.get("cell_id") and cell_to_cat.get(r["cell_id"]) == "validation"
        })
        if val_cells and master_df is not None:
            print("\nValidation cells touched:")
            existing_night = labels_df.loc[
                labels_df["final_label"] == "night", ["image", "grey"]
            ]
            night_with_pos = existing_night.merge(
                master_df[["image", "lat", "lon"]], on="image", how="inner"
            )
            night_with_pos = night_with_pos.dropna(subset=["lat", "lon", "grey"])
            night_with_pos["cell_id"] = [
                lat_lon_to_cell_id(la, lo)
                for la, lo in zip(night_with_pos["lat"], night_with_pos["lon"])
            ]
            for cid in val_cells:
                manual_greys = [r["night_grey"] for r in accepted if r.get("cell_id") == cid]
                manual_mean = float(np.mean(manual_greys))
                existing_in_cell = night_with_pos.loc[
                    night_with_pos["cell_id"] == cid, "grey"
                ].astype(float).tolist()
                if existing_in_cell:
                    existing_str = ", ".join(f"{g:.2f}" for g in existing_in_cell)
                    diff = abs(manual_mean - float(np.mean(existing_in_cell)))
                    print(f"  {cid}  manual={manual_mean:.2f}  "
                          f"existing=[{existing_str}]  |Δ|={diff:.2f}")
                else:
                    print(f"  {cid}  manual={manual_mean:.2f}  "
                          f"existing=[]  |Δ|=n/a")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    if not INPUT_DIR.exists():
        fail(f"Input directory {INPUT_DIR} does not exist.")

    # Validate inputs early so failures happen before any image work.
    print("Loading inputs…")
    targets_df = load_targets()
    labels_df  = load_labels()
    master_df  = None  # only loaded if we need pairs / validation summary
    print(f"  targets: {len(targets_df):,} cells")
    print(f"  labels:  {len(labels_df):,} rows")

    # Build a lookup from cell_id to its full target row.
    targets_by_id = targets_df.set_index("cell_id").to_dict(orient="index")

    # Enumerate inputs.
    raw_files = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    print(f"\nFound {len(raw_files)} raw photo(s) in {INPUT_DIR}.")

    if not args.dry_run:
        OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    # We rewrite skipped.log fresh on each run so the file always reflects the
    # latest run; CSV outputs are still merged idempotently.
    skipped_lines: list[str] = []

    accepted: list[dict] = []
    rejected: list[Tuple[str, str]] = []
    skipped_existing: list[str] = []

    for src in raw_files:
        out_name = output_filename_for(src)
        out_path = OUTPUT_IMG_DIR / out_name
        rel_path = f"{REL_NIGHT_PARENT}/{out_name}"

        if out_path.exists():
            print(f"  [skip-existing] {src.name} -> {rel_path}")
            skipped_existing.append(rel_path)
            continue

        lat, lon, dt, night_grey, err = process_image(src, out_path, args.dry_run)
        if err is not None:
            print(f"  [reject:{err}] {src.name}")
            rejected.append((src.name, err))
            skipped_lines.append(f"{src.name}\t{err}")
            continue

        lat_min, lat_max, lon_min, lon_max = BBOX
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            err = f"out_of_bbox lat={lat:.6f} lon={lon:.6f}"
            print(f"  [reject:{err}] {src.name}")
            rejected.append((src.name, "out_of_bbox"))
            skipped_lines.append(f"{src.name}\t{err}")
            # We may have already saved the JPEG; remove it to keep
            # OUTPUT_IMG_DIR aligned with the accepted set.
            if not args.dry_run and out_path.exists():
                try:
                    out_path.unlink()
                except OSError:
                    pass
            continue

        # Cell lookup (optional, never a rejection reason).
        cell_id = lat_lon_to_cell_id(lat, lon)
        target = targets_by_id.get(cell_id)
        rec = {
            "input_name": src.name,
            "image_rel":  rel_path,
            "lat":        lat,
            "lon":        lon,
            "dt":         dt,
            "night_grey": night_grey,
            "cell_id":              cell_id if target else "",
            "modal_heading":        target["modal_heading"]         if target else "",
            "heading_confidence":   target["heading_confidence"]    if target else "",
            "target_day_image_hint":target["target_day_image_hint"] if target else "",
            "cell_category":        target["cell_category"]         if target else "",
        }
        accepted.append(rec)
        print(f"  [ok] {src.name} -> {rel_path}  "
              f"lat={lat:.6f} lon={lon:.6f} grey={night_grey:.2f} "
              f"cell={rec['cell_id'] or '<none>'}")

    # Always (re)write skipped.log, even on dry-run, so the operator can see
    # exactly which inputs would be rejected.
    if not args.dry_run:
        SKIPPED_LOG.parent.mkdir(parents=True, exist_ok=True)
        SKIPPED_LOG.write_text(
            "# input_filename\treason\n" + "\n".join(skipped_lines) + ("\n" if skipped_lines else ""),
            encoding="utf-8",
        )

    # ── Build metadata CSV first ─────────────────────────────────────────────
    metadata_rows = [{
        "image":                 r["image_rel"],
        "lat":                   r["lat"],
        "lon":                   r["lon"],
        "taken_on":              r["dt"].isoformat() if r["dt"] is not None else "",
        "period":                "night",
        "cell_id":               r["cell_id"],
        "modal_heading":         r["modal_heading"],
        "heading_confidence":    r["heading_confidence"],
        "target_day_image_hint": r["target_day_image_hint"],
        "cell_category":         r["cell_category"],
    } for r in accepted]
    metadata_df = pd.DataFrame(
        metadata_rows,
        columns=[
            "image", "lat", "lon", "taken_on", "period",
            "cell_id", "modal_heading", "heading_confidence",
            "target_day_image_hint", "cell_category",
        ],
    )

    if not args.dry_run and not metadata_df.empty:
        upsert_csv(METADATA_OUT, metadata_df, key_col="image")
        print(f"\nWrote {METADATA_OUT}  (+{len(metadata_df)} new rows)")
    else:
        print(f"\n(dry-run) would write {len(metadata_df)} rows to {METADATA_OUT}")

    # ── Build pairs CSV ──────────────────────────────────────────────────────
    pairs_df: Optional[pd.DataFrame] = None
    if not args.skip_pairs and accepted:
        print("\nBuilding pairs against day pool…")
        master_df = load_master()
        pairs_df = build_pairs(accepted, master_df, labels_df)
        if not args.dry_run and not pairs_df.empty:
            upsert_csv(PAIRS_OUT, pairs_df, key_col="night_image")
            print(f"Wrote {PAIRS_OUT}  (+{len(pairs_df)} new rows)")
        else:
            print(f"(dry-run or no pairs) would write {len(pairs_df)} rows to {PAIRS_OUT}")
    elif args.skip_pairs:
        print("\n--skip-pairs set: skipping pair construction.")

    # Master is also useful for the validation-cell summary even if --skip-pairs.
    if master_df is None and accepted and any(
        r.get("cell_category") == "validation" for r in accepted
    ):
        print("\nLoading master CSV for validation-cell brightness summary…")
        master_df = load_master()

    # ── Append to labels_final.csv last ─────────────────────────────────────
    if not args.dry_run and accepted:
        print("\nUpdating labels_final.csv…")
        labels_df, n_appended = append_labels(accepted, labels_df)
        print(f"  appended {n_appended} new night row(s).")
    elif args.dry_run and accepted:
        already = set(labels_df["image"].astype(str))
        n_would = sum(1 for r in accepted if r["image_rel"] not in already)
        print(f"\n(dry-run) would append {n_would} row(s) to {LABELS_CSV}.")

    print_summary(
        n_total=len(raw_files),
        accepted=accepted,
        rejected=rejected,
        skipped_existing=skipped_existing,
        pairs_df=pairs_df,
        targets_df=targets_df,
        master_df=master_df,
        labels_df=labels_df,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
