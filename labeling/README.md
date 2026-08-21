**overview**

- **`day_night_classification/`** — the authoritative pipeline that splits all images into night/day buckets (astronomical sunrise/sunset + brightness), plus manual review tooling. Produces `labels_final.csv`.
  - `brightness_scorer.py` — scores every image by greyscale brightness
  - `finalize_labels.py` — combines timestamp + brightness rules → final night/day labels
  - `review_tool.py` / `review_tool.html` — manual correction interface for edge cases
  - `heatmap.py` / `heatmap.html` — visualizes label distribution
  - `labels_final.csv`, `all_brightness.csv`, `manual_corrections.csv` — outputs

- **`brightness_metrics/`** — compares different ways of measuring nighttime brightness from paired images; its output (`paired_dataset_with_brightness.csv`) is what `model-training/` actually trains on.
  - `run_brightness_metric_experiments.py` — runs the comparison
  - `brightness_target_notes.md` — notes on findings

- **`legacy_brightness_exploration/`** — an earlier, exploratory pass at brightness scoring, kept for reference (not used by anything downstream).
  - `ground_truth.py`, `darkest_viewer.html`, `evening_viewer.html`
- must use server to use htmls. 