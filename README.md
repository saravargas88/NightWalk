# NightWalk

Predicting nighttime street brightness from daytime street-level imagery.

## Project overview

This project explores whether daytime street-level images can be used to predict
how bright a street will be at night. The approach has two stages:

1. **Feature extraction** — use Grounding DINO to count objects in daytime images
   that correlate with nighttime illumination (trees, streetlights, storefronts,
   doorman buildings).
2. **Brightness prediction** — fine-tune an EfficientNet model to predict nighttime
   pixel brightness from daytime imagery, using day-night image pairs as training data.

This repository contains preliminary feasibility experiments for both stages.

## Dataset

~100,000 street-level images of Washington Square Park, NYC, captured via
car-mounted camera. Images span multiple times of day. Metadata includes GPS
coordinates, heading, and timestamp.

```
urban-mosaic/
└── washington-square/       # images
washington-square.csv        # metadata
```

## Repository structure

```
NightWalk/
├── config/
│   └── paths.py                  # central path constants — edit here if data moves
│
├── brightness_experiments/
│   ├── brightness_scorer.py      # scores all images by greyscale brightness, generates darkest_viewer.html
│   ├── ground_truth.py           # compares 5 brightness metrics on evening images, generates evening_viewer.html
│   ├── all_brightness.csv        # brightness scores for full dataset
│   ├── evening_brightness.csv    # brightness scores for evening images only
│   ├── darkest_viewer.html       # viewer for darkest N images with threshold slider
│   └── evening_viewer.html       # viewer for evening images with metric toggle
│
├── dino_experiments/
│   ├── dino_exps.py              # main experiment runner (grid mode + count mode)
│   ├── prompts.yaml              # all prompt configurations — add new prompts here
│   ├── proxy_viewer.html         # viewer for count runs — drop CSV + JSON to load
│   ├── dino_counts/              # outputs from count mode
│   │   ├── dino_counts_*.csv     # detection counts per image
│   │   └── dino_counts_*.json    # bounding box data per image (for viewer)
│   └── dino_grid/                # outputs from grid mode
│       ├── summary.csv           # mean counts across all experiments
│       └── {prompt}__{thresh}/   # annotated images + counts.csv per experiment
│
├── pairs_experiments/
│   ├── match_images.py           # matches night images to nearest daytime image by GPS
│   ├── pairs_viewer.html         # side-by-side day-night pair viewer
│   ├── day_matched/              # copied daytime images matched to night images
│   └── evening/                  # copied evening images
│
├── notebooks/
│   └── init-explore.ipynb        # initial data exploration
│
├── urban-mosaic/                 # image data (not tracked in git)
├── requirements.txt
└── README.md
```

## Setup

```bash
conda activate nw
pip install -r requirements.txt
```

## Running experiments

### Brightness scoring
```bash
cd brightness_experiments
python brightness_scorer.py      # scores full dataset → all_brightness.csv + darkest_viewer.html
python ground_truth.py           # scores evening images → evening_brightness.csv + evening_viewer.html
```

### DINO detection
Edit `MODE` and `COUNT_PROMPT_NAME` at the top of `dino_exps.py`, then:
```bash
cd dino_experiments
python dino_exps.py
```

- `MODE = "count"` — runs a single prompt on `SAMPLES` images, saves to `dino_counts/`
- `MODE = "grid"` — runs all prompts × all thresholds on `N_IMAGES` sample images, saves to `dino_grid/`

To add a new prompt, add an entry to `prompts.yaml` — no Python changes needed.

### Day-night pairing
```bash
cd pairs_experiments
python match_images.py
```

Matches each nighttime image to the nearest daytime image by GPS coordinates,
copies pairs locally, and generates `pairs_viewer.html`.

## Viewing results

All viewers require a local server to load images correctly:

```bash
cd /Users/mariasilva/Documents/NightWalk
python3 -m http.server 8000
```

Then open in browser:
- `http://localhost:8000/brightness_experiments/darkest_viewer.html`
- `http://localhost:8000/brightness_experiments/evening_viewer.html`
- `http://localhost:8000/dino_experiments/proxy_viewer.html` — drop `dino_counts/*.csv` and `dino_counts/*.json`
- `http://localhost:8000/pairs_experiments/pairs_viewer.html`