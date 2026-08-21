# NightWalk — Map Matcher: Setup & Usage Guide

## What you'll need

- Python 3.8+
- The following Python packages:
  ```
  pip install pandas PyQt5 PyQtWebEngine
  ```
- The `map_matcher.py` script (shared separately)
- Your folder of night photos
- The daytime image CSV (`washington-square.csv` or equivalent)
- Your own pre-initialized output CSV (shared separately — **do not share this file with other labelers**)

---

## Folder structure

Your working directory should look like this:

```
nightwalk/
├── map_matcher.py
├── night-photos/          ← your folder of night images
│   ├── IMG_4971.JPG
│   └── ...
├── data/
│   └── washington-square.csv   ← the shared daytime image dataset
└── matches_yourname.csv        ← your personal output file (given to you)
```

---

## Running the tool

Open a terminal, navigate to your `nightwalk/` folder, and run:

```bash
python map_matcher.py night-photos-all/ urban-mosaic/washington-square.csv --output matches_yourname.csv
```

Replace `yourname` with your actual name (e.g. `matches_alice.csv`). **Everyone uses a different output file** — this is how we avoid overwriting each other's work.

### Modes

The tool has three modes controlled by the `--mode` flag:

| Mode | Command | When to use |
|------|---------|-------------|
| Auto (default) | _(no flag)_ | First run, or resuming — the tool figures out where you left off |
| Continue | `--mode continue` | Pick up from where you stopped; shows only unlabeled photos |
| Skipped | `--mode skipped` | Go back through photos you previously skipped |

**First time running?** No flag needed — the tool will create your output file automatically and start from the beginning.

**Resuming after a break?** Just run the same command again with no flag. It will skip already-labeled photos and continue from where you left off.

**Want to revisit skipped photos?** Run with `--mode skipped`:
```bash
python map_matcher.py night-photos-all/ urban-mosaic/washington-square.csv --output matches_remapped.csv --mode skipped
```

---

## How to use the interface

The window has three panels:

**Left — Night photo**
The photo you need to match. The filename and your progress (`Photo X of Y`) are shown above it.

**Center — Map**
A map pre-centered on the photo's GPS coordinates. A red dot marks the original GPS location. You can click anywhere on the map to search for candidates from that point instead.

**Right — Candidates**
Daytime photos taken near the pin location, sorted by distance. Each card shows a thumbnail, distance from the pin, and camera heading.

### Workflow for each photo

1. Look at the night photo on the left.
2. The map will auto-load candidates from the original GPS pin (red dot).
3. Browse the candidate cards on the right and find the best daytime match — same building, same angle.
4. Click a card to select it (it highlights blue).
5. Press **✓ Confirm match**.
6. If you can't find a match, press **Skip / Can't find** — you can revisit skipped photos later.

### Tips

- Use the **Show** spinner (top-right) to increase the number of candidates if the right match isn't visible.
- If the GPS is off, click a different spot on the map to re-search candidates from there.
- Cards with a red border and ⚠️ warning are already matched by you in a previous session — avoid re-using them.

---

## Saving your work

Your matches are saved automatically to your output CSV every time you confirm a match. You do not need to do anything special to save — just keep working and close the window when done.

---

## Sending your results back

When you're finished (or want to share progress), send your `matches_yourname.csv` file back. Do **not** rename it.