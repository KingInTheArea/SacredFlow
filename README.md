# SacredFlow

**Real-time, perspective-aware crowd monitoring for high-density temple environments.**

SacredFlow is a hybrid computer vision pipeline that combines object detection, multi-object tracking, and density estimation to estimate occupancy and monitor crowd flow in real time — built specifically for the conditions found at large Indian temple gatherings: dense, disordered crowds and severe perspective distortion between near and far camera regions.

---

## Problem Statement

Indian temple crowds present two compounding challenges that break traditional single-model crowd monitoring approaches:

1. **Heavy, disordered crowds** — unlike queued or structured crowds (e.g., transit stations), temple crowds move unpredictably and pack densely, causing frequent occlusion.
2. **Severe perspective distortion** — a single wide-angle camera view typically captures both a *near field* (large, individually distinguishable people close to the camera) and a *far field* (a distant, densely packed mass where individuals shrink to a handful of pixels). Standard object detectors are reliable in the near field but degrade sharply in the far field, where people become too small and overlapping for per-instance detection.

A single detection model cannot handle both regimes well. SacredFlow addresses this by splitting each frame by depth and applying a different, purpose-suited model to each region.

---

## Approach

The frame is split into two regions along a configurable horizontal ratio (currently top 40% / bottom 60%):

- **Far field (top of frame):** Individual detection is unreliable here, so SacredFlow uses **DM-Count**, a density-estimation model (VGG19-based, pretrained on ShanghaiTech Part A — a dataset whose crowd density characteristics closely resemble Indian temple crowds) that predicts a density map and integrates it into a count, without needing to detect individual bounding boxes.
- **Near field (bottom of frame):** Individuals are visible and separable, so SacredFlow uses **YOLO11x** for detection, feeding results into **DeepSORT** for identity-persistent multi-object tracking. This maintains consistent IDs across frames despite partial occlusion, enabling per-person dwell-time measurement.

The two outputs — a far-field density count and a near-field tracked-identity count — are combined into a single real-time occupancy estimate.

### Why this model combination

- **YOLO11x** was chosen as a state-of-the-art real-time object detector with strong small-object and dense-scene performance.
- **DeepSORT** was added specifically to combat occlusion-driven ID loss, which is common in dense, non-queued crowds.
- **DM-Count** was selected after evaluating Bayesian Loss-based counting approaches; DM-Count's pretrained ShanghaiTech-A weights offered better out-of-the-box performance on far-field, high-density regions without requiring temple-specific fine-tuning.

---

## Architecture

```
                          ┌─────────────────────────┐
                          │      Video Frame Input    │
                          │   (RTSP / file source)    │
                          └────────────┬─────────────┘
                                       │
                          ┌────────────▼─────────────┐
                          │   Frame Split (by depth)   │
                          │   configurable ratio       │
                          └──────┬─────────────┬──────┘
                                 │             │
                  ┌──────────────▼───┐   ┌─────▼──────────────┐
                  │   Far Field        │   │   Near Field        │
                  │   (top region)     │   │   (bottom region)   │
                  └──────────┬─────────┘   └─────────┬───────────┘
                             │                        │
                  ┌──────────▼─────────┐   ┌──────────▼───────────┐
                  │     DM-Count         │   │      YOLO11x           │
                  │  density estimation  │   │  person detection      │
                  └──────────┬─────────┘   └──────────┬───────────┘
                             │                        │
                             │              ┌──────────▼───────────┐
                             │              │      DeepSORT           │
                             │              │  identity tracking      │
                             │              └──────────┬───────────┘
                             │                        │
                             │              ┌──────────▼───────────┐
                             │              │    Dwell Tracker        │
                             │              │ per-ID duration in-frame│
                             │              └──────────┬───────────┘
                             │                        │
                  ┌──────────▼────────────────────────▼───────────┐
                  │            Combined Occupancy Estimate           │
                  │     far-field count + near-field tracked IDs     │
                  └──────────┬────────────────────────────────────┘
                             │
              ┌───────────────┼────────────────────┐
              │               │                    │
   ┌──────────▼──────┐ ┌──────▼───────────┐ ┌──────▼──────────────┐
   │  Live overlay      │ │  CSV count log     │ │  Dataset auto-export │
   │  (OpenCV display)   │ │  (periodic avg)     │ │  (image + .mat pairs)│
   └────────────────────┘ └────────────────────┘ └───────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Detection | YOLO11x (Ultralytics) |
| Tracking | DeepSORT (`deep-sort-realtime`) |
| Far-field density estimation | DM-Count (VGG19 backbone, pretrained on ShanghaiTech Part A) |
| Video I/O & visualization | OpenCV |
| Tensor / inference backend | PyTorch |
| Data handling | NumPy, SciPy (`.mat` export for dataset generation) |
| Language | Python |

---

## Key Results

Metrics below are from live testing on real temple crowd footage. Where a metric is based on informal observation rather than a rigorous benchmark, that is noted explicitly — this project prioritizes honest, verifiable numbers over polished-sounding ones.

- **Combined real-time occupancy estimate of 330-350 people per frame**, sustained consistently across a multi-minute logged session on live temple footage (not a single spike — verified via periodic count logging).
- **100+ concurrently identity-tracked individuals** in the near field at a single point in time, with **110+ unique identities** tracked across the session, via YOLO11x + DeepSORT.
- **230+ additional individuals estimated in the far field** via DM-Count density estimation, in regions where per-person detection is not reliable.
- **Accuracy validation:** informally cross-checked against manually eyeballed counts on ~30–40 sample frames from real footage; It was also observed during live execution of app.py where total count was visible and could ne eyeballed against actual number but no ground-truth-labeled validation set has been used yet for a rigorous, reproducible accuracy metric. Formal validation against labeled data is a planned next step.
- **Per-identity dwell-time tracking**, computing how long each tracked individual remains in frame — intended as a proxy for estimating average time-to-darshan (time to reach the front of a temple queue), to support crowd flow planning.
- **~76.5% reduction in per-frame compute time (4.3x FPS improvement)** via adaptive frame-skipping for the far-field density estimation branch, benchmarked on real footage with instrumented timing (full breakdown below).

### Adaptive Frame Skipping — ~76.5% compute reduction, 4.3x FPS improvement

The two inference branches (YOLO11x detection, DM-Count density estimation) originally ran on every frame. Since far-field crowd density changes slowly relative to frame rate, DM-Count was moved to a skip-and-cache pattern; YOLO was left running every frame after skipping was found to break tracking. All figures below are measured on CPU (AMD integrated GPU — no CUDA available), averaged over stable 20-frame windows with the first window excluded to remove model warm-up bias.

**Baseline (no skipping, `imgsz=1920`):**

| Stage | Avg ms/frame |
|---|---|
| DM-Count inference | ~13,700 ms |
| YOLO detection | ~4,300 ms |
| Full frame | ~20,000 ms |
| Effective FPS | 0.05 |

DM-Count was the dominant bottleneck, roughly 3x slower per frame than YOLO.

**Optimizations applied:**

1. **DM-Count frame skipping** (`DM_SKIP_INTERVAL = 10`) — DM-Count inference now runs only every 10th frame; the last computed density map and count are cached and reused on skipped frames, since far-field density is stable across short windows.
2. **Reduced YOLO input resolution** (`imgsz`: 1920 → 1280) — roughly halved YOLO inference time with no observed impact on DeepSORT tracking continuity or dwell-time accuracy.
3. **YOLO frame skipping — tested and rejected.** Skipping YOLO (`YOLO_SKIP_INTERVAL = 3`) was tested but caused DeepSORT to fail to promote tentative tracks to confirmed status, since insufficient detection frequency prevented tracks from confirming before `max_age` expired — resulting in zero confirmed tracks. YOLO is kept at every frame (`YOLO_SKIP_INTERVAL = 1`) as a result.

**Benchmark comparison:**

| Configuration | DM-Count ms | YOLO ms | Total ms | FPS |
|---|---|---|---|---|
| Baseline (`imgsz=1920`, no skip) | 13,700 | 4,300 | 20,000 | 0.05 |
| `imgsz=1920`, DM skip=10 | 1,362 | 4,300 | 7,611 | 0.13 |
| `imgsz=1280`, no skip | ~13,700 | 1,950 | ~15,650 | ~0.06 |
| **`imgsz=1280`, DM skip=10 (final)** | **1,372** | **1,967** | **4,699** | **0.21** |

**Net result:** DM-Count per-frame time reduced ~10x (13,700ms → 1,372ms), YOLO reduced ~2.2x (4,300ms → 1,967ms), for an overall **~76.5% reduction in total frame time** (20,000ms → 4,699ms) and a **~4.3x improvement in effective FPS** (0.05 → 0.21).

Per-stage timing instrumentation (`time.time()` wrappers around each inference block, `BENCHMARK_MODE = True` in `config.py`) is built into `app.py` for reproducibility.

*Note: all benchmarks were run on CPU only, since CUDA is unavailable on the test machine's AMD integrated GPU. On a dedicated NVIDIA GPU, absolute inference times would be substantially lower, and higher skip intervals may be unnecessary.*

### Planned / in progress

- Formal accuracy benchmarking against a ground-truth-labeled validation set.
- Stress testing on higher-density footage (400+ person crowds) to characterize tracking behavior at extreme density.

---

## Setup Instructions

### Prerequisites
- Python 3.9+
- A CUDA-capable GPU is recommended for real-time performance; the pipeline also runs on CPU (tested on CPU-only for the DM-Count stage).

### Installation

```bash
git clone https://github.com/<your-username>/SacredFlow.git
cd SacredFlow

# Install main pipeline dependencies
pip install -r requirements.txt

# Install DM-Count dependencies (vendored framework)
pip install -r DM-Count/requirements.txt
```

### Model weights

- **YOLO11x weights** (`yolo11x.pt`) — download from the [Ultralytics releases](https://github.com/ultralytics/assets/releases) and place in `models/`.
- **DM-Count pretrained weights** (`model_sh_A.pth`, trained on ShanghaiTech Part A) — place in `DM-Count/pretrained_models/`.

Update paths in `config.py` to match your local setup:

```python
MODEL_PATH          = "models/yolo11x.pt"
DMCOUNT_MODEL_PATH  = "DM-Count/pretrained_models/model_sh_A.pth"
DMCOUNT_REPO_PATH   = "DM-Count"
SPLIT_RATIO         = 0.4     # top 40% of frame -> DM-Count far field
DMCOUNT_ENABLED     = True    # disable to run YOLO+DeepSORT only
VIDEO_SOURCE        = "path/to/your/video_or_rtsp_stream"

DM_SKIP_INTERVAL    = 10      # run DM-Count every N frames (cached in between)
YOLO_SKIP_INTERVAL  = 1       # YOLO must run every frame — see Key Results for why
BENCHMARK_MODE      = False   # set True to print per-stage timing to terminal
BENCHMARK_FRAMES    = 20      # frames per benchmark window
```

### Run

```bash
python app.py
```

Press `q` to stop. On exit, a dwell-time report is printed to console and saved to `outputs/dwell_report.json`.

---

## Folder Structure

```
SacredFlow/
├── app.py                     # Main pipeline: detection, tracking, density estimation, display
├── config.py                  # Paths, model config, split ratio, feature toggles
├── dwell_tracker.py           # Per-ID dwell-time tracking and reporting
├── utils.py                   # Shared helpers (dir setup, CSV logging, label drawing)
├── requirements.txt           # Core pipeline dependencies
├── DM-Count/                  # Vendored DM-Count framework (unmodified, see its own README)
│   ├── models.py
│   ├── train.py / test.py / demo.py
│   ├── pretrained_models/     # DM-Count checkpoint weights (gitignored)
│   └── requirements.txt
├── models/                    # YOLO11x weights (gitignored — see Setup)
├── dataset/ , dataset2/       # Auto-exported frame/annotation pairs (gitignored)
├── outputs/                   # Logs, dwell reports (gitignored)
└── LICENSE
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details. The vendored `DM-Count/` framework retains its own original license and README from the [official DM-Count repository](https://github.com/cvlab-stonybrook/DM-Count); it is included unmodified for attribution and reproducibility.
