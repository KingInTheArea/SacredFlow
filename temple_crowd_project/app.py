import cv2
import time
from scipy.io import savemat
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from config import *
from utils import ensure_dirs, init_csv, log_count, draw_label
from dwell_tracker import DwellTracker
# -----------------------------
# DM-Count Imports
# -----------------------------
import sys
sys.path.insert(0, DMCOUNT_REPO_PATH)

import torch
from models import vgg19
from torchvision import transforms
from PIL import Image

# -----------------------------
# Setup
# -----------------------------
ensure_dirs(CSV_PATH, SNAPSHOT_DIR)
init_csv(CSV_PATH)

# Dataset folders for Bayesian Loss
import os

os.makedirs("dataset2/images", exist_ok=True)
os.makedirs("dataset2/ground_truth", exist_ok=True)

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

print("Loading DeepSORT...")
tracker = DeepSort(max_age=30)
# -----------------------------
# DM-Count Setup
# -----------------------------
dm_device = torch.device('cpu')  # AMD integrated GPU — CPU only

if DMCOUNT_ENABLED:
    print("Loading DM-Count model...")
    dm_model = vgg19()
    dm_model.to(dm_device)
    dm_checkpoint = torch.load(DMCOUNT_MODEL_PATH, map_location=dm_device)
    dm_model.load_state_dict(dm_checkpoint)
    dm_model.eval()
    print("DM-Count loaded successfully.")

print("Opening video...")
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    raise Exception(f"Cannot open video source: {VIDEO_SOURCE}")

# -----------------------------
# Video Information
# -----------------------------
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video Resolution: {video_width} x {video_height}")
print(f"FPS: {fps}")

# -----------------------------
# Dwell Tracker Setup
# -----------------------------
dwell = DwellTracker(fps=fps)
print(f"DwellTracker initialised at {fps:.2f} FPS")

# -----------------------------
# Resizable Window 
# -----------------------------
cv2.namedWindow("Temple Crowd Monitoring", cv2.WINDOW_NORMAL)

# Maximum display size
MAX_W = 1280
MAX_H = 720

last_save_time = time.time()
count_buffer = []

unique_ids_seen = set()

frame_id = 0
SAVE_EVERY = 5
# -----------------------------
# Benchmark Timing Setup
# -----------------------------
t_dm_total    = 0.0
t_yolo_total  = 0.0
t_frame_total = 0.0
bench_count   = 0
# -----------------------------
# Frame Skip Cache
# -----------------------------
last_dm_count   = 0
last_dm_outputs = None
last_vis_resized = None

# -----------------------------
# Main Loop
# -----------------------------
try:
    while True:
        t0_frame = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Video finished.")
            break
        # -------------------------
        # Frame Split
        # -------------------------
        split_y = int(video_height * SPLIT_RATIO)

        far_field  = frame[0:split_y, :]          # top strip → DM-Count
        near_field = frame[split_y:, :]           # bottom strip → YOLO (unchanged)

        # Draw split line on frame for debugging
        cv2.line(frame, (0, split_y), (video_width, split_y), (255, 0, 255), 2)
        # -------------------------
        # DM-Count Inference (far field)
        # -------------------------
        # -------------------------
        # DM-Count Inference (far field) with frame skipping
        # -------------------------
        t0_dm = time.time()
        if DMCOUNT_ENABLED and (frame_id % DM_SKIP_INTERVAL == 0):
            # Run inference this frame
            pil_img = Image.fromarray(cv2.cvtColor(far_field, cv2.COLOR_BGR2RGB))
            inp_tensor = transforms.ToTensor()(pil_img).unsqueeze(0).to(dm_device)
            with torch.no_grad():
                dm_outputs, _ = dm_model(inp_tensor)
            last_dm_count = int(torch.sum(dm_outputs).item())

            # Build and cache the density map overlay
            vis = dm_outputs[0, 0].cpu().numpy()
            vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-5)
            vis = (vis * 255).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            last_vis_resized = cv2.resize(vis, (far_field.shape[1], far_field.shape[0]))

        # Always use last known values (reuse on skipped frames)
        dm_count = last_dm_count
        t_dm_total += time.time() - t0_dm

        # -------------------------
        # DM-Count Density Map Overlay (reuse cached on skipped frames)
        # -------------------------
        if DMCOUNT_ENABLED and last_vis_resized is not None:
            frame[0:split_y, :] = cv2.addWeighted(
                frame[0:split_y, :], 0.5,
                last_vis_resized, 0.5,
                0
            )
        # -------------------------
        # YOLO Detection
        # -------------------------
        # -------------------------
        # YOLO Detection with frame skipping
        # -------------------------
        t0_yolo = time.time()
        detections = []
        points     = []

        if frame_id % YOLO_SKIP_INTERVAL == 0:
            # Run full YOLO detection this frame
            results = model.predict(
                near_field,
                conf=0.10,
                classes=[0],
                imgsz=1280,
                verbose=False
            )

            for r in results:
                for box in r.boxes:

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    y1 += split_y
                    y2 += split_y

                    conf = float(box.conf[0])
                    cx = (x1 + x2) / 2
                    cy = y1 + 0.18 * (y2 - y1)

                    points.append([cx, cy])

                    detections.append(
                        (
                            [x1, y1, x2 - x1, y2 - y1],
                            conf,
                            "person"
                        )
                    )

        t_yolo_total += time.time() - t0_yolo


        # -------------------------
        # DeepSORT Tracking
        # -------------------------
        tracks = tracker.update_tracks(
        detections,
        frame=frame
        )

        # YOLO detections count
        detected_count = len(detections)

        # Only confirmed DeepSORT tracks
        active_tracks = [
        track for track in tracks
        if track.is_confirmed()
        ]

        # DeepSORT tracked count
        tracked_count = len(active_tracks)
        # -------------------------
        # Dwell Tracker Update
        # -------------------------
        active_ids = {track.track_id for track in active_tracks}
        dwell.update(active_ids, frame_id)
        current_ids = set()

        for track in active_tracks:

            

            tid = track.track_id
            current_ids.add(tid)
            # Add to all-time seen IDs
            unique_ids_seen.add(tid)

            x1, y1, x2, y2 = map(
                int,
                track.to_ltrb()
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )
            live_secs = dwell.get_live_duration(tid, frame_id)

            draw_label(
                frame,
                f"ID {tid} | {live_secs}s",
                x1,
                max(20, y1 - 10),
                (0, 255, 0)
            )

        # -------------------------
        # Display Counts
        # -------------------------
        # Combined final count
        final_count = tracked_count + dm_count

        draw_label(frame, f"Far Field (DM) : {dm_count}", 20, 160, (255, 128, 0))
        draw_label(frame, f"TOTAL COUNT : {final_count}", 20, 200, (255, 255, 255))
        count_buffer.append(final_count)


        draw_label(
            frame,
            f"Detected : {detected_count}",
            20,
                40,
        (0,255,255)
        )

        draw_label(
            frame,
            f"Tracked : {tracked_count}",
            20,
            80,
            (0,0,255)
        )

        draw_label(
            frame,
            f"Unique Seen : {len(unique_ids_seen)}",
            20,
            120,
            (255,255,0)
        )
        annPoints = np.array(
        points,
        dtype=np.float32
        )
        if frame_id % SAVE_EVERY == 0:


            image_name = f"img_{frame_id:05d}.jpg"
            mat_name = f"img_{frame_id:05d}.mat"


            cv2.imwrite(

                f"C:/temple_crowd_project/dataset/images/{image_name}",

                frame

            )


            savemat(

                f"C:/temple_crowd_project/dataset/ground_truth/{mat_name}",


                {

                    'annPoints':annPoints

                }

            )


            print(

                f"Saved {image_name}"

            )
    
        # DEBUG
        # print(points[:5])
        # -------------------------
        # Auto Resize Display
        # -------------------------
        h, w = frame.shape[:2]

        scale = min(
            MAX_W / w,
            MAX_H / h
        )

        new_w = int(w * scale)
        new_h = int(h * scale)

        display_frame = cv2.resize(
            frame,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

        cv2.imshow(
            "Temple Crowd Monitoring",
            display_frame
        )

        # -------------------------
        # Save Average Count
        # -------------------------
        if time.time() - last_save_time >= COUNT_INTERVAL:

            avg_count = (
                int(sum(count_buffer) / len(count_buffer))
                if count_buffer
                else 0
            )

            log_count(
                CSV_PATH,
                avg_count
            )

            print(
                f"Logged Count: {avg_count}"
            )

            count_buffer = []

            last_save_time = time.time()

        # -------------------------
        # Exit
        # -------------------------
        t_frame_total += time.time() - t0_frame
        bench_count   += 1

        # -------------------------
        # Benchmark Print
        # -------------------------
        if BENCHMARK_MODE and bench_count == BENCHMARK_FRAMES:
            avg_dm    = (t_dm_total   / bench_count) * 1000
            avg_yolo  = (t_yolo_total / bench_count) * 1000
            avg_frame = (t_frame_total / bench_count) * 1000
            avg_fps   = 1000 / avg_frame if avg_frame > 0 else 0

            print(f"\n===== BENCHMARK ({bench_count} frames) =====")
            print(f"  DM-Count  avg : {avg_dm:.1f} ms/frame")
            print(f"  YOLO      avg : {avg_yolo:.1f} ms/frame")
            print(f"  Full frame avg: {avg_frame:.1f} ms/frame")
            print(f"  Effective FPS : {avg_fps:.2f}")
            print(f"==========================================\n")

            # Reset for next window
            t_dm_total = t_yolo_total = t_frame_total = 0.0
            bench_count = 0

        frame_id += 1

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
# Dwell Tracker Finalise
# -----------------------------
# -----------------------------
# Cleanup + Report (always runs)
# -----------------------------
except KeyboardInterrupt:
    print("\nInterrupted by user — saving report...")

finally:
    # Dwell Tracker Finalise
    dwell.finalise(frame_id)
    print(f"DwellTracker finalised. Total IDs tracked: {len(dwell.completed)}")

    # Dwell Time Report
    report = dwell.report()

    print("\n========== DWELL TIME REPORT ==========")
    print(f"Total IDs tracked     : {report['total_ids']}")
    print(f"Mean dwell time       : {report['mean_dwell_sec']} seconds")
    print("----------------------------------------")
    print("Per-ID breakdown:")
    for tid, duration in sorted(report['per_id'].items()):
        print(f"  ID {tid:>4} : {duration:.2f} seconds")
    print("========================================\n")

    # Save JSON
    import json
    report_path = r"C:\temple_crowd_project\outputs\dwell_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Dwell report saved to: {report_path}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Finished.")