import os
import csv
from datetime import datetime

def ensure_dirs(csv_path, snapshot_dir):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)

def init_csv(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "visible_people_count"])

def log_count(csv_path, count):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), count])

def draw_label(frame, text, x, y, color=(0, 0, 255)):
    import cv2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)