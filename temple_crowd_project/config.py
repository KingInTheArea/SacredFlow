VIDEO_SOURCE = r"C:\temple_crowd_project\temple_crowd_project\Video\new.mp4"
MODEL_PATH = "models/yolo11x.pt"
CONF_THRES = 0.15
COUNT_INTERVAL = 10
CSV_PATH = "outputs/counts.csv"
SNAPSHOT_DIR = "outputs/snapshots"
USE_YOLO_CLASSES = [0]   # person only

# -----------------------------
# DM-Count Configuration
# -----------------------------
DMCOUNT_MODEL_PATH = r"C:\temple_crowd_project\DM-Count\pretrained_models\model_sh_A.pth"
DMCOUNT_REPO_PATH  = r"C:\temple_crowd_project\DM-Count"
SPLIT_RATIO        = 0.4   # top 40% of frame goes to DM-Count
DMCOUNT_ENABLED    = True  # set False to disable without removing code