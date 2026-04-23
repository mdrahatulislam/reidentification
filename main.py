# manual_fixed_auto_tracking_reid.py
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from reid_model import ReIDModel

# -----------------------------
# Settings
# -----------------------------
VIDEO_PATH = r"D:\research\clip_3.mp4"
OUTPUT_VIDEO = "output/manual_fixed_auto_reid.mp4"
TRAJECTORY_CSV = "output/trajectory.csv"
SUMMARY_CSV = "output/person_summary.csv"
MODEL_PATH = r"best.pt"

CONF_THRESHOLD = 0.5
SIM_THRESHOLD = 0.60   # cosine similarity; 0.55~0.70 সাধারণত ঠিকঠাক
MAX_EMB_PER_ID = 5     # প্রতিটি ID-র জন্য সর্বোচ্চ কত এমবেডিং গড়ে রাখব

# -----------------------------
# Prep
# -----------------------------
os.makedirs("output", exist_ok=True)
device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

# YOLO (person only)
detector = YOLO(MODEL_PATH)
detector.classes = [0]

cap = cv2.VideoCapture(VIDEO_PATH)
width, height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# ReID model
reid = ReIDModel()

# storages
id_to_embs = {}         # "1" -> [np(2048,), ...]  (একাধিক এমবেডিং রেখে গড় নেব)
id_to_color = {}
trajectories = {}       # id -> [(frame, t, x, y)]
csv_records = []        # [frame, time, id, x, y]

def get_color(pid):
    if pid not in id_to_color:
        id_to_color[pid] = tuple(np.random.randint(0,255,3).tolist())
    return id_to_color[pid]

def mean_emb(emb_list):
    """L2-normalized mean embedding"""
    if len(emb_list) == 1:
        v = emb_list[0]
    else:
        v = np.mean(np.stack(emb_list, 0), axis=0)
    # re-normalize
    n = np.linalg.norm(v) + 1e-12
    return (v / n).astype(np.float32)

# manual assignment via click
current_boxes = []      # list of (x1,y1,x2,y2)
current_crops = []      # list of BGR crops
current_embs = None     # np array (N, 2048)

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(current_boxes) > 0:
        # কোন বক্সে ক্লিক করা হলো?
        for i, (x1,y1,x2,y2) in enumerate(current_boxes):
            if x1 < x < x2 and y1 < y < y2:
                pid = input(f"Enter ID for person at ({x},{y}): ").strip()
                if pid:
                    # ঐ ডিটেকশনের এমবেডিং
                    emb = current_embs[i].copy()
                    if pid not in id_to_embs:
                        id_to_embs[pid] = [emb]
                    else:
                        id_to_embs[pid].append(emb)
                        if len(id_to_embs[pid]) > MAX_EMB_PER_ID:
                            id_to_embs[pid] = id_to_embs[pid][-MAX_EMB_PER_ID:]
                    print(f"✅ Assigned/updated ID={pid} (total refs: {len(id_to_embs[pid])})")
                break

cv2.namedWindow("Manual ID ReID Tracking")
cv2.setMouseCallback("Manual ID ReID Tracking", mouse_cb)

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    t_sec = frame_idx / fps
    res = detector(frame, verbose=False)

    # safe parse
    boxes_xyxy = res[0].boxes.xyxy
    boxes_conf = res[0].boxes.conf if hasattr(res[0].boxes, "conf") else []

    # collect crops for ReID in batch
    current_boxes = []
    current_crops = []
    for i, b in enumerate(boxes_xyxy):
        b = b.cpu().numpy()
        conf = float(boxes_conf[i]) if len(boxes_conf) > i else 1.0
        if conf < CONF_THRESHOLD:
            continue
        x1,y1,x2,y2 = map(int, b)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(width-1, x2); y2 = min(height-1, y2)
        if x2<=x1 or y2<=y1: 
            continue
        crop = frame[y1:y2, x1:x2]
        current_boxes.append((x1,y1,x2,y2))
        current_crops.append(crop)

    # get embeddings for all detections at once
    if current_crops:
        current_embs = reid.embed_crops(current_crops)  # (N,2048) L2-normalized
    else:
        current_embs = np.zeros((0,2048), dtype=np.float32)

    # matching
    if len(id_to_embs) > 0 and len(current_embs) > 0:
        # prepare mean embeddings per id
        ids = list(id_to_embs.keys())
        id_means = np.stack([mean_emb(id_to_embs[_id]) for _id in ids], 0)  # [M, 2048]

        # cosine = dot (already normalized)
        sims = current_embs @ id_means.T  # [N, M]

        for i, (x1,y1,x2,y2) in enumerate(current_boxes):
            row = sims[i]
            best_j = int(np.argmax(row))
            best_sim = float(row[best_j])
            if best_sim >= SIM_THRESHOLD:
                pid = ids[best_j]
                color = get_color(pid)
                cx, cy = (x1+x2)//2, (y1+y2)//2
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, f"ID {pid} ({best_sim:.2f})", (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.circle(frame, (cx,cy), 4, color, -1)

                trajectories.setdefault(pid, []).append((frame_idx, t_sec, cx, cy))
                csv_records.append([frame_idx, round(t_sec,3), pid, cx, cy])
            else:
                # unknown / unmatched
                cv2.rectangle(frame, (x1,y1), (x2,y2), (120,120,120), 1)

    # draw trajectories
    for pid, pts in trajectories.items():
        col = get_color(pid)
        for k in range(1, len(pts)):
            x_prev, y_prev = int(pts[k-1][2]), int(pts[k-1][3])
            x_now,  y_now  = int(pts[k][2]),  int(pts[k][3])
            cv2.line(frame, (x_prev,y_prev), (x_now,y_now), col, 2)

    cv2.putText(frame, f"Frame: {frame_idx}", (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Manual ID ReID Tracking", frame)
    out.write(frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# save CSVs
df = pd.DataFrame(csv_records, columns=["frame","time_sec","id","x","y"])
df.to_csv(TRAJECTORY_CSV, index=False)

summary_rows = []
for pid, g in df.groupby("id"):
    f0 = int(g["frame"].min()); f1 = int(g["frame"].max())
    t0 = float(g["time_sec"].min()); t1 = float(g["time_sec"].max())
    summary_rows.append([pid, f0, f1, round(t0,3), round(t1,3),
                         round(t1-t0,3), len(g)])
summary = pd.DataFrame(summary_rows, columns=[
    "id","appear_frame","disappear_frame",
    "appear_time_sec","disappear_time_sec",
    "total_time_sec","num_frames"
])
summary.to_csv(SUMMARY_CSV, index=False)

print("\n✅ Done!")
print(f"🎥 Video: {OUTPUT_VIDEO}")
print(f"📄 Trajectory: {TRAJECTORY_CSV}")
print(f"📊 Summary: {SUMMARY_CSV}")
