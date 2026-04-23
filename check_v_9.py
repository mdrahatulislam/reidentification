import os, cv2, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from ultralytics import YOLO
from reid_model import ReIDModel
from datetime import timedelta
import itertools

# =============================
# Settings (edit as needed)
# =============================
VIDEO_PATH = r"D:\research\clip_1.mp4"
MODEL_PATH = "best.pt"
JSON_PATH  = r"D:\research\id_mappings old\frame_ids.json"
OUTPUT_DIR = "output5250"

CONF_THRESHOLD = 0.5
SIM_THRESHOLD  = 0.6
SUMMARY_DURATION = 5  # seconds

# --- Social Interaction params ---
INTERACTION_DISTANCE_PX = 150   # pixel threshold for "close"
INTERACTION_MIN_SEC     = 1.5   # minimum duration to call an interaction event

os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "final_output_summary.mp4")

# =============================
# Load models & manual labels
# =============================
model = YOLO(MODEL_PATH)
model.classes = [0]  # person only
reid = ReIDModel()

with open(JSON_PATH, "r") as f:
    manual_ids = json.load(f)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(3)), int(cap.get(4))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_duration = total_frames / fps

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# =============================
# Storages
# =============================
id_embeddings  = {}
colors         = {}
trajectories   = {}   # {id: [(x,y), ...]}
screen_time    = {}   # {id: frames_present}
csv_records    = []   # [frame, time_sec, id, x, y]

def get_color(pid):
    if pid not in colors:
        np.random.seed(int(pid) if str(pid).isdigit() else (hash(pid) % 1000))
        colors[pid] = tuple(np.random.randint(0,255,3).tolist())
    return colors[pid]

# =============================
# Main loop
# =============================
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    time_sec = frame_idx / fps
    time_str = str(timedelta(seconds=int(time_sec)))

    # Detect
    results = model(frame, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy() if hasattr(results[0].boxes, "conf") else np.ones(len(boxes))
    valid_boxes, crops = [], []
    for i, box in enumerate(boxes):
        if confs[i] < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box[:4])
        if (x2-x1) > 0 and (y2-y1) > 0:
            valid_boxes.append((x1, y1, x2, y2))
            crops.append(frame[y1:y2, x1:x2])

    embeds = reid.embed_crops(crops) if crops else np.zeros((0, 2048))

    # ---- Manual ID phase ----
    if str(frame_idx) in manual_ids:
        for obj in manual_ids[str(frame_idx)]:
            pid = str(obj["id"])
            x, y = int(obj["x"]), int(obj["y"])
            color = get_color(pid)

            cv2.circle(frame, (x, y), 6, color, -1)
            cv2.putText(frame, f"ID {pid}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            matched_emb = None
            for (x1, y1, x2, y2), emb in zip(valid_boxes, embeds):
                if (x1-10) < x < (x2+10) and (y1-10) < y < (y2+10):
                    matched_emb = emb
                    break
            if matched_emb is not None:
                id_embeddings[pid] = matched_emb

            trajectories.setdefault(pid, []).append((x, y))
            screen_time[pid] = screen_time.get(pid, 0) + 1
            csv_records.append([frame_idx, round(time_sec,2), pid, x, y])

    # ---- ReID phase ----
    elif len(embeds) > 0 and len(id_embeddings) > 0:
        known_ids  = list(id_embeddings.keys())
        known_embs = np.stack([id_embeddings[i] for i in known_ids])
        sims = embeds @ known_embs.T

        matched_ids = set()
        for i, (x1, y1, x2, y2) in enumerate(valid_boxes):
            best_idx = np.argmax(sims[i])
            if sims[i][best_idx] > SIM_THRESHOLD:
                pid = known_ids[best_idx]
                if pid in matched_ids:
                    continue 
                matched_ids.add(pid)

                color = get_color(pid)
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {pid}", (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                trajectories.setdefault(pid, []).append((cx, cy))
                screen_time[pid] = screen_time.get(pid, 0) + 1
                csv_records.append([frame_idx, round(time_sec,2), pid, cx, cy])

    for pid, pts in trajectories.items():
        color = get_color(pid)
        for i in range(1, len(pts)):
            cv2.line(frame, (int(pts[i-1][0]), int(pts[i-1][1])), (int(pts[i][0]),  int(pts[i][1])), color, 2)

    cv2.rectangle(frame, (10, 10), (260, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"Frame: {frame_idx}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Time: {time_str}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    out.write(frame)
    cv2.imshow("Tracking Output", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# =============================
# Save Analytics
# =============================
csv_path = os.path.join(OUTPUT_DIR, "trajectory_data_summary.csv")
df = pd.DataFrame(csv_records, columns=["frame","time_sec","id","x","y"])
df.to_csv(csv_path, index=False)

# Trajectory chart
trajectory_png = os.path.join(OUTPUT_DIR, "trajectory_summary.png")
plt.figure(figsize=(6,6))
for pid, pts in trajectories.items():
    pts_np = np.array(pts)
    if len(pts_np) > 1:
        plt.plot(pts_np[:,0], pts_np[:,1], label=f"ID {pid}", color=np.array(get_color(pid))/255)
plt.gca().invert_yaxis()
plt.legend()
plt.title("Trajectory Summary")
plt.xlabel("X position"); plt.ylabel("Y position")
plt.tight_layout(); plt.savefig(trajectory_png, dpi=300); plt.close()

# --- Screen-time chart (MODIFIED FOR LABELS INSIDE BOXES) ---
screen_chart_png = os.path.join(OUTPUT_DIR, "screen_time_chart.png")
ids = list(screen_time.keys())
vals = [screen_time[i] for i in ids]
sec_vals  = [round(v / fps, 2) for v in vals]
perc_vals = [round((v / total_frames) * 100, 1) for v in vals]

plt.figure(figsize=(10, 6))
bars = plt.barh(ids, vals, color=[np.array(get_color(i))/255 for i in ids])

max_width = max(vals) if vals else 1
for i, (bar, s, p) in enumerate(zip(bars, sec_vals, perc_vals)):
    label_text = f"{s}s ({p}%)"
    width = bar.get_width()
    
    # Logic to place text inside the box
    # We use a small padding from the right edge of the bar
    x_pos = width - (0.02 * max_width) 
    
    # If the bar is too short for the text, place it outside instead
    ha_val = 'right'
    text_color = 'white'
    if width < (max_width * 0.15): 
        x_pos = width + (0.01 * max_width)
        ha_val = 'left'
        text_color = 'black'

    plt.text(
        x_pos, 
        bar.get_y() + bar.get_height()/2, 
        label_text, 
        va='center', 
        ha=ha_val, 
        color=text_color,
        fontweight='bold',
        fontsize=10
    )

plt.title("Screen Time Summary")
plt.xlabel("Frames Visible")
plt.tight_layout()
plt.savefig(screen_chart_png, dpi=300)
plt.close()

# =============================
# Social Interaction Mapping
# =============================
interaction_csv = os.path.join(OUTPUT_DIR, "interaction_summary.csv")
interaction_records = []

for frame_no, frame_data in df.groupby('frame'):
    coords = {str(row['id']): (row['x'], row['y']) for _, row in frame_data.iterrows()}
    ids_in_frame = list(coords.keys())
    for (id1, id2) in itertools.combinations(ids_in_frame, 2):
        (x1, y1), (x2, y2) = coords[id1], coords[id2]
        dist = float(np.hypot(x1 - x2, y1 - y2))
        if dist < INTERACTION_DISTANCE_PX:
            interaction_records.append([frame_no, id1, id2, dist])

if interaction_records:
    inter_df = pd.DataFrame(interaction_records, columns=["frame","id1","id2","distance"])
    inter_df["time_sec"] = inter_df["frame"] / fps
    
    id1_clean, id2_clean = [], []
    for a, b in zip(inter_df["id1"], inter_df["id2"]):
        if str(a) <= str(b): id1_clean.append(str(a)); id2_clean.append(str(b))
        else: id1_clean.append(str(b)); id2_clean.append(str(a))
    inter_df["id1"], inter_df["id2"] = id1_clean, id2_clean

    inter_summary = inter_df.groupby(["id1","id2"]).agg(
        frames_close=("frame","nunique"),
        avg_distance=("distance","mean"),
        duration_sec=("time_sec","nunique")
    ).reset_index()
    inter_summary["interaction_event"] = (inter_summary["duration_sec"] >= INTERACTION_MIN_SEC).astype(int)
    inter_summary.to_csv(interaction_csv, index=False)

# =============================
# Field Coverage
# =============================
coverage_csv = os.path.join(OUTPUT_DIR, "field_coverage_summary.csv")
coverage_png = os.path.join(OUTPUT_DIR, "field_coverage_map.png")
frame_area = width * height
coverage_rows = []

plt.figure(figsize=(6,6))
for pid, pts in trajectories.items():
    if len(pts) < 3: continue
    pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts_arr)
    area = float(cv2.contourArea(hull))
    coverage_rows.append([pid, round(area,2), round((area/frame_area)*100, 4)])
    hull2d = hull.reshape(-1, 2)
    plt.fill(hull2d[:,0], hull2d[:,1], alpha=0.3, color=np.array(get_color(pid))/255, label=f"ID {pid}")

plt.gca().invert_yaxis()
plt.title("Field Coverage per Person")
plt.legend(); plt.tight_layout(); plt.savefig(coverage_png, dpi=300); plt.close()
pd.DataFrame(coverage_rows, columns=["id","area_pixels","coverage_percent"]).to_csv(coverage_csv, index=False)

# =============================
# Final Summary Screen
# =============================
summary_img = np.zeros((height, width, 3), dtype=np.uint8); summary_img[:] = (10, 10, 10)
cv2.putText(summary_img, "Tracking Summary", (int(width/3), 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
y_offset = 160
for pid in sorted(screen_time.keys(), key=lambda k: int(k) if str(k).isdigit() else k):
    text = f"ID {pid} -> {round(screen_time[pid]/fps, 2)}s ({round((screen_time[pid]/total_frames)*100, 1)}%)"
    cv2.putText(summary_img, text, (int(width/4), y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, get_color(pid), 2)
    y_offset += 60

for _ in range(int(fps * SUMMARY_DURATION)):
    out.write(summary_img)

out.release()
print("\n✅ Processing complete. Analytics saved to:", OUTPUT_DIR)