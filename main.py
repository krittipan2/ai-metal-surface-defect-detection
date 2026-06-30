from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
import time
import threading
from collections import defaultdict
import datetime
import torch
import os
import subprocess
import sys
import json

app = FastAPI()
defect_counter = defaultdict(int)
NG_DIR = "ng_images"
os.makedirs(NG_DIR, exist_ok=True)

# -------------------- LOAD MODEL --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = YOLO("best.pt")
model.to(device)
if device == "cuda":
    model.fuse()
print(f"🚀 Running on: {device}")

# -------------------- CAMERA --------------------
camera_index = 0
camera       = None

def get_camera(index=0):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Cannot open camera {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)  # ← เพิ่ม resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,    0)     # ← ปิด autofocus
    print(f"✅ Camera opened at index {index}")
    return cap

camera = get_camera(camera_index)

# -------------------- FRAME BUFFER (Optimization #1) --------------------
latest_frame      = None
frame_lock        = threading.Lock()
camera_lock       = threading.Lock()

def camera_reader():
    """อ่านกล้องใน thread แยก — ทุก stream ใช้ frame ร่วมกัน ลด CPU"""
    global latest_frame
    while True:
        with camera_lock:
            success, frame = camera.read()
        if success and frame is not None:
            with frame_lock:
                latest_frame = frame
        else:
            time.sleep(0.01)

threading.Thread(target=camera_reader, daemon=True).start()
print("✅ Camera reader thread started")

# -------------------- Global Variables --------------------
confidence_threshold = 0.90
current_status       = "OK"
current_defects      = []
current_inference_time = 0
current_fps          = 0
current_defect_count = 0
current_confidence   = 0
total_inspected      = 0
brightness           = 0
contrast             = 1.0
hourly_defect_counter = defaultdict(int)

capture_delay    = 15.0
capture_interval = 3.0
capture_max      = 3
ok_reset_delay   = 2.0

object_detected  = False
pending_save     = None
capture_count    = 0
best_frame       = None
best_conf_saved  = 0
ok_since         = None

# -------------------- ROI --------------------
ROI         = (160, 120, 1120, 840)
roi_enabled = True

# -------------------- Display Options --------------------
show_labels = True   # เปิด/ปิด label+conf บน frame

# -------------------- Config Load/Save --------------------
CONFIG_PATH = "config.json"

def load_config():
    global brightness, contrast, confidence_threshold, ROI, roi_enabled
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
            brightness           = cfg.get("brightness", 0)
            contrast             = cfg.get("contrast",   1.0)
            confidence_threshold = cfg.get("conf",       0.90)
            ROI                  = tuple(cfg.get("roi",  [80, 60, 560, 420]))
            roi_enabled          = cfg.get("roi_enabled", True)
            print("✅ Config loaded:", cfg)
        except Exception as e:
            print(f"⚠️ Config load failed: {e}")

def save_config():
    try:
        cfg = {
            "brightness":  brightness,
            "contrast":    contrast,
            "conf":        confidence_threshold,
            "roi":         list(ROI),
            "roi_enabled": roi_enabled
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
        print("💾 Config saved:", cfg)
    except Exception as e:
        print(f"⚠️ Config save failed: {e}")

load_config()

# -------------------- JPEG ENCODE PARAMS (Optimization #2) --------------------
JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 75]

np.random.seed(42)
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------- DEFECT PRESETS --------------------
DEFECT_PRESETS = {
    "default":     {"brightness": 0,  "contrast": 1.0, "conf": 0.9},
    "spot":        {"brightness": 8,  "contrast": 1.7, "conf": 0.6},
    "scratch":     {"brightness": 5,  "contrast": 1.6, "conf": 0.6},
    "orange_peel": {"brightness": -5, "contrast": 1.3, "conf": 0.60},
    "stain":       {"brightness": 10, "contrast": 1.1, "conf": 0.6}
}

# -------------------- generate_stream --------------------
def generate_stream(mode="raw", t1=100, t2=200):
    global total_inspected
    global current_status, current_defects
    global current_inference_time, current_defect_count
    global current_fps, object_detected, current_confidence
    global capture_count, best_frame, best_conf_saved
    global pending_save, ok_since, show_labels

    frame_count    = 0          # สำหรับ skip frame
    last_display   = None       # frame ล่าสุดที่ encode แล้ว (กัน None)

    while True:

        # ---------- อ่าน frame จาก buffer ----------
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            frame = latest_frame.copy()

        frame_count += 1

        # ---------------- IMAGE PROCESS ----------------
        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        display_frame = frame.copy()

        if mode == "detect":

            rx1, ry1, rx2, ry2 = ROI

            # วาด ROI
            if roi_enabled:
                cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (0, 200, 255), 2)
                cv2.putText(display_frame, "ROI", (rx1 + 4, ry1 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
                roi_crop = frame[ry1:ry2, rx1:rx2]
            else:
                roi_crop = frame

            # ---------- Skip frame (Optimization #3) ----------
            # detect ทุก 2 frame เพื่อลด CPU
            if frame_count % 2 == 0:

                t_start = time.time()
                results = model(
                    roi_crop,
                    imgsz=640,
                    conf=confidence_threshold,
                    iou=0.45,                                          # NMS (Optimization #4)
                    half=True if device == "cuda" else False,         # FP16 (Optimization #5)
                    device=device,
                    verbose=False
                )[0]
                t_infer = time.time() - t_start
                current_inference_time = round(t_infer, 4)
                current_fps            = round(1 / t_infer, 1)

                detected_classes = []
                best_conf  = 0
                best_label = ""
                boxes      = []

                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    label  = model.names[cls_id]
                    detected_classes.append(label)

                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    if roi_enabled:
                        bx1 += rx1; bx2 += rx1
                        by1 += ry1; by2 += ry1
                    boxes.append((bx1, by1, bx2, by2, label, conf))

                    if conf > best_conf:
                        best_conf  = conf
                        best_label = label
                        current_confidence = round(conf * 100, 1)

                # ---------------- DRAW ----------------
                if boxes:
                    color      = (0, 255, 255)
                    font       = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness  = 1
                    h_fr, w_fr = display_frame.shape[:2]

                    if best_label in ["spot", "scratch"]:
                        for b in boxes:
                            bx1, by1, bx2, by2, lbl, c = b
                            cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), color, 2)
                            if show_labels:
                                tag = f"{lbl} {c:.2f}"
                                (tw, th), _ = cv2.getTextSize(tag, font, font_scale, thickness)
                                ty_tag = by1 - 6 if by1 - 6 > th + 2 else by2 + th + 6
                                ty_tag = max(th + 2, min(ty_tag, h_fr - 4))
                                tx_tag = max(2, min(bx1, w_fr - tw - 4))
                                cv2.rectangle(display_frame,
                                              (tx_tag - 2, ty_tag - th - 2),
                                              (tx_tag + tw + 2, ty_tag + 2),
                                              (0, 0, 0), -1)
                                cv2.putText(display_frame, tag, (tx_tag, ty_tag),
                                            font, font_scale, color, thickness)
                    else:
                        mx1 = min(b[0] for b in boxes)
                        my1 = min(b[1] for b in boxes)
                        mx2 = max(b[2] for b in boxes)
                        my2 = max(b[3] for b in boxes)
                        cv2.rectangle(display_frame, (mx1, my1), (mx2, my2), color, 3)

                        if show_labels:
                            for b in boxes:
                                bx1, by1, bx2, by2, lbl, c = b
                                tag = f"{lbl} {c:.2f}"
                                (tw, th), _ = cv2.getTextSize(tag, font, font_scale, thickness)
                                ty_tag = by1 - 6 if by1 - 6 > th + 2 else by2 + th + 6
                                ty_tag = max(th + 2, min(ty_tag, h_fr - 4))
                                tx_tag = max(2, min(bx1, w_fr - tw - 4))
                                cv2.rectangle(display_frame,
                                              (tx_tag - 2, ty_tag - th - 2),
                                              (tx_tag + tw + 2, ty_tag + 2),
                                              (0, 0, 0), -1)
                                cv2.putText(display_frame, tag, (tx_tag, ty_tag),
                                            font, font_scale, color, thickness)

                # ---------------- STATUS ----------------
                current_defect_count = len(detected_classes)
                now = time.time()

                if current_defect_count > 0:
                    current_status  = "NG"
                    current_defects = list(set(detected_classes))
                else:
                    current_status  = "OK"
                    current_defects = []

                # ---------------- SAVE LOGIC ----------------
                current_set = set(detected_classes)

                if current_status == "NG":
                    ok_since = None
                    if not object_detected:
                        object_detected = True
                        pending_save    = now + capture_delay
                        capture_count   = 0
                        best_frame      = None
                        best_conf_saved = 0

                    if best_conf > best_conf_saved:
                        best_conf_saved = best_conf
                        best_frame      = display_frame.copy()

                    if pending_save and now >= pending_save and capture_count < capture_max:
                        capture_count += 1
                        pending_save   = now + capture_interval
                        if capture_count == 1:
                            for defect in current_set:
                                defect_counter[defect] += 1

                        now_dt      = datetime.datetime.now()
                        date_folder = now_dt.strftime("%Y-%m-%d")
                        main_defect = (current_defects[0] if current_defects else "unknown").replace(" ", "_")
                        save_dir    = os.path.join(NG_DIR, date_folder, main_defect)
                        os.makedirs(save_dir, exist_ok=True)
                        timestamp   = now_dt.strftime("%Y%m%d_%H%M%S")
                        filename    = f"{timestamp}_s{capture_count}.jpg"
                        filepath    = os.path.join(save_dir, filename)

                        frame_to_save = cv2.resize(display_frame, (1280, 960))

                        cv2.imwrite(filepath, frame_to_save, JPEG_PARAMS)
                        cv2.imwrite("ng_images/latest.jpg", frame_to_save, JPEG_PARAMS)

                        with open(os.path.join(NG_DIR, "log.txt"), "a") as f:
                            f.write(
                                f"{date_folder}/{main_defect}/{filename}"
                                f"|{','.join(current_defects)}"
                                f"|{timestamp}"
                                f"|shot{capture_count}\n"
                            )
                        print(f"📸 Shot {capture_count}/{capture_max} saved: {filename}")

                else:
                    if ok_since is None:
                        ok_since = now
                    if now - ok_since >= ok_reset_delay:
                        object_detected = False
                        pending_save    = None
                        capture_count   = 0
                        best_frame      = None
                        best_conf_saved = 0
                        ok_since        = None
                        print("✅ Reset: ready for next part")

                last_display = display_frame  # บันทึก frame ล่าสุด

            else:
                # frame ที่ skip — ใช้ display เดิม (วาด ROI ทับ)
                if last_display is not None:
                    display_frame = last_display.copy()
                    if roi_enabled:
                        cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (0, 200, 255), 2)
                        cv2.putText(display_frame, "ROI", (rx1+4, ry1+18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

        elif mode == "edge":
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, t1, t2)
            display_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        total_inspected += 1

        # ---------- Encode JPEG quality 75 (Optimization #2) ----------
        _, buffer   = cv2.imencode('.jpg', display_frame, JPEG_PARAMS)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# -------------------- VIDEO ROUTES --------------------
@app.get("/video_raw")
def video_raw():
    return StreamingResponse(generate_stream("raw"),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_detect")
def video_detect():
    return StreamingResponse(generate_stream("detect"),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_edge")
def video_edge(t1: int = Query(100), t2: int = Query(200)):
    return StreamingResponse(generate_stream("edge", t1, t2),
                             media_type="multipart/x-mixed-replace; boundary=frame")

# -------------------- API --------------------
@app.get("/")
def dashboard():
    return FileResponse("static/dashboard.html")

@app.get("/api/status")
def get_status():
    return {"status": current_status, "defects": current_defects}

@app.get("/api/live_data")
def live_data():
    return {
        "total_defects":    current_defect_count,
        "inference_time":   current_inference_time,
        "fps":              current_fps,
        "detected_defects": current_defects,
        "confidence":       current_confidence
    }

@app.get("/api/hourly_defects")
def hourly():
    return dict(hourly_defect_counter)

@app.get("/api/set_conf")
def set_conf(value: float):
    global confidence_threshold
    confidence_threshold = value
    save_config()
    return {"conf": confidence_threshold}

@app.get("/api/latest_ng")
def latest_ng():
    path = "ng_images/latest.jpg"
    return FileResponse(path if os.path.exists(path) else "static/no-image.png")

@app.get("/api/defect_summary")
def defect_summary():
    return dict(defect_counter)

@app.get("/api/ng_gallery")
def ng_gallery():
    files    = []
    log_path = "ng_images/log.txt"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            lines = f.readlines()[-10:]
        for line in reversed(lines):
            parts = line.strip().split("|")
            path, defects, timestamp = parts[0], parts[1], parts[2]
            files.append({"img": f"/ng_images/{path}", "defects": defects, "time": timestamp})
    return files

app.mount("/ng_images", StaticFiles(directory="ng_images"), name="ng_images")

@app.get("/api/set_brightness")
def set_brightness(value: int):
    global brightness
    brightness = value
    save_config()
    return {"brightness": brightness}

@app.get("/api/set_contrast")
def set_contrast(value: float):
    global contrast
    contrast = value
    save_config()
    return {"contrast": contrast}

@app.get("/api/set_camera")
def set_camera(index: int):
    global camera, camera_index
    try:
        with camera_lock:
            new_cam = get_camera(index)
            if camera is not None:
                camera.release()
            camera       = new_cam
            camera_index = index
        return {"success": True, "camera_index": camera_index}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/set_preset")
def set_preset(name: str):
    global brightness, contrast, confidence_threshold
    preset               = DEFECT_PRESETS[name]
    brightness           = preset["brightness"]
    contrast             = preset["contrast"]
    confidence_threshold = preset["conf"]
    save_config()
    return preset

@app.get("/api/get_config")
def get_config():
    return {"brightness": brightness, "contrast": contrast, "conf": confidence_threshold}

@app.get("/api/get_roi")
def get_roi():
    return {"roi": list(ROI), "enabled": roi_enabled}

@app.get("/api/set_roi")
def set_roi(x1: int, y1: int, x2: int, y2: int):
    global ROI
    ROI = (x1, y1, x2, y2)
    save_config()
    return {"roi": list(ROI)}

@app.get("/api/toggle_roi")
def toggle_roi():
    global roi_enabled
    roi_enabled = not roi_enabled
    save_config()
    return {"enabled": roi_enabled}

@app.get("/api/toggle_labels")
def toggle_labels():
    global show_labels
    show_labels = not show_labels
    return {"show_labels": show_labels}

@app.get("/api/get_show_labels")
def get_show_labels():
    return {"show_labels": show_labels}

@app.get("/api/open_ng_folder")
def open_ng_folder():
    folder = os.path.abspath(NG_DIR)
    if sys.platform == "win32":
        subprocess.Popen(f'explorer "{folder}"')
    elif sys.platform == "darwin":
        subprocess.Popen(["open", folder])
    else:
        subprocess.Popen(["xdg-open", folder])
    return {"opened": folder}