import cv2
import numpy as np
import os
import csv
from datetime import datetime
from ultralytics import YOLO

# 1. Initialize YOLO Models
vehicle_model = YOLO("yolov8n.pt") 
plate_model = YOLO("license-plate-finetune-v1n.pt") 

# 2. Calibration Configuration
CALIBRATED_PIXELS = 65.6743
KNOWN_HEIGHT_INCHES = 77.4367
PIXEL_TO_INCH_RATIO = KNOWN_HEIGHT_INCHES / CALIBRATED_PIXELS

# ─── SPEED & DETECTOR SETTINGS ────────────────────────────────────────────────
ZONE_LENGTH_FEET = 40.0  
FPS = 30.0               

# ─── 🚨 ALERT VIOLATION THRESHOLDS ────────────────────────────────────────────
MAX_ALLOWED_HEIGHT_INCHES = 90.0   
MAX_ALLOWED_SPEED_MPH = 65.0       
# ──────────────────────────────────────────────────────────────────────────────

# ─── DATA LOGGER SETUP ───────────────────────────────────────────────────────
OUTPUT_FOLDER = "alerts_folder"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

csv_path = os.path.join(OUTPUT_FOLDER, "violation_alerts_log.csv")

if not os.path.exists(csv_path):
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Frame", "Vehicle_ID", "Type", "Height_In", "Speed_MPH", "License_Plate", "Violation_Reason", "Snapshot_File"])

logged_vehicles = set()
# ──────────────────────────────────────────────────────────────────────────────

# ==============================================================================
# 🛠️ MOUSE CALIBRATION TOOL
# ==============================================================================
ZONE_POLYGON = np.array([[14, 527], [9, 367], [474, 351], [666, 435]], np.int32)
# ==============================================================================

video_path = "2.mp4" 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ ERROR: Could not open video file at '{video_path}'. Check if the file exists.")

video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps > 0:
    FPS = video_fps

vehicle_timers = {}
vehicle_speeds = {} 
frame_counter = 0

clicked_points = []
def pick_coordinates(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append([x, y])
            print(f"Point recorded: [{x}, {y}]")
            
            display_frame = param.copy()
            for pt in clicked_points:
                cv2.circle(display_frame, (pt[0], pt[1]), 5, (0, 0, 255), -1)
            
            if len(clicked_points) > 1:
                pts_array = np.array(clicked_points, np.int32)
                cv2.polylines(display_frame, [pts_array], False, (0, 255, 255), 2)
            
            if len(clicked_points) == 4:
                pts_array = np.array(clicked_points, np.int32)
                cv2.polylines(display_frame, [pts_array], True, (0, 165, 255), 2)
                print("\n🎉 SUCCESS! Copy and paste this exact array into your script:")
                print(f"ZONE_POLYGON = np.array({clicked_points}, np.int32)\n")
            
            cv2.imshow("CALIBRATION: Click 4 Corners of Your Lane, then press ANY key", display_frame)

success, calibration_frame = cap.read()
if success:
    vis_frame = calibration_frame.copy()
    cv2.namedWindow("CALIBRATION: Click 4 Corners of Your Lane, then press ANY key")
    cv2.setMouseCallback("CALIBRATION: Click 4 Corners of Your Lane, then press ANY key", pick_coordinates, calibration_frame)
    cv2.imshow("CALIBRATION: Click 4 Corners of Your Lane, then press ANY key", vis_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if len(clicked_points) == 4:
        ZONE_POLYGON = np.array(clicked_points, np.int32)
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# --- Main Video Processing Loop ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame_counter += 1
    clean_frame_copy = frame.copy()  

    overlay = frame.copy()
    cv2.fillPoly(overlay, [ZONE_POLYGON], (0, 165, 255))
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.polylines(frame, [ZONE_POLYGON], True, (0, 200, 255), 2)

    # 💡 CHANGED: stream=False removes generator deadlocks when calling the plate model inside the loop
    results = vehicle_model.track(frame, classes=[2, 5, 7], persist=True, stream=False, verbose=False)

    target_height_in = 50
    target_pixel_height = 20
    target_speed_mph = 50
    vehicle_detected_in_zone = False

    for r in results:
        boxes = r.boxes
        names = r.names
        for box in boxes:
            if box.id is None:
                continue
            track_id = int(box.id)
            class_id = int(box.cls)
            vehicle_type = names[class_id]
            
            coords = box.xyxy.cpu().numpy().astype(int)[0]
            x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
            conf = float(box.conf[0])
            
            if conf > 0.6:
                bottom_center_x = int((x1 + x2) / 2)
                bottom_center_y = int(y2)
                
                inside_zone = cv2.pointPolygonTest(ZONE_POLYGON, (bottom_center_x, bottom_center_y), False)
                
                if inside_zone >= 0: 
                    vehicle_detected_in_zone = True
                    
                    if track_id not in vehicle_timers:
                        vehicle_timers[track_id] = frame_counter
                    
                    target_pixel_height = float(y2 - y1)
                    target_height_in = target_pixel_height * PIXEL_TO_INCH_RATIO
                    
                    total_frames_spent = frame_counter - vehicle_timers[track_id]
                    if total_frames_spent > 2: 
                        time_seconds = total_frames_spent / FPS
                        speed_mph = (ZONE_LENGTH_FEET / time_seconds) * 0.681818
                        vehicle_speeds[track_id] = speed_mph
                    
                    target_speed_mph = vehicle_speeds.get(track_id, 0.0)
                    
                    height_violation = target_height_in > MAX_ALLOWED_HEIGHT_INCHES
                    speed_violation = target_speed_mph > MAX_ALLOWED_SPEED_MPH
                    
                    # ─── 🚨 CONDITIONAL ALERT PROCESSING ──────────────────────
                    if (height_violation or speed_violation) and track_id not in logged_vehicles and total_frames_spent >= 6:
                        logged_vehicles.add(track_id)
                        
                        reasons = []
                        if height_violation: reasons.append("OVERHEIGHT")
                        if speed_violation: reasons.append("SPEEDING")
                        violation_reason = " & ".join(reasons)
                        
                        h, w, _ = clean_frame_copy.shape
                        crop_x1, crop_y1 = max(0, x1), max(0, y1)
                        crop_x2, crop_y2 = min(w, x2), min(h, y2)
                        vehicle_crop = clean_frame_copy[crop_y1:crop_y2, crop_x1:crop_x2]
                        
                        plate_text = "NOT_DETECTED"
                        
                        if vehicle_crop.size > 0:
                            plate_results = plate_model(vehicle_crop, verbose=False)
                            for pr in plate_results:
                                if len(pr.boxes) > 0:
                                    p_box = pr.boxes
                                    plate_text = f"PLATE_DETECTED_CONF_{float(p_box.conf[0]):.2f}"
                        
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        img_filename = f"ALERT_ID_{track_id}_{violation_reason}.jpg"
                        img_save_path = os.path.join(OUTPUT_FOLDER, img_filename)
                        cv2.imwrite(img_save_path, clean_frame_copy)
                        
                        with open(csv_path, mode='a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                timestamp_str, frame_counter, track_id, vehicle_type, 
                                round(target_height_in, 2), round(target_speed_mph, 1), 
                                plate_text, violation_reason, img_filename
                            ])
                        print(f"🚨 ALERT RECORDED: Vehicle {track_id} logged due to: {violation_reason}")
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                    
                    bar_left = x1 + int((x2 - x1) * 0.15)
                    bar_right = x1 + int((x2 - x1) * 0.30)
                    height_overlay = frame.copy()
                    cv2.rectangle(height_overlay, (bar_left, y1), (bar_right, y2), (0, 0, 255), -1)
                    cv2.addWeighted(height_overlay, 0.6, frame, 0.4, 0, frame)
                    
                    risk_status = "CRITICAL WARNING" if (target_height_in > MAX_ALLOWED_HEIGHT_INCHES or target_speed_mph > MAX_ALLOWED_SPEED_MPH) else "Risk: Low"
                    risk_color = (0, 0, 255) if risk_status == "CRITICAL WARNING" else (0, 255, 0)
                    
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f"{target_speed_mph:.1f} MPH", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(frame, risk_status, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 2)
                    
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
                    if track_id in vehicle_timers and (frame_counter - vehicle_timers[track_id]) > 150:
                        vehicle_timers.pop(track_id, None)

    # 4. Draw the Top-Left Information Telemetry Panel
    # 4. Draw the Top-Left Information Telemetry Panel
    cv2.rectangle(frame, (0, 0), (400, 230), (20, 20, 20), -1)
    cv2.putText(frame, "Target Vehicles:", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "----------------", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Format telemetry strings based on detection status
    if vehicle_detected_in_zone:
        txt_height = f"1: Height (in.) ...... {target_height_in:.2f}"
        txt_pixels = f"1: Pixel Height ...... {target_pixel_height:.1f} px"
        txt_speed = f"1: Speed Score  ...... {target_speed_mph:.1f} MPH"
    else:
        txt_height = "1: Height (in.) ...... 0.00"
        txt_pixels = "1: Pixel Height ...... 0.0 px"
        txt_speed = "1: Speed Score  ...... 0.0 MPH"

    # Render telemetry text lines onto the panel
    cv2.putText(frame, txt_height, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, txt_pixels, (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, txt_speed, (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow("Overheight Vehicle Detection UI System", frame)

    # Loop break condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup resources
cap.release()
cv2.destroyAllWindows()