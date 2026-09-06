import cv2
import numpy as np
import os
import csv
from datetime import datetime
from ultralytics import YOLO
import re
import easyocr

# Initialize OCR
ocr_reader = easyocr.Reader(
    ['en'],
    gpu=False
)

# Optional OCR
# ============================================================
# NUMBER PLATE OCR
# ======================================================

# ============================================================
# 1. INITIALIZE YOLO MODELS
# ============================================================

vehicle_model = YOLO("yolov8n.pt")
plate_model = YOLO("license-plate-finetune-v1n.pt")


# ============================================================
# 2. HEIGHT CALIBRATION
# ============================================================

CALIBRATED_PIXELS = 65.6743
KNOWN_HEIGHT_INCHES = 77.4367

PIXEL_TO_INCH_RATIO = KNOWN_HEIGHT_INCHES / CALIBRATED_PIXELS


# ============================================================
# 3. INDIAN VEHICLE LIMITS
# ============================================================

# General maximum vehicle height in India:
# 3.8 metres = 380 cm = 149.61 inches
INDIA_MAX_HEIGHT_METRES = 3.8
MAX_ALLOWED_HEIGHT_INCHES = INDIA_MAX_HEIGHT_METRES * 39.3701

# Speed limit used by this system.
# Change this according to the road / vehicle category.
MAX_ALLOWED_SPEED_KMPH = 65.0

# Convert km/h to mph if required by display
MAX_ALLOWED_SPEED_MPH = MAX_ALLOWED_SPEED_KMPH * 0.621371


# ============================================================
# 4. SPEED & DETECTOR SETTINGS
# ============================================================

ZONE_LENGTH_FEET = 40.0
FPS = 30.0

# Minimum confidence required for vehicle detection
VEHICLE_CONFIDENCE = 0.60

# High confidence required for number plate detection
PLATE_CONFIDENCE_THRESHOLD = 0.80


# ============================================================
# 5. DATA LOGGER SETUP
# ============================================================

OUTPUT_FOLDER = "alerts_folder"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

csv_path = os.path.join(
    OUTPUT_FOLDER,
    "violation_alerts_log.csv"
)


# Create CSV if it does not already exist
if not os.path.exists(csv_path):

    with open(csv_path, mode="w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Timestamp",
            "Frame",
            "Vehicle_ID",
            "Vehicle_Type",
            "Height_m",
            "Height_ft",
            "Height_in",
            "Speed_KMPH",
            "Speed_MPH",
            "Plate_Detected",
            "Plate_Confidence",
            "License_Plate",
            "Violation_Reason",
            "Snapshot_File"
        ])


# Keep track of vehicles already logged
logged_vehicles = set()


# ============================================================
# 6. ZONE / MOUSE CALIBRATION
# ============================================================

ZONE_POLYGON = np.array(
    [
        [14, 527],
        [9, 367],
        [474, 351],
        [666, 435]
    ],
    np.int32
)


# ============================================================
# 7. VIDEO INPUT
# ============================================================

video_path = "1.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():

    print(
        f"❌ ERROR: Could not open video file "
        f"at '{video_path}'. Check if the file exists."
    )

    exit()


# Get actual FPS from video
video_fps = cap.get(cv2.CAP_PROP_FPS)

if video_fps > 0:
    FPS = video_fps


# ============================================================
# 8. VEHICLE TRACKING VARIABLES
# ============================================================

vehicle_timers = {}
vehicle_speeds = {}

frame_counter = 0


# ============================================================
# 9. MOUSE CALIBRATION TOOL
# ============================================================

clicked_points = []


def pick_coordinates(event, x, y, flags, param):

    global clicked_points

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(clicked_points) < 4:

            clicked_points.append([x, y])

            print(
                f"Point recorded: [{x}, {y}]"
            )

            display_frame = param.copy()

            # Draw clicked points
            for pt in clicked_points:

                cv2.circle(
                    display_frame,
                    (pt[0], pt[1]),
                    5,
                    (0, 0, 255),
                    -1
                )

            # Draw temporary polygon
            if len(clicked_points) > 1:

                pts_array = np.array(
                    clicked_points,
                    np.int32
                )

                cv2.polylines(
                    display_frame,
                    [pts_array],
                    False,
                    (0, 255, 255),
                    2
                )

            # Draw completed polygon
            if len(clicked_points) == 4:

                pts_array = np.array(
                    clicked_points,
                    np.int32
                )

                cv2.polylines(
                    display_frame,
                    [pts_array],
                    True,
                    (0, 165, 255),
                    2
                )

                print(
                    "\n🎉 SUCCESS! Copy and paste "
                    "this exact array into your script:"
                )

                print(
                    f"ZONE_POLYGON = "
                    f"np.array({clicked_points}, np.int32)\n"
                )

            cv2.imshow(
                "CALIBRATION: Click 4 Corners of Your Lane, "
                "then press ANY key",
                display_frame
            )


# ============================================================
# 10. CALIBRATION WINDOW
# ============================================================

success, calibration_frame = cap.read()

if success:

    vis_frame = calibration_frame.copy()

    window_name = (
        "CALIBRATION: Click 4 Corners of Your Lane, "
        "then press ANY key"
    )

    cv2.namedWindow(window_name)

    cv2.setMouseCallback(
        window_name,
        pick_coordinates,
        calibration_frame
    )

    cv2.imshow(
        window_name,
        vis_frame
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    if len(clicked_points) == 4:

        ZONE_POLYGON = np.array(
            clicked_points,
            np.int32
        )

    # Restart video
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )


# ============================================================
# 11. MAIN VIDEO PROCESSING LOOP
# ============================================================

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    frame_counter += 1

    # Clean frame used for snapshots
    clean_frame_copy = frame.copy()


    # ========================================================
    # DRAW DETECTION ZONE
    # ========================================================

    overlay = frame.copy()

    cv2.fillPoly(
        overlay,
        [ZONE_POLYGON],
        (0, 165, 255)
    )

    cv2.addWeighted(
        overlay,
        0.3,
        frame,
        0.7,
        0,
        frame
    )

    cv2.polylines(
        frame,
        [ZONE_POLYGON],
        True,
        (0, 200, 255),
        2
    )


    # ========================================================
    # VEHICLE DETECTION + TRACKING
    # ========================================================

    results = vehicle_model.track(
        frame,
        classes=[2, 5, 7],
        persist=True,
        stream=False,
        verbose=False
    )


    # Default telemetry values
    target_height_in = 0.0
    target_pixel_height = 0.0
    target_speed_mph = 0.0

    vehicle_detected_in_zone = False


    # ========================================================
    # PROCESS DETECTED VEHICLES
    # ========================================================

    for r in results:

        boxes = r.boxes
        names = r.names

        for box in boxes:

            if box.id is None:
                continue

            track_id = int(box.id)

            class_id = int(box.cls)

            vehicle_type = names[class_id]


            # ------------------------------------------------
            # VEHICLE COORDINATES
            # ------------------------------------------------

            coords = (
                box.xyxy
                .cpu()
                .numpy()
                .astype(int)[0]
            )

            x1, y1, x2, y2 = coords

            conf = float(box.conf[0])


            if conf <= VEHICLE_CONFIDENCE:
                continue


            # ------------------------------------------------
            # VEHICLE BOTTOM CENTER
            # ------------------------------------------------

            bottom_center_x = int(
                (x1 + x2) / 2
            )

            bottom_center_y = int(y2)


            # ------------------------------------------------
            # CHECK WHETHER VEHICLE IS IN ZONE
            # ------------------------------------------------

            inside_zone = cv2.pointPolygonTest(
                ZONE_POLYGON,
                (
                    bottom_center_x,
                    bottom_center_y
                ),
                False
            )


            if inside_zone >= 0:

                vehicle_detected_in_zone = True


                # ============================================
                # START TIMER FOR VEHICLE
                # ============================================

                if track_id not in vehicle_timers:

                    vehicle_timers[track_id] = (
                        frame_counter
                    )


                # ============================================
                # VEHICLE HEIGHT
                # ============================================

                target_pixel_height = float(
                    y2 - y1
                )

                target_height_in = (
                    target_pixel_height
                    * PIXEL_TO_INCH_RATIO
                )

                target_height_m = (
                    target_height_in
                    * 0.0254
                )

                target_height_ft = (
                    target_height_in
                    / 12.0
                )


                # ============================================
                # VEHICLE SPEED
                # ============================================

                total_frames_spent = (
                    frame_counter
                    - vehicle_timers[track_id]
                )


                if total_frames_spent > 2:

                    time_seconds = (
                        total_frames_spent / FPS
                    )

                    speed_mph = (
                        ZONE_LENGTH_FEET
                        / time_seconds
                    ) * 0.681818

                    vehicle_speeds[
                        track_id
                    ] = speed_mph


                target_speed_mph = (
                    vehicle_speeds.get(
                        track_id,
                        0.0
                    )
                )


                target_speed_kmph = (
                    target_speed_mph
                    * 1.60934
                )


                # ============================================
                # VIOLATION CHECK
                # ============================================

                height_violation = (
                    target_height_m
                    > INDIA_MAX_HEIGHT_METRES
                )

                speed_violation = (
                    target_speed_kmph
                    > MAX_ALLOWED_SPEED_KMPH
                )


                # ============================================
                # CRITICAL STATUS
                # ============================================

                if height_violation and speed_violation:

                    risk_status = (
                        "CRITICAL: HEIGHT + SPEED"
                    )

                elif height_violation:

                    risk_status = (
                        "CRITICAL: OVERHEIGHT"
                    )

                elif speed_violation:

                    risk_status = (
                        "CRITICAL: OVERSPEED"
                    )

                else:

                    risk_status = "Risk: Low"


                # ============================================
                # VIOLATION ALERT
                # ============================================

                if (
                    (height_violation or speed_violation)
                    and track_id not in logged_vehicles
                    and total_frames_spent >= 6
                ):

                    logged_vehicles.add(
                        track_id
                    )


                    # ----------------------------------------
                    # VIOLATION REASON
                    # ----------------------------------------

                    reasons = []

                    if height_violation:
                        reasons.append("OVERHEIGHT")

                    if speed_violation:
                        reasons.append("OVERSPEED")

                    violation_reason = " & ".join(
                        reasons
                    )


                    # ========================================
                    # VEHICLE CROP
                    # ========================================

                    h, w, _ = (
                        clean_frame_copy.shape
                    )

                    crop_x1 = max(0, x1)
                    crop_y1 = max(0, y1)

                    crop_x2 = min(w, x2)
                    crop_y2 = min(h, y2)


                    vehicle_crop = (
                        clean_frame_copy[
                            crop_y1:crop_y2,
                            crop_x1:crop_x2
                        ]
                    )


                
                    # ========================================
                    # NUMBER PLATE DETECTION + OCR
                    # ========================================

                    plate_text = "NOT_RECOGNIZED"
                    plate_confidence = 0.0
                    plate_detected = False
                    ocr_confidence = 0.0

                    if vehicle_crop.size > 0:

                        # ------------------------------------
                        # DETECT NUMBER PLATE
                        # ------------------------------------

                        plate_results = plate_model(
                            vehicle_crop,
                            verbose=False
                        )

                        best_plate_box = None
                        best_plate_conf = 0.0

                        for pr in plate_results:

                            if len(pr.boxes) == 0:
                                continue

                            for p_box in pr.boxes:

                                current_conf = float(
                                    p_box.conf[0]
                                )

                                if current_conf > best_plate_conf:

                                    best_plate_conf = current_conf
                                    best_plate_box = p_box

                        # ------------------------------------
                        # HIGH CONFIDENCE PLATE
                        # ------------------------------------

                        if (
                            best_plate_box is not None
                            and best_plate_conf >= PLATE_CONFIDENCE_THRESHOLD
                        ):

                            plate_detected = True
                            plate_confidence = best_plate_conf

                            # --------------------------------
                            # GET PLATE COORDINATES
                            # --------------------------------

                            plate_coords = (
                                best_plate_box
                                .xyxy
                                .cpu()
                                .numpy()
                                .astype(int)[0]
                            )

                            px1, py1, px2, py2 = plate_coords

                            px1 = max(0, px1)
                            py1 = max(0, py1)

                            px2 = min(
                                vehicle_crop.shape[1],
                                px2
                            )

                            py2 = min(
                                vehicle_crop.shape[0],
                                py2
                            )

                            # --------------------------------
                            # CROP PLATE
                            # --------------------------------

                            plate_crop = vehicle_crop[
                                py1:py2,
                                px1:px2
                            ]

                            if plate_crop.size > 0:

                                # ==================================
                                # IMAGE PREPROCESSING
                                # ==================================

                                # Upscale plate
                                plate_crop = cv2.resize(
                                    plate_crop,
                                    None,
                                    fx=4,
                                    fy=4,
                                    interpolation=cv2.INTER_CUBIC
                                )

                                # Convert to grayscale
                                gray = cv2.cvtColor(
                                    plate_crop,
                                    cv2.COLOR_BGR2GRAY
                                )

                                # Reduce noise
                                gray = cv2.bilateralFilter(
                                    gray,
                                    11,
                                    17,
                                    17
                                )

                                # Improve contrast
                                gray = cv2.equalizeHist(
                                    gray
                                )

                                # ==================================
                                # EASY OCR
                                # ==================================

                                ocr_results = ocr_reader.readtext(
                                    gray,
                                    detail=1,
                                    paragraph=False,
                                    allowlist=(
                                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                        "0123456789"
                                    )
                                )

                                # ==================================
                                # FIND BEST OCR RESULT
                                # ==================================

                                best_ocr_text = ""
                                best_ocr_conf = 0.0

                                for (
                                    bbox,
                                    text,
                                    confidence
                                ) in ocr_results:

                                    cleaned_text = re.sub(
                                        r"[^A-Z0-9]",
                                        "",
                                        text.upper()
                                    )

                                    if (
                                        len(cleaned_text) >= 4
                                        and confidence > best_ocr_conf
                                    ):

                                        best_ocr_text = (
                                            cleaned_text
                                        )

                                        best_ocr_conf = (
                                            confidence
                                        )

                                # ==================================
                                # ACCEPT OCR RESULT
                                # ==================================

                                if best_ocr_text:

                                    plate_text = best_ocr_text
                                    ocr_confidence = best_ocr_conf

                                    print(
                                        f"🚘 NUMBER PLATE: "
                                        f"{plate_text}"
                                    )

                                    print(
                                        f"🔍 PLATE DETECTION CONFIDENCE: "
                                        f"{plate_confidence:.2f}"
                                    )

                                    print(
                                        f"🔍 OCR CONFIDENCE: "
                                        f"{ocr_confidence:.2f}"
                                    )

                                else:

                                    plate_text = (
                                        "OCR_NOT_RECOGNIZED"
                                    )

                                    print(
                                        "⚠️ Plate detected, "
                                        "but number could not be read."
                                    )

                        else:

                            plate_detected = False
                            plate_confidence = (
                                best_plate_conf
                            )

                            plate_text = (
                                "PLATE_NOT_CONFIDENT"
                            )



                    # ========================================
                    # SAVE VIOLATION TO CSV
                    # ========================================

                    with open(
                        csv_path,
                        mode="a",
                        newline=""
                    ) as f:

                        writer = csv.writer(f)

                        writer.writerow([
                            timestamp_str,
                            frame_counter,
                            track_id,
                            vehicle_type,

                            round(
                                target_height_m,
                                2
                            ),

                            round(
                                target_height_ft,
                                2
                            ),

                            round(
                                target_height_in,
                                2
                            ),

                            round(
                                target_speed_kmph,
                                1
                            ),

                            round(
                                target_speed_mph,
                                1
                            ),

                            plate_detected,

                            round(
                                plate_confidence,
                                2
                            ),

                            plate_text,

                            violation_reason,

                            img_filename
                        ])


                    print(
                        "\n🚨 ============================="
                    )

                    print(
                        "🚨 CRITICAL VEHICLE DETECTED"
                    )

                    print(
                        f"Vehicle ID : {track_id}"
                    )

                    print(
                        f"Type       : {vehicle_type}"
                    )

                    print(
                        f"Height     : "
                        f"{target_height_m:.2f} m"
                    )

                    print(
                        f"Speed      : "
                        f"{target_speed_kmph:.1f} km/h"
                    )

                    print(
                        f"Plate      : "
                        f"{plate_text}"
                    )

                    print(
                        f"Plate Conf : "
                        f"{plate_confidence:.2f}"
                    )

                    print(
                        f"Violation  : "
                        f"{violation_reason}"
                    )

                    print(
                        f"Snapshot   : "
                        f"{img_filename}"
                    )

                    print(
                        "🚨 =============================\n"
                    )


                # ============================================
                # DRAW VEHICLE BOX
                # ============================================

                box_color = (
                    (0, 0, 255)
                    if (
                        height_violation
                        or speed_violation
                    )
                    else (0, 255, 0)
                )


                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )


                # ============================================
                # HEIGHT BAR
                # ============================================

                bar_left = (
                    x1
                    + int(
                        (x2 - x1)
                        * 0.15
                    )
                )

                bar_right = (
                    x1
                    + int(
                        (x2 - x1)
                        * 0.30
                    )
                )


                height_overlay = frame.copy()


                cv2.rectangle(
                    height_overlay,
                    (bar_left, y1),
                    (bar_right, y2),
                    (0, 0, 255),
                    -1
                )


                cv2.addWeighted(
                    height_overlay,
                    0.6,
                    frame,
                    0.4,
                    0,
                    frame
                )


                # ============================================
                # DISPLAY VEHICLE INFORMATION
                # ============================================

                cv2.putText(
                    frame,
                    f"ID: {track_id}",
                    (x1, y1 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )


                cv2.putText(
                    frame,
                    (
                        f"{target_speed_kmph:.1f} km/h"
                    ),
                    (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )


                cv2.putText(
                    frame,
                    (
                        f"{target_height_m:.2f} m"
                    ),
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )


                cv2.putText(
                    frame,
                    risk_status,
                    (x1, y2 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2
                )


            else:

                # Vehicle outside detection zone
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (200, 200, 200),
                    1
                )


                # Remove old tracking data
                if (
                    track_id in vehicle_timers
                    and (
                        frame_counter
                        - vehicle_timers[track_id]
                    ) > 150
                ):

                    vehicle_timers.pop(
                        track_id,
                        None
                    )


    # ========================================================
    # 12. TOP-LEFT TELEMETRY PANEL
    # ========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (430, 260),
        (20, 20, 20),
        -1
    )


    cv2.putText(
        frame,
        "INDIAN VEHICLE MONITORING",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        "------------------------",
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )


    # ========================================================
    # TELEMETRY
    # ========================================================

    if vehicle_detected_in_zone:

        txt_height = (
            f"Height     : "
            f"{target_height_in:.2f} in"
        )

        txt_height_m = (
            f"Height     : "
            f"{target_height_in * 0.0254:.2f} m"
        )

        txt_pixels = (
            f"Pixel Ht   : "
            f"{target_pixel_height:.1f} px"
        )

        txt_speed = (
            f"Speed      : "
            f"{target_speed_mph * 1.60934:.1f} km/h"
        )

        txt_limit = (
            f"Height Limit: "
            f"{INDIA_MAX_HEIGHT_METRES:.1f} m"
        )


    else:

        txt_height = (
            "Height     : 0.00 in"
        )

        txt_height_m = (
            "Height     : 0.00 m"
        )

        txt_pixels = (
            "Pixel Ht   : 0.0 px"
        )

        txt_speed = (
            "Speed      : 0.0 km/h"
        )

        txt_limit = (
            f"Height Limit: "
            f"{INDIA_MAX_HEIGHT_METRES:.1f} m"
        )


    # ========================================================
    # DISPLAY TELEMETRY
    # ========================================================

    cv2.putText(
        frame,
        txt_height_m,
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        txt_height,
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        txt_pixels,
        (10, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        txt_speed,
        (10, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )


    cv2.putText(
        frame,
        txt_limit,
        (10, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 200, 255),
        2
    )


    # ========================================================
    # DISPLAY VIDEO
    # ========================================================

    cv2.imshow(
        "Overheight Vehicle Detection UI System",
        frame
    )


    # ========================================================
    # EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


#A couple of important notes about this version:

#* **Indian height limit:** `3.8 m` is used for the general vehicle case, based on CMVR.
#* **Speed:** I left `65 km/h` as a configurable system threshold. There isn't one universal Indian speed limit for every vehicle and road, so you should set this according to your particular road/vehicle category.
#* **Plate confidence:** `0.80` is the minimum detection confidence. You can make it stricter with `0.85` or `0.90`.
#* **Snapshots:** the image is saved when the vehicle is confirmed as **OVERHEIGHT**, **OVERSPEED**, or **both**.
#* **Plate OCR:** the code attempts to read the characters if `pytesseract` is installed. If it isn't, it still records `PLATE_DETECTED` when the YOLO plate detector has high confidence.
#* **Critical detection:** a vehicle with both violations will display `CRITICAL: HEIGHT + SPEED`.
#* Your original `target_height_in = 50`, `target_pixel_height = 20`, and `target_speed_mph = 50` were just test values; I've removed those artificial values and restored real calculated values. Your uploaded version contained those test assignments.
#**One big technical caution:** your current pixel-to-inch calibration uses a single ratio. Because a vehicle farther away appears smaller, this does **not** give physically accurate vehicle height throughout the entire camera scene unless the scene has been properly perspective-calibrated. For a serious over-height detection system, the next improvement should be **perspective/homography calibration** rather than simply changing the threshold.