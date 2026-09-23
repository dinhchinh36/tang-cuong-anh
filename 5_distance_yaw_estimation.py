import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = Path(r"D:\TANGCUONGANH\runs\pose\pallet_pose_project\pallet_model-2\weights\best.pt")
IMAGE_DIR = Path(r"D:\TANGCUONGANH\val\images")
OUTPUT_DIR = Path(__file__).resolve().parent / "distance_results"
CONFIDENCE = 0.5

# Pallet dimensions and camera focal length used for pinhole distance estimation.
LENGTH_REAL_MM = 90.0
WIDTH_REAL_MM = 90.0
HEIGHT_REAL_MM = 10.0
FOCAL_LENGTH_X_PX = 800.0

RAW_EDGE_KEYPOINTS = (0, 1)
ROBUST_EDGE_KEYPOINTS = (0, 4, 5, 8, 9, 1)
BOTTOM_KEYPOINTS = (2, 3)

# OpenCV colors are BGR.
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
POSE_GREEN = (0, 140, 0)
KEYPOINT_GREEN = (0, 200, 0)
WHITE = (255, 255, 255)
MAGENTA = (255, 0, 255)


def draw_text(image, text, position, color=WHITE, scale=0.62, thickness=2):
    cv2.putText(
        image,
        text,
        tuple(int(value) for value in position),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_diagnostic_panel(image, lines):
    panel_height = 24 + len(lines) * 28
    overlay = image.copy()
    cv2.rectangle(overlay, (12, 10), (430, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, image, 0.32, 0.0, image)
    for line_index, (text, color) in enumerate(lines):
        draw_text(image, text, (25, 32 + line_index * 28), color)


def dashed_line(image, start, end, color, thickness=2, dash_length=14):
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    distance = float(np.linalg.norm(end - start))
    if distance <= 0.0:
        return

    direction = (end - start) / distance
    for offset in np.arange(0.0, distance, dash_length * 2.0):
        segment_start = start + direction * offset
        segment_end = start + direction * min(offset + dash_length, distance)
        cv2.line(
            image,
            tuple(np.round(segment_start).astype(int)),
            tuple(np.round(segment_end).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )


def fit_robust_yaw(keypoints):
    points = keypoints[list(ROBUST_EDGE_KEYPOINTS)]
    valid = np.isfinite(points).all(axis=1)
    if int(valid.sum()) < 2:
        return None

    slope, intercept = np.polyfit(points[valid, 0], points[valid, 1], 1)
    yaw = float(np.degrees(np.arctan(slope)))
    return yaw, float(slope), float(intercept)


def get_bottom_reference_y(keypoints, box):
    bottom_points = keypoints[list(BOTTOM_KEYPOINTS)]
    valid = bottom_points[np.isfinite(bottom_points).all(axis=1)]
    keypoint_y = float(np.max(valid[:, 1])) if len(valid) else -np.inf
    return float(max(keypoint_y, box[3]))


def draw_bottom_reference_axis(image, y_value):
    height, width = image.shape[:2]
    y_pixel = int(np.clip(round(y_value), 0, height - 1))
    cv2.line(image, (0, y_pixel), (width - 1, y_pixel), CYAN, 2, cv2.LINE_AA)
    draw_text(
        image,
        "Horizontal Reference Line (0 deg)",
        (20, max(22, y_pixel - 8)),
        CYAN,
        0.55,
        1,
    )


def select_detection(result):
    if result.boxes is None or result.keypoints is None:
        return None
    if len(result.boxes) == 0 or len(result.keypoints) == 0:
        return None

    confidences = result.boxes.conf.cpu().numpy()
    detection_index = int(np.argmax(confidences))
    box = result.boxes.xyxy[detection_index].cpu().numpy().astype(np.float32)
    keypoints = result.keypoints.xy[detection_index].cpu().numpy().astype(np.float32)
    if len(keypoints) < 12:
        return None

    return box, keypoints, float(confidences[detection_index])


def calculate_raw_yaw(point_left, point_right):
    dx = float(point_right[0] - point_left[0])
    dy = float(point_right[1] - point_left[1])
    return float(np.degrees(np.arctan2(dy, dx)))


def estimate_distance_z(point_left, point_right):
    pixel_width = float(np.linalg.norm(point_right - point_left))
    if pixel_width <= 1e-6:
        return None, pixel_width
    distance_z_mm = (FOCAL_LENGTH_X_PX * LENGTH_REAL_MM) / pixel_width
    return float(distance_z_mm), pixel_width


def draw_detection(image, box, keypoints, image_index, confidence):
    x_min, y_min, x_max, y_max = np.round(box).astype(int)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), MAGENTA, 2)
    draw_text(image, f"Conf: {confidence:.2f}", (x_min, max(20, y_min - 8)), MAGENTA, 0.5, 1)

    point_left = keypoints[RAW_EDGE_KEYPOINTS[0]]
    point_right = keypoints[RAW_EDGE_KEYPOINTS[1]]
    raw_yaw = calculate_raw_yaw(point_left, point_right)
    distance_z_mm, pixel_width = estimate_distance_z(point_left, point_right)

    draw_bottom_reference_axis(image, get_bottom_reference_y(keypoints, box))
    cv2.line(
        image,
        tuple(np.round(point_left).astype(int)),
        tuple(np.round(point_right).astype(int)),
        POSE_GREEN,
        2,
        cv2.LINE_AA,
    )
    draw_text(
        image,
        "Pose Raw Line (0-1)",
        (point_left + point_right) / 2.0 + np.array([8.0, 24.0]),
        POSE_GREEN,
        0.5,
        1,
    )

    robust_result = fit_robust_yaw(keypoints)
    robust_yaw = None
    if robust_result is not None:
        robust_yaw, slope, intercept = robust_result
        height, width = image.shape[:2]
        line_start = np.array([0.0, intercept], dtype=np.float32)
        line_end = np.array([float(width - 1), slope * (width - 1) + intercept], dtype=np.float32)
        dashed_line(image, line_start, line_end, YELLOW, 2)
        draw_text(image, "Robust LR Line", line_start + np.array([12.0, 36.0]), YELLOW, 0.5, 1)

    for keypoint_index, point in enumerate(keypoints[:12]):
        point_xy = tuple(np.round(point).astype(int))
        cv2.circle(image, point_xy, 4, KEYPOINT_GREEN, -1, cv2.LINE_AA)
        draw_text(image, str(keypoint_index), point_xy + np.array([6, -6]), KEYPOINT_GREEN, 0.45, 1)

    delta_yaw = abs(raw_yaw - robust_yaw) if robust_yaw is not None else None
    spec_text = f"Pallet Spec: {LENGTH_REAL_MM:.0f}x{WIDTH_REAL_MM:.0f}x{HEIGHT_REAL_MM:.0f} mm"
    distance_text = f"Distance (Z_est): {distance_z_mm:.2f} mm" if distance_z_mm is not None else "Distance (Z_est): N/A"
    robust_text = f"{robust_yaw:.2f}" if robust_yaw is not None else "N/A"
    delta_text = f"{delta_yaw:.2f}" if delta_yaw is not None else "N/A"
    diagnostic_lines = [
        (spec_text, WHITE),
        (distance_text, WHITE),
        (f"Pose Yaw (0-1): {raw_yaw:.2f} deg", POSE_GREEN),
        (f"Robust Yaw (LR): {robust_text} deg", YELLOW),
        (f"Delta Yaw Error: {delta_text} deg", WHITE),
        (f"W_pixel: {pixel_width:.2f} px | fx: {FOCAL_LENGTH_X_PX:.0f} px", WHITE),
    ]
    draw_diagnostic_panel(image, diagnostic_lines)

    return distance_z_mm, raw_yaw, robust_yaw, delta_yaw


def main():
    parser = argparse.ArgumentParser(description="Estimate pallet distance and yaw from images.")
    parser.add_argument("--source", type=Path, default=IMAGE_DIR, help="Image file or directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for a single image; batch mode writes distance_results/.",
    )
    args = parser.parse_args()

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not args.source.exists():
        raise FileNotFoundError(f"Image source not found: {args.source}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    results = model.predict(source=str(args.source), conf=CONFIDENCE, verbose=False)

    saved_count = 0
    for image_index, result in enumerate(results):
        detection = select_detection(result)
        if detection is None:
            print(f"Image {image_index}: no valid pallet detection; skipped.")
            continue

        box, keypoints, confidence = detection
        image = result.orig_img.copy()
        distance_z_mm, raw_yaw, robust_yaw, delta_yaw = draw_detection(
            image, box, keypoints, image_index, confidence
        )
        output_path = (
            args.output
            if args.output is not None and args.source.is_file()
            else OUTPUT_DIR / f"distance_test_{image_index}.jpg"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"Could not write output image: {output_path}")

        distance_text = f"{distance_z_mm:.2f}" if distance_z_mm is not None else "N/A"
        robust_text = f"{robust_yaw:.2f}" if robust_yaw is not None else "N/A"
        delta_text = f"{delta_yaw:.2f}" if delta_yaw is not None else "N/A"
        print(
            f"Image {image_index}: Z_est={distance_text} mm, "
            f"Yaw_Raw={raw_yaw:.2f} deg, Yaw_LR={robust_text} deg, "
            f"Delta={delta_text} deg -> {output_path}"
        )
        saved_count += 1

    print(f"Saved {saved_count} distance test image(s) to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
