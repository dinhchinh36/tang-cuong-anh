from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = Path(r"D:\TANGCUONGANH\runs\pose\pallet_pose_project\pallet_model-2\weights\best.pt")
IMAGE_DIR = Path(r"D:\TANGCUONGANH\val\images")
OUTPUT_DIR = Path(__file__).resolve().parent
CONFIDENCE = 0.5
PALLET_WIDTH_MM = 90.0
TOP_EDGE_KEYPOINTS = [0, 1, 4, 5, 8, 9]
LEFT_CENTER_KEYPOINTS = [4, 5, 6, 7]
RIGHT_CENTER_KEYPOINTS = [8, 9, 10, 11]

# OpenCV colors are BGR.
YELLOW = (0, 255, 255)
BASELINE = (255, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 200, 0)
POSE_GREEN = (0, 140, 0)
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


def fit_top_edge_yaw(keypoints):
    points = keypoints[TOP_EDGE_KEYPOINTS]
    valid = np.isfinite(points).all(axis=1)
    if int(valid.sum()) < 2:
        return None

    x_values = points[valid, 0]
    y_values = points[valid, 1]
    slope, intercept = np.polyfit(x_values, y_values, 1)
    yaw = float(np.degrees(np.arctan(slope)))
    return yaw, float(slope), float(intercept)


def get_detection(result):
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

    x_min, y_min, x_max, y_max = box
    features = np.array([x_min, y_min, x_max - x_min, y_max - y_min], dtype=np.float32)
    left_center = keypoints[LEFT_CENTER_KEYPOINTS].mean(axis=0)
    right_center = keypoints[RIGHT_CENTER_KEYPOINTS].mean(axis=0)
    if not np.isfinite(features).all() or not np.isfinite([*left_center, *right_center]).all():
        return None

    return {
        "box": box,
        "features": features,
        "keypoints": keypoints,
        "left_center": left_center,
        "right_center": right_center,
        "confidence": float(confidences[detection_index]),
    }


def fit_center_regression(samples):
    features = np.asarray([sample["features"] for sample in samples], dtype=np.float32)
    targets = np.asarray(
        [
            np.concatenate((sample["left_center"], sample["right_center"]))
            for sample in samples
        ],
        dtype=np.float32,
    )
    design_matrix = np.column_stack((np.ones(len(features), dtype=np.float32), features))
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, targets, rcond=None)
    return coefficients


def predict_centers(model, features):
    design_row = np.concatenate(([1.0], features)).reshape(1, -1)
    return (design_row @ model)[0]


def leave_one_out_prediction(samples, sample_index, model):
    if len(samples) <= 2:
        return predict_centers(model, samples[sample_index]["features"])

    training_samples = [sample for index, sample in enumerate(samples) if index != sample_index]
    loo_model = fit_center_regression(training_samples)
    return predict_centers(loo_model, samples[sample_index]["features"])


def draw_result(image, detection, predicted_centers, sample_index, center_model, sample_count):
    keypoints = detection["keypoints"]
    box = detection["box"]
    x_min, y_min, x_max, y_max = np.round(box).astype(int)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), MAGENTA, 2)

    yaw_result = fit_top_edge_yaw(keypoints)
    point_left, point_right = keypoints[0], keypoints[1]
    raw_yaw = float(
        np.degrees(np.arctan2(point_right[1] - point_left[1], point_right[0] - point_left[0]))
    )
    # The horizontal baseline is the 0-degree reference for both angle calculations.
    baseline_start = np.array([x_min, point_left[1]], dtype=np.float32)
    baseline_end = np.array([x_max, point_left[1]], dtype=np.float32)
    cv2.line(
        image,
        tuple(np.round(baseline_start).astype(int)),
        tuple(np.round(baseline_end).astype(int)),
        BASELINE,
        2,
        cv2.LINE_AA,
    )
    draw_text(
        image,
        "Baseline (0 deg)",
        baseline_start + np.array([8.0, -8.0]),
        BASELINE,
        0.5,
        1,
    )
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
        (point_left + point_right) / 2.0 + np.array([8.0, -10.0]),
        POSE_GREEN,
        0.5,
        1,
    )
    robust_yaw = None
    if yaw_result is not None:
        robust_yaw, slope, intercept = yaw_result
        height, width = image.shape[:2]
        line_start = np.array([0.0, intercept], dtype=np.float32)
        line_end = np.array([float(width - 1), slope * (width - 1) + intercept], dtype=np.float32)
        dashed_line(image, line_start, line_end, YELLOW, 3)

    pose_left = detection["left_center"]
    pose_right = detection["right_center"]
    predicted_left = predicted_centers[:2]
    predicted_right = predicted_centers[2:]

    for index, point in enumerate(keypoints[:12]):
        point_xy = tuple(np.round(point).astype(int))
        cv2.circle(image, point_xy, 4, GREEN, -1, cv2.LINE_AA)
        draw_text(image, str(index), point_xy + np.array([6, -6]), GREEN, 0.45, 1)

    for point, color, label in [
        (pose_left, RED, "L Pose"),
        (pose_right, BLUE, "R Pose"),
        (predicted_left, YELLOW, "L LR"),
        (predicted_right, YELLOW, "R LR"),
    ]:
        point_xy = tuple(np.round(point).astype(int))
        cv2.circle(image, point_xy, 8, color, 2, cv2.LINE_AA)
        draw_text(image, label, point_xy + np.array([10, 4]), color, 0.5, 1)

    pixel_width = float(np.linalg.norm(point_right - point_left))
    mm_per_pixel = PALLET_WIDTH_MM / pixel_width if pixel_width > 0.0 else 0.0
    error_l_pixels = float(np.linalg.norm(predicted_left - pose_left))
    error_l_mm = error_l_pixels * mm_per_pixel

    delta_yaw = abs(raw_yaw - robust_yaw) if robust_yaw is not None else None
    draw_text(image, f"Pose Yaw (0-1): {raw_yaw:.2f} deg", (25, 32), POSE_GREEN)
    robust_text = f"{robust_yaw:.2f}" if robust_yaw is not None else "N/A"
    draw_text(image, f"Robust Yaw (Linear Regression): {robust_text} deg", (25, 60), YELLOW)
    delta_text = f"{delta_yaw:.2f}" if delta_yaw is not None else "N/A"
    draw_text(image, f"Delta Yaw: {delta_text} deg", (25, 88), WHITE)
    draw_text(image, f"Error L_center (BB->LR vs Pose): {error_l_mm:.2f} mm", (25, 116), RED)
    draw_text(image, f"BB->LR samples: {sample_count}", (25, 144), WHITE, 0.5, 1)

    return raw_yaw, robust_yaw, delta_yaw, error_l_mm


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Validation image directory not found: {IMAGE_DIR}")

    model = YOLO(str(MODEL_PATH))
    results = model.predict(source=str(IMAGE_DIR), conf=CONFIDENCE, verbose=False)
    detections = [get_detection(result) for result in results]
    samples = [detection for detection in detections if detection is not None]
    if not samples:
        raise RuntimeError("No valid pallet detection with 12 keypoints was found.")

    center_model = fit_center_regression(samples)
    print(f"Fitted multivariate BB -> centers regression using {len(samples)} sample(s).")

    saved_count = 0
    sample_index = 0
    for image_index, (result, detection) in enumerate(zip(results, detections)):
        if detection is None:
            print(f"Image {image_index}: no valid detection; skipped.")
            continue

        image = result.orig_img.copy()
        predicted_centers = leave_one_out_prediction(samples, sample_index, center_model)
        raw_yaw, robust_yaw, delta_yaw, error_l_mm = draw_result(
            image,
            detection,
            predicted_centers,
            sample_index,
            center_model,
            len(samples),
        )
        output_path = OUTPUT_DIR / f"lr_result_{image_index}.jpg"
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"Could not write output image: {output_path}")

        print(
            f"Image {image_index}: Pose Yaw (Raw)={raw_yaw:.2f} deg, "
            f"Robust Yaw (Linear Regression)={robust_yaw:.2f} deg, "
            f"Delta Yaw={delta_yaw:.2f} deg, "
            f"Error L_center={error_l_mm:.2f} mm -> {output_path}"
        )
        saved_count += 1
        sample_index += 1

    print(f"Saved {saved_count} image(s) to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
