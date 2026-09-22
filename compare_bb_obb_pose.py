from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = Path(r"D:\TANGCUONGANH\runs\pose\pallet_pose_project\pallet_model-2\weights\best.pt")
IMAGE_DIR = Path(r"D:\TANGCUONGANH\val\images")
OUTPUT_DIR = Path(__file__).resolve().parent
CONFIDENCE = 0.5
KEYPOINT_COUNT = 12

# Physical dimensions of the square pallet front face in mm.
PALLET_WIDTH_MM = 90.0
PALLET_HEIGHT_MM = 90.0

# OpenCV colors are BGR.
RED = (0, 0, 255)
CYAN = (255, 255, 0)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
ORANGE = (0, 165, 255)


def draw_label(image, text, position, color, scale=0.55, thickness=2):
    x, y = position
    cv2.putText(
        image,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def calculate_bb_angle(image, box):
    """
    Xác định góc xoay của pallet bên trong Bounding Box (BB) thay vì gán cố định 0.00 độ.
    Thuật toán:
    1. Trích xuất vùng ROI của BB.
    2. Dùng Canny edge detection + HoughLinesP tìm các cạnh ngang chính của pallet.
    3. Nếu có cạnh ngang, tính góc trung bình có trọng số theo chiều dài.
    4. Fallback dùng PCA trên các điểm biên cạnh để xác định trục chính của vật thể.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    h_img, w_img = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, None

    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=35, minLineLength=25, maxLineGap=10
    )
    line_angles = []
    weights = []
    best_line = None
    max_len = 0.0

    if lines is not None:
        for line in lines:
            lx1, ly1, lx2, ly2 = np.asarray(line).reshape(-1)
            dx = float(lx2 - lx1)
            dy = float(ly2 - ly1)
            length = float(np.hypot(dx, dy))
            if length < 1e-3:
                continue
            deg = float(np.degrees(np.arctan2(dy, dx)))
            # Chuẩn hóa về [-90, 90]
            while deg > 90.0:
                deg -= 180.0
            while deg <= -90.0:
                deg += 180.0
            # Mép pallet nằm ngang nên góc thường nằm trong khoảng [-40, 40]
            if abs(deg) <= 40.0:
                line_angles.append(deg)
                weights.append(length)
                if length > max_len:
                    max_len = length
                    best_line = (lx1 + x1, ly1 + y1, lx2 + x1, ly2 + y1)

    if line_angles:
        bb_angle = float(np.average(line_angles, weights=weights))
        return bb_angle, best_line

    # Fallback: PCA trên tọa độ các điểm cạnh
    pts = np.argwhere(edges > 0)
    if len(pts) >= 20:
        pts_xy = np.fliplr(pts).astype(np.float32)
        _, eigenvectors = cv2.PCACompute(pts_xy, mean=None)
        v = eigenvectors[0]
        deg = float(np.degrees(np.arctan2(v[1], v[0])))
        while deg > 90.0:
            deg -= 180.0
        while deg <= -90.0:
            deg += 180.0
        if deg > 45.0:
            deg -= 90.0
        elif deg < -45.0:
            deg += 90.0
        return deg, None

    return 0.0, None


def normalize_obb_angle(rect):
    """
    Chuẩn hóa góc trả về từ cv2.minAreaRect() theo trục dài của pallet.
    """
    (cx, cy), (w, h), angle = rect
    if w < h:
        angle += 90.0

    while angle > 45.0:
        angle -= 90.0
    while angle < -45.0:
        angle += 90.0

    return float(angle)


def rectify_points_with_homography(points):
    """Map image keypoints to a square 90 x 90 mm pallet plane."""
    if len(points) < 4 or not np.isfinite(points[:4]).all():
        return None

    image_corners = points[:4].astype(np.float32)
    plane_corners = np.array([
        [0.0, 0.0],
        [PALLET_WIDTH_MM, 0.0],
        [PALLET_WIDTH_MM, PALLET_HEIGHT_MM],
        [0.0, PALLET_HEIGHT_MM],
    ], dtype=np.float32)
    homography, _ = cv2.findHomography(image_corners, plane_corners, 0)
    if homography is None:
        return None

    return cv2.perspectiveTransform(
        points.reshape(-1, 1, 2).astype(np.float32), homography
    ).reshape(-1, 2)


def calculate_pnp_yaw(points, img_shape):
    """
    Tính góc YAW 3D thực tế bằng thuật toán PnP (Perspective-n-Point)
    dựa trên 4 keypoints góc ngoài (0: Trên-Trái, 1: Trên-Phải, 2: Dưới-Phải, 3: Dưới-Trái).
    """
    if len(points) < 4 or not np.isfinite(points[:4]).all():
        return None, None, None

    half_w = PALLET_WIDTH_MM / 2.0
    half_h = PALLET_HEIGHT_MM / 2.0

    # Tọa độ 3D của mặt trước vuông 90 x 90 mm, gốc tại tâm, Z = 0.
    obj_points = np.array([
        [-half_w, -half_h, 0.0],  # 0: Trên - Trái
        [ half_w, -half_h, 0.0],  # 1: Trên - Phải
        [ half_w,  half_h, 0.0],  # 2: Dưới - Phải
        [-half_w,  half_h, 0.0],  # 3: Dưới - Trái
    ], dtype=np.float64)

    img_points = points[:4].astype(np.float64)

    h, w = img_shape[:2]
    focal_length = float(w)  # Xấp xỉ tiêu cự f ~ w khi chưa calibrate
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array([
        [focal_length, 0.0, cx],
        [0.0, focal_length, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        obj_points, img_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None, None, None

    # Ma trận xoay Rodrigues
    R, _ = cv2.Rodrigues(rvec)

    # Trích xuất góc Euler (độ):
    # Hệ trục Camera OpenCV: X sang phải, Y hướng xuống, Z nhìn thẳng ra trước
    # Yaw (quay quanh trục Y: lệch hướng trái/phải trên mặt sàn)
    yaw_3d = float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))
    pitch_3d = float(np.degrees(-np.arcsin(np.clip(R[1, 2], -1.0, 1.0))))
    roll_3d = float(np.degrees(np.arctan2(R[1, 0], R[1, 1])))

    return yaw_3d, pitch_3d, roll_3d


def calculate_pose_angles(points, img_shape):
    """Return in-plane roll and out-of-plane yaw from the four outer corners."""
    if len(points) < 4 or not np.isfinite(points[:4]).all():
        return None, None, None

    rectified_points = rectify_points_with_homography(points)
    if rectified_points is None:
        return None, None, None

    point_left, point_right = points[0], points[1]
    roll_2d = float(np.degrees(np.arctan2(
        point_right[1] - point_left[1],
        point_right[0] - point_left[0],
    )))
    yaw_3d, _, _ = calculate_pnp_yaw(points, img_shape)
    return roll_2d, yaw_3d, rectified_points


def draw_detection(image, box, keypoints):
    """Draw BB, OBB, and Pose for one detected pallet with properly calculated angles."""
    x1, y1, x2, y2 = [int(value) for value in box]

    # 1. BB: Xác định góc thực tế từ ROI thay vì gán 0.00
    bb_angle, best_line = calculate_bb_angle(image, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), RED, 2)
    if best_line is not None:
        cv2.line(
            image,
            (best_line[0], best_line[1]),
            (best_line[2], best_line[3]),
            (0, 0, 180),
            2,
            cv2.LINE_AA,
        )
    draw_label(image, f"BB Angle: {bb_angle:.2f} deg", (x1, max(20, y1 - 8)), RED)

    points = np.asarray(keypoints, dtype=np.float32)
    if len(points) != KEYPOINT_COUNT:
        return bb_angle, None, None, None

    valid = np.isfinite(points).all(axis=1)
    if int(valid.sum()) < 3:
        return bb_angle, None, None, None

    # 2. OBB: minAreaRect từ keypoints chuẩn hóa góc
    valid_points = points[valid].reshape(-1, 1, 2)
    rect = cv2.minAreaRect(valid_points)
    obb_points = cv2.boxPoints(rect).astype(np.int32)
    cv2.polylines(image, [obb_points], True, CYAN, 2, cv2.LINE_AA)
    obb_angle = normalize_obb_angle(rect)
    draw_label(
        image,
        f"OBB Yaw: {obb_angle:.2f} deg",
        (x1, min(image.shape[0] - 10, y2 + 22)),
        CYAN,
    )

    # 3. Roll trong mặt phẳng ảnh: góc của cạnh 0 -> 1.
    roll_2d = None
    if valid[0] and valid[1]:
        point_left = points[0]
        point_right = points[1]
        dx = point_right[0] - point_left[0]
        dy = point_right[1] - point_left[1]
        roll_2d = float(np.degrees(np.arctan2(dy, dx)))
        cv2.line(
            image,
            tuple(np.round(point_left).astype(int)),
            tuple(np.round(point_right).astype(int)),
            GREEN,
            3,
            cv2.LINE_AA,
        )

    # 4. Homography giữ các cặp cạnh/lỗ song song trên mặt phẳng thực;
    # Yaw được lấy riêng từ PnP để không nhầm với Roll 2D.
    roll_2d, yaw_3d, rectified_points = calculate_pose_angles(points, image.shape)
    if roll_2d is not None:
        draw_label(
            image,
            f"Roll (In-plane): {roll_2d:.2f} deg",
            (x1, min(image.shape[0] - 34, y2 + 44)),
            GREEN,
        )
    if yaw_3d is not None:
        draw_label(
            image,
            f"Yaw (3D Out-of-plane): {yaw_3d:.2f} deg",
            (x1, min(image.shape[0] - 10, y2 + 66)),
            ORANGE,
        )

    # Vẽ 12 keypoints và số thứ tự
    for index, point in enumerate(points):
        if not valid[index]:
            continue
        center = tuple(np.round(point).astype(int))
        cv2.circle(image, center, 5, GREEN, -1, cv2.LINE_AA)
        draw_label(image, str(index), (center[0] + 7, center[1] - 7), WHITE, 0.5, 1)

    # Vẽ tâm lỗ L (4-7) và R (8-11)
    if valid[4:8].all():
        left_center = np.round(points[4:8].mean(axis=0)).astype(int)
        cv2.circle(image, tuple(left_center), 8, RED, -1, cv2.LINE_AA)
        draw_label(image, "L", left_center + np.array([10, 5]), RED)
    if valid[8:12].all():
        right_center = np.round(points[8:12].mean(axis=0)).astype(int)
        cv2.circle(image, tuple(right_center), 8, BLUE, -1, cv2.LINE_AA)
        draw_label(image, "R", right_center + np.array([10, 5]), BLUE)

    return bb_angle, obb_angle, roll_2d, yaw_3d


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Validation image directory not found: {IMAGE_DIR}")

    model = YOLO(str(MODEL_PATH))
    results = model.predict(source=str(IMAGE_DIR), conf=CONFIDENCE, verbose=False)

    saved_count = 0
    for image_index, result in enumerate(results):
        image = result.orig_img.copy()
        bb_values = []
        obb_values = []
        pose_2d_values = []
        pose_3d_values = []

        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is not None and keypoints is not None:
            box_values = boxes.xyxy.cpu().numpy()
            point_values = keypoints.xy.cpu().numpy()

            for detection_index, box in enumerate(box_values):
                bb_ang, obb_ang, pose_2d, yaw_3d = draw_detection(
                    image,
                    box,
                    point_values[detection_index],
                )
                if bb_ang is not None:
                    bb_values.append(bb_ang)
                if obb_ang is not None:
                    obb_values.append(obb_ang)
                if pose_2d is not None:
                    pose_2d_values.append(pose_2d)
                if yaw_3d is not None:
                    pose_3d_values.append(yaw_3d)
        elif boxes is not None:
            for box in boxes.xyxy.cpu().numpy():
                bb_ang, _ = calculate_bb_angle(image, box)
                x1, y1, x2, y2 = [int(value) for value in box]
                cv2.rectangle(image, (x1, y1), (x2, y2), RED, 2)
                draw_label(image, f"BB Angle: {bb_ang:.2f} deg", (x1, max(20, y1 - 8)), RED)
                bb_values.append(bb_ang)

        bb_text = (
            "BB Angle: " + ", ".join(f"{val:.2f} deg" for val in bb_values)
            if bb_values
            else "BB Angle: N/A"
        )
        obb_text = (
            "OBB Yaw: " + ", ".join(f"{val:.2f} deg" for val in obb_values)
            if obb_values
            else "OBB Yaw: N/A"
        )
        pose_2d_text = (
            "Roll (In-plane): " + ", ".join(f"{val:.2f} deg" for val in pose_2d_values)
            if pose_2d_values
            else "Roll (In-plane): N/A"
        )
        pose_3d_text = (
            "Yaw (3D Out-of-plane): " + ", ".join(f"{val:.2f} deg" for val in pose_3d_values)
            if pose_3d_values
            else "Yaw (3D Out-of-plane): N/A"
        )

        draw_label(image, bb_text, (30, 30), RED, 0.7, 2)
        draw_label(image, obb_text, (30, 58), CYAN, 0.7, 2)
        draw_label(image, pose_2d_text, (30, 86), GREEN, 0.7, 2)
        draw_label(image, pose_3d_text, (30, 114), ORANGE, 0.7, 2)

        output_path = OUTPUT_DIR / f"compare_all_{image_index}.jpg"
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"Could not write output image: {output_path}")
        saved_count += 1
        print(
            f"Image {image_index}: {bb_text} | {obb_text} | "
            f"{pose_2d_text} | {pose_3d_text} -> {output_path}"
        )

    print(f"Saved {saved_count} comparison image(s) to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()