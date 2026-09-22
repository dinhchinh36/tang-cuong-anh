import cv2
import numpy as np
from ultralytics import YOLO

PALLET_WIDTH_MM = 90.0
PALLET_HEIGHT_MM = 90.0

MODEL_PATH = r"D:\TANGCUONGANH\runs\pose\pallet_pose_project\pallet_model-2\weights\best.pt"
VAL_IMAGE_DIR = r"D:\TANGCUONGANH\val\images"

YAW_LEFT_KPT = 0
YAW_RIGHT_KPT = 1
LEFT_HOLE_KPTS = [4, 5, 6, 7]
RIGHT_HOLE_KPTS = [8, 9, 10, 11]

model = YOLO(MODEL_PATH)
results = model.predict(source=VAL_IMAGE_DIR, conf=0.5)

for i, result in enumerate(results):
    img = result.orig_img.copy()
    height, width = img.shape[:2]
    camera_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)

    if result.keypoints is None or len(result.keypoints) == 0:
        print(f"Anh {i}: Khong phat hien keypoints.")
        continue

    kpts = result.keypoints.xy[0].cpu().numpy()
    if len(kpts) < 12:
        print(f"Anh {i}: Chi co {len(kpts)} keypoints, can du 12.")
        continue

    # 1. Tinh yaw chuan tu doan noi diem 0 va diem 1
    pt_left = kpts[YAW_LEFT_KPT]
    pt_right = kpts[YAW_RIGHT_KPT]
    direction = pt_right - pt_left
    yaw_raw = np.degrees(np.arctan2(direction[1], direction[0]))

    # 2. Nan phang toan bo he toa do keypoints
    center = tuple(((pt_left + pt_right) / 2.0).astype(float))
    rotation_matrix = cv2.getRotationMatrix2D(center, float(yaw_raw), 1.0)
    ones = np.ones((len(kpts), 1), dtype=np.float32)
    kpts_aligned = rotation_matrix.dot(np.hstack([kpts, ones]).T).T

    # 3. Tinh tam trung binh cua hai nhom lo tren he toa do da nan
    left_center_aligned = np.mean(kpts_aligned[LEFT_HOLE_KPTS], axis=0)
    right_center_aligned = np.mean(kpts_aligned[RIGHT_HOLE_KPTS], axis=0)

    # 4. Dua hai tam nguoc ve he toa do anh goc
    inverse_rotation = cv2.invertAffineTransform(rotation_matrix)
    left_center_original = inverse_rotation.dot(
        np.append(left_center_aligned, 1.0)
    )
    right_center_original = inverse_rotation.dot(
        np.append(right_center_aligned, 1.0)
    )

    # 5. Quy doi mm theo chieu rong pallet 90 mm
    pallet_width_px = np.linalg.norm(
        kpts_aligned[YAW_RIGHT_KPT] - kpts_aligned[YAW_LEFT_KPT]
    )
    mm_per_pixel = PALLET_WIDTH_MM / pallet_width_px if pallet_width_px > 0 else 0.0
    left_offset_mm = (left_center_original - camera_center) * mm_per_pixel
    right_offset_mm = (right_center_original - camera_center) * mm_per_pixel
    hole_distance_mm = np.linalg.norm(right_center_original - left_center_original) * mm_per_pixel

    # Ve day du 12 keypoint va danh so ngay ben canh tung diem.
    for keypoint_index, point in enumerate(kpts[:12]):
        point_xy = tuple(np.round(point).astype(int))
        cv2.circle(img, point_xy, 5, (0, 200, 0), -1)
        cv2.putText(
            img,
            str(keypoint_index),
            (point_xy[0] + 7, point_xy[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )

    # Ve duong tham chieu Yaw 0 -> 1 va danh dau hai dau mut noi bat.
    cv2.line(
        img,
        tuple(np.round(pt_left).astype(int)),
        tuple(np.round(pt_right).astype(int)),
        (144, 238, 144),
        2,
    )
    for point, label in [(pt_left, "0"), (pt_right, "1")]:
        point_xy = tuple(np.round(point).astype(int))
        cv2.circle(img, point_xy, 10, (0, 0, 255), -1)
        cv2.putText(
            img,
            label,
            (point_xy[0] + 10, point_xy[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    for point, color, label in [
        (left_center_original, (0, 0, 255), "L"),
        (right_center_original, (255, 0, 0), "R"),
    ]:
        point_xy = tuple(np.round(point).astype(int))
        cv2.circle(img, point_xy, 9, color, -1)
        cv2.putText(
            img,
            label,
            (point_xy[0] + 12, point_xy[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        img,
        f"Yaw: {yaw_raw:.2f} deg",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    output_path = f"pallet_holes_perfect_{i}.jpg"
    cv2.imwrite(output_path, img)

    print(f"=== ANH {i} ===")
    print(f" - Yaw chuan (diem 0 -> 1): {yaw_raw:.2f} deg")
    print(f" - Ty le: 1 pixel = {mm_per_pixel:.3f} mm")
    print(
        f" - Tam L: X = {left_offset_mm[0]:.1f} mm, "
        f"Y = {left_offset_mm[1]:.1f} mm"
    )
    print(
        f" - Tam R: X = {right_offset_mm[0]:.1f} mm, "
        f"Y = {right_offset_mm[1]:.1f} mm"
    )
    print(f" - Khoang cach hai tam: {hole_distance_mm:.1f} mm")
    print(f" - Da luu: {output_path}\n")
