import os
import cv2
import numpy as np
from ultralytics import YOLO

# Kich thuoc thuc te cua pallet (milimet)
PALLET_WIDTH_MM = 90.0
PALLET_HEIGHT_MM = 90.0

# Load mo hinh
model_path = r"D:\TANGCUONGANH\runs\pose\pallet_pose_project\pallet_model-2\weights\best.pt"
model = YOLO(model_path)

results = model.predict(source=r"D:\TANGCUONGANH\val\images", conf=0.5)

# Chi so keypoint dai dien cho 2 lo
LEFT_HOLE_KPTS = [2, 3, 4, 5]
RIGHT_HOLE_KPTS = [6, 7, 8, 9]

OUTPUT_DIR = r"D:\TANGCUONGANH\real_world_centers"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for i, r in enumerate(results):
    img = r.orig_img.copy()
    h, w = img.shape[:2]
    cam_center_x, cam_center_y = w / 2.0, h / 2.0

    if r.keypoints is not None and len(r.keypoints) > 0:
        kpts = r.keypoints.xy[0].cpu().numpy()

        # 1. Tinh goc Yaw tho va nan phang
        pt_left, pt_right = kpts[0], kpts[1]
        dx = pt_right[0] - pt_left[0]
        dy = pt_right[1] - pt_left[1]
        yaw_angle = np.degrees(np.arctan2(dy, dx))

        center = (
            float((pt_left[0] + pt_right[0]) / 2),
            float((pt_left[1] + pt_right[1]) / 2),
        )
        M = cv2.getRotationMatrix2D(center, yaw_angle, 1.0)
        img_aligned = cv2.warpAffine(img, M, (w, h))

        # Nan toa do keypoint
        ones = np.ones((len(kpts), 1))
        points_homo = np.hstack([kpts, ones])
        kpts_aligned = M.dot(points_homo.T).T

        # 2. Tinh ty le pixel sang mm
        pallet_width_px = np.linalg.norm(kpts_aligned[1] - kpts_aligned[0])
        mm_per_pixel = PALLET_WIDTH_MM / pallet_width_px if pallet_width_px > 0 else 0

        # 3. Tinh tam 2 lo tren anh nan phang
        c_left_px = np.mean(kpts_aligned[LEFT_HOLE_KPTS], axis=0)
        c_right_px = np.mean(kpts_aligned[RIGHT_HOLE_KPTS], axis=0)

        # Tinh tam tren anh goc de danh dau vi tri thuc te
        c_left_original_px = np.mean(kpts[LEFT_HOLE_KPTS], axis=0)
        c_right_original_px = np.mean(kpts[RIGHT_HOLE_KPTS], axis=0)

        # Danh dau hai tam tren anh goc: do = trai, xanh duong = phai
        img_original_marked = img.copy()
        for point, color, label in [
            (c_left_original_px, (0, 0, 255), "L"),
            (c_right_original_px, (255, 0, 0), "R"),
        ]:
            point_xy = (int(point[0]), int(point[1]))
            cv2.circle(img_original_marked, point_xy, 8, color, -1)
            cv2.putText(
                img_original_marked,
                label,
                (point_xy[0] + 10, point_xy[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, f"original_centers_{i}.jpg"),
            img_original_marked,
        )

        # Danh dau hai tam tren anh da nan
        img_aligned_marked = img_aligned.copy()
        for point, color, label in [
            (c_left_px, (0, 0, 255), "L"),
            (c_right_px, (255, 0, 0), "R"),
        ]:
            point_xy = (int(point[0]), int(point[1]))
            cv2.circle(img_aligned_marked, point_xy, 8, color, -1)
            cv2.putText(
                img_aligned_marked,
                label,
                (point_xy[0] + 10, point_xy[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, f"aligned_centers_{i}.jpg"),
            img_aligned_marked,
        )

        # 4. Quy doi do lech so voi tam camera sang mm
        offset_left_mm = (
            (c_left_px[0] - cam_center_x) * mm_per_pixel,
            (c_left_px[1] - cam_center_y) * mm_per_pixel,
        )
        offset_right_mm = (
            (c_right_px[0] - cam_center_x) * mm_per_pixel,
            (c_right_px[1] - cam_center_y) * mm_per_pixel,
        )

        distance_mm = np.linalg.norm(c_right_px - c_left_px) * mm_per_pixel

        print(f"=== UOC LUONG KHOANG CACH THUC TE (Anh {i}) ===")
        print(f" - Goc Yaw tho: {yaw_angle:.2f} deg")
        print(f" - Ty le quy doi: 1 Pixel = {mm_per_pixel:.3f} mm")
        print(
            f" - Tam lo TRAI so voi Camera: X = {offset_left_mm[0]:.1f} mm, "
            f"Y = {offset_left_mm[1]:.1f} mm"
        )
        print(
            f" - Tam lo PHAI so voi Camera: X = {offset_right_mm[0]:.1f} mm, "
            f"Y = {offset_right_mm[1]:.1f} mm"
        )
        print(f" - Khoang cach giua 2 tam lo: {distance_mm:.1f} mm\n")
