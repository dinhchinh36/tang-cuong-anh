import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load mo hinh best.pt
model_path = r"D:\TANGCUONGANH\runs\pose\pallet_pose_project\pallet_model-2\weights\best.pt"
model = YOLO(model_path)

results = model.predict(source=r"D:\TANGCUONGANH\val\images", conf=0.5)

for i, r in enumerate(results):
    img = r.orig_img.copy()
    h, w = img.shape[:2]

    if r.keypoints is not None and len(r.keypoints) > 0:
        kpts = r.keypoints.xy[0].cpu().numpy()

        # Lay 2 diem tham chieu canh mep mat truoc
        pt_left = kpts[0]
        pt_right = kpts[1]

        # Lan 1: Tinh goc tho
        dx1 = pt_right[0] - pt_left[0]
        dy1 = pt_right[1] - pt_left[1]
        yaw_raw = np.degrees(np.arctan2(dy1, dx1))

        # Xoay nan phang anh va keypoints ve cung mat phang ngang
        center = (
            float((pt_left[0] + pt_right[0]) / 2),
            float((pt_left[1] + pt_right[1]) / 2),
        )
        M = cv2.getRotationMatrix2D(center, yaw_raw, 1.0)

        # Nan anh
        img_aligned = cv2.warpAffine(img, M, (w, h))

        # Nan toa do toan bo 12 keypoints
        ones = np.ones((len(kpts), 1))
        points_homo = np.hstack([kpts, ones])
        kpts_aligned = M.dot(points_homo.T).T

        # Lan 2: Tinh goc tinh chinh tren he toa do da phang
        pt_left_aligned = kpts_aligned[0]
        pt_right_aligned = kpts_aligned[1]

        dx2 = pt_right_aligned[0] - pt_left_aligned[0]
        dy2 = pt_right_aligned[1] - pt_left_aligned[1]
        yaw_residual = np.degrees(np.arctan2(dy2, dx2))

        yaw_final = yaw_raw + yaw_residual

        # Ve kiem tra tren anh da nan
        cv2.line(
            img_aligned,
            (int(pt_left_aligned[0]), int(pt_left_aligned[1])),
            (int(pt_right_aligned[0]), int(pt_right_aligned[1])),
            (255, 0, 0),
            2,
        )
        cv2.putText(
            img_aligned,
            f"Yaw Refined (Lan 2): {yaw_final:.2f} deg",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        # Luu anh ket qua lan 2
        output_path = f"yaw_refined_result_{i}.jpg"
        cv2.imwrite(output_path, img_aligned)
        print(
            f"Anh {i}: Yaw Tho = {yaw_raw:.2f} deg | "
            f"Yaw Tinh Chinh = {yaw_final:.2f} deg"
        )
