import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load mo hinh
model_path = r"D:\TANGCUONGANH\runs\pose\pallet_pose_project\pallet_model-2\weights\best.pt"
model = YOLO(model_path)

# 2. Du doan tren thu muc validation
results = model.predict(source=r"D:\TANGCUONGANH\val\images", conf=0.5)

for i, r in enumerate(results):
    img = r.orig_img.copy()

    if r.keypoints is not None and len(r.keypoints) > 0:
        # Lay toa do 12 keypoint
        kpts = r.keypoints.xy[0].cpu().numpy()

        # Gia dinh Keypoint 0 la goc duoi-trai va Keypoint 1 la goc duoi-phai
        pt_left = kpts[0]
        pt_right = kpts[1]

        # Tinh do lech dx, dy
        dx = pt_right[0] - pt_left[0]
        dy = pt_right[1] - pt_left[1]

        # Tinh goc Yaw tho (don vi: Do)
        yaw_angle = np.degrees(np.arctan2(dy, dx))

        # Ve duong noi 2 diem canh day va in goc len anh
        cv2.line(
            img,
            (int(pt_left[0]), int(pt_left[1])),
            (int(pt_right[0]), int(pt_right[1])),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img,
            f"Yaw Thoi: {yaw_angle:.2f} deg",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        # Luu anh ket qua
        output_path = f"yaw_raw_result_{i}.jpg"
        cv2.imwrite(output_path, img)
        print(f"Anh {i}: Goc Yaw tho = {yaw_angle:.2f} deg (Da luu {output_path})")
