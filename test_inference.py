import os
from ultralytics import YOLO

# 1. Load mo hinh best.pt vua train xong
model_path = r"D:\TANGCUONGANH\runs\pose\pallet_pose_project\pallet_model-2\weights\best.pt"
model = YOLO(model_path)

# 2. Chay du doan tren tap anh validation
val_img_dir = r"D:\TANGCUONGANH\val\images"
results = model.predict(
    source=val_img_dir,
    save=True,
    conf=0.5,
    project="runs/pose",
    name="test_results"
)

print("==> Da xuat ket qua anh du doan vao thu muc: runs/pose/test_results")
