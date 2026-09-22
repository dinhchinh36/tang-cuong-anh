import os
from roboflow import Roboflow

# Khoi tao Roboflow voi API Key cua ban
rf = Roboflow(api_key="SRjYlGV6aD2z9b6WXO7J")

# Truy cap workspace va project
project = rf.workspace("le-dinh-chinh-wwvhm").project("pallettest39")
version = project.version(1)

print("--> Dang ket noi va tai dataset tu Roboflow...")
# Tai du lieu dinh dang YOLOv8 (Roboflow se tu nhan dien Pose/Keypoints cua project)
dataset = version.download("yolov8")

print(f"==> Tai ve thanh cong tai thu muc: {dataset.location}")
