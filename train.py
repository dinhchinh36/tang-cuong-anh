import os
import torch
from ultralytics import YOLO

def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Khong tim thay CUDA trong PyTorch. Hay cai ban PyTorch CUDA "
            "truoc khi train bang GPU."
        )

    device = 0
    print(f"--> Dang su dung GPU: {torch.cuda.get_device_name(device)}")

    # 1. Đường dẫn data.yaml
    # Neu da tai ve bang Roboflow vao thu muc con:
    roboflow_yaml = os.path.join(r"d:\TANGCUONGANH", "pallettest39-1", "data.yaml")
    # Neu dung truc tiep data.yaml o thu muc hien tai:
    local_yaml = os.path.join(r"d:\TANGCUONGANH", "data.yaml")
    
    data_path = roboflow_yaml if os.path.exists(roboflow_yaml) else local_yaml
    print(f"--> Su dung du lieu tu: {data_path}")

    # 2. Tai mo hinh YOLOv8n-pose khoi tao (Pretrained weights)
    print("--> Dang tai mo hinh yolov8n-pose.pt...")
    model = YOLO('yolov8n-pose.pt')

    # 3. Bat dau huan luyen
    print("--> Bat dau qua trinh train mo hinh...")
    results = model.train(
        data=data_path,
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        workers=2,                       # Thích hợp và ổn định trên Windows
        project="pallet_pose_project",
        name="pallet_model"
    )
    
    print("==> Huan luyen hoan tat! Trong so tot nhat luu tai: pallet_pose_project/pallet_model/weights/best.pt")

if __name__ == '__main__':
    # Bat buoc tren Windows de tranh loi multiprocessing cua PyTorch DataLoader
    main()
