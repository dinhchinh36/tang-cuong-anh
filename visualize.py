import os
import glob
import random
import cv2

WORKSPACE_DIR = r"d:\TANGCUONGANH"
TRAIN_IMG_DIR = os.path.join(WORKSPACE_DIR, "train", "images")
TRAIN_LBL_DIR = os.path.join(WORKSPACE_DIR, "train", "labels")
PREVIEW_DIR = os.path.join(WORKSPACE_DIR, "samples_preview")

os.makedirs(PREVIEW_DIR, exist_ok=True)

# Màu sắc (BGR)
COLOR_BBOX = (0, 255, 0)       # Xanh lá
COLOR_OUTER = (255, 100, 0)    # Xanh dương
COLOR_HOLE1 = (0, 255, 255)    # Vàng
COLOR_HOLE2 = (0, 140, 255)    # Cam
COLOR_KPT = (0, 0, 255)        # Đỏ
COLOR_TEXT = (255, 255, 255)   # Trắng

def draw_pose(img, label_path):
    if not os.path.exists(label_path):
        return img
        
    h, w = img.shape[:2]
    canvas = img.copy()
    
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 41:
            continue
            
        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        
        # Bbox
        xmin = int((cx - bw / 2.0) * w)
        ymin = int((cy - bh / 2.0) * h)
        xmax = int((cx + bw / 2.0) * w)
        ymax = int((cy + bh / 2.0) * h)
        
        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), COLOR_BBOX, 2)
        cv2.putText(canvas, f"KHUNG", (xmin, max(15, ymin - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BBOX, 2)
        
        # 12 Keypoints
        kpts = []
        kpt_data = parts[5:41]
        for i in range(12):
            kx = int(float(kpt_data[i * 3]) * w)
            ky = int(float(kpt_data[i * 3 + 1]) * h)
            kv = int(float(kpt_data[i * 3 + 2]))
            kpts.append((kx, ky, kv))
            
        # Vẽ đường nối các cụm điểm
        # 1. Khung ngoài (0 -> 1 -> 2 -> 3 -> 0)
        outer_idx = [0, 1, 2, 3, 0]
        for idx in range(len(outer_idx) - 1):
            p1 = kpts[outer_idx[idx]]
            p2 = kpts[outer_idx[idx + 1]]
            if p1[2] > 0 and p2[2] > 0:
                cv2.line(canvas, (p1[0], p1[1]), (p2[0], p2[1]), COLOR_OUTER, 2)
                
        # 2. Lỗ nâng trái (4 -> 5 -> 6 -> 7 -> 4)
        hole1_idx = [4, 5, 6, 7, 4]
        for idx in range(len(hole1_idx) - 1):
            p1 = kpts[hole1_idx[idx]]
            p2 = kpts[hole1_idx[idx + 1]]
            if p1[2] > 0 and p2[2] > 0:
                cv2.line(canvas, (p1[0], p1[1]), (p2[0], p2[1]), COLOR_HOLE1, 2)
                
        # 3. Lỗ nâng phải (8 -> 9 -> 10 -> 11 -> 8)
        hole2_idx = [8, 9, 10, 11, 8]
        for idx in range(len(hole2_idx) - 1):
            p1 = kpts[hole2_idx[idx]]
            p2 = kpts[hole2_idx[idx + 1]]
            if p1[2] > 0 and p2[2] > 0:
                cv2.line(canvas, (p1[0], p1[1]), (p2[0], p2[1]), COLOR_HOLE2, 2)
                
        # Vẽ từng điểm và đánh số thứ tự 0 - 11
        for i, (kx, ky, kv) in enumerate(kpts):
            if kv == 0:
                continue
            pt_color = COLOR_KPT if kv == 2 else (100, 100, 100) # xám nếu bị che khuất
            cv2.circle(canvas, (kx, ky), 4, pt_color, -1)
            cv2.circle(canvas, (kx, ky), 5, (255, 255, 255), 1)
            cv2.putText(canvas, str(i), (kx + 4, ky - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1)
                        
    return canvas

def main():
    # Lấy 1 ảnh gốc và 5 ảnh tăng cường ngẫu nhiên
    orig_files = [f for f in glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpg")) if not os.path.basename(f).startswith("aug_")]
    aug_files = sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, "aug_*.jpg")))
    
    selected_files = []
    if orig_files:
        selected_files.append(orig_files[0])
    if aug_files:
        sample_aug = random.sample(aug_files, min(5, len(aug_files)))
        selected_files.extend(sample_aug)
        
    print(f"--> Dang tao anh mau xem truoc cho {len(selected_files)} anh vao: {PREVIEW_DIR}")
    
    for fpath in selected_files:
        fname = os.path.basename(fpath)
        base = os.path.splitext(fname)[0]
        lpath = os.path.join(TRAIN_LBL_DIR, base + ".txt")
        
        img = cv2.imread(fpath)
        if img is None:
            continue
            
        vis_img = draw_pose(img, lpath)
        out_path = os.path.join(PREVIEW_DIR, f"preview_{fname}")
        cv2.imwrite(out_path, vis_img)
        print(f"    + Da luu: preview_{fname}")
        
    print(f"==> Xem thu anh mau thanh cong trong thu muc samples_preview/.")

if __name__ == "__main__":
    main()
