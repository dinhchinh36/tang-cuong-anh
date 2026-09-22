import os
import glob
import random
import math
import cv2
import numpy as np

# Cấu hình thư mục
WORKSPACE_DIR = r"d:\TANGCUONGANH"
TRAIN_IMG_DIR = os.path.join(WORKSPACE_DIR, "train", "images")
TRAIN_LBL_DIR = os.path.join(WORKSPACE_DIR, "train", "labels")
TARGET_COUNT = int(os.environ.get("AUGMENT_TARGET_COUNT", "500"))

random.seed(42)
np.random.seed(42)

def parse_yolo_pose_label(label_path, img_w, img_h):
    """
    Đọc file nhãn YOLO pose:
    class cx cy w h x1 y1 v1 ... x12 y12 v12 (tổng cộng 41 giá trị)
    """
    annotations = []
    if not os.path.exists(label_path):
        return annotations
    
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 41:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        
        # Bbox sang pixel [xmin, ymin, xmax, ymax]
        xmin = (cx - bw / 2.0) * img_w
        ymin = (cy - bh / 2.0) * img_h
        xmax = (cx + bw / 2.0) * img_w
        ymax = (cy + bh / 2.0) * img_h
        
        # 12 Keypoints sang pixel
        keypoints = []
        kpt_data = parts[5:41]
        for i in range(12):
            kx = float(kpt_data[i * 3]) * img_w
            ky = float(kpt_data[i * 3 + 1]) * img_h
            kv = int(float(kpt_data[i * 3 + 2]))
            keypoints.append([kx, ky, kv])
            
        annotations.append({
            'class_id': cls_id,
            'bbox': [xmin, ymin, xmax, ymax],
            'keypoints': keypoints
        })
    return annotations

def format_yolo_pose_label(annotations, img_w, img_h):
    """
    Chuyển annotations về định dạng dòng text YOLO pose chuẩn hóa [0, 1]
    """
    lines = []
    for ann in annotations:
        cls_id = ann['class_id']
        xmin, ymin, xmax, ymax = ann['bbox']
        
        # Clamp vào biên ảnh
        xmin = max(0.0, min(float(img_w), xmin))
        xmax = max(0.0, min(float(img_w), xmax))
        ymin = max(0.0, min(float(img_h), ymin))
        ymax = max(0.0, min(float(img_h), ymax))
        
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h
        cx = (xmin + xmax) / (2.0 * img_w)
        cy = (ymin + ymax) / (2.0 * img_h)
        
        tokens = [str(cls_id), f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
        for kx, ky, kv in ann['keypoints']:
            norm_kx = max(0.0, min(1.0, kx / img_w))
            norm_ky = max(0.0, min(1.0, ky / img_h))
            tokens.extend([f"{norm_kx:.6f}", f"{norm_ky:.6f}", str(kv)])
            
        lines.append(" ".join(tokens))
    return "\n".join(lines)

# ==================== CÁC PHÉP BIẾN ĐỔI QUANG HỌC ====================

def apply_brightness_contrast(img):
    alpha = random.uniform(0.75, 1.25)  # Contrast
    beta = random.uniform(-25, 25)      # Brightness
    return np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

def apply_gamma(img):
    gamma = random.uniform(0.75, 1.30)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)

def apply_clahe(img):
    clip_limit = random.uniform(1.5, 3.0)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    if len(img.shape) == 2:
        return clahe.apply(img)
    elif img.shape[2] == 1:
        return clahe.apply(img[:, :, 0])[:, :, np.newaxis]
    else:
        # Ảnh BGR
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_blur(img):
    blur_type = random.choice(['gaussian', 'motion'])
    if blur_type == 'gaussian':
        ksize = random.choice([3, 5])
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    else:
        # Motion blur
        size = random.choice([3, 5])
        kernel = np.zeros((size, size))
        if random.random() < 0.5:
            kernel[int((size - 1) / 2), :] = np.ones(size)
        else:
            np.fill_diagonal(kernel, 1)
        kernel = kernel / size
        return cv2.filter2D(img, -1, kernel)

def apply_noise(img):
    noise_type = random.choice(['gaussian', 'salt_pepper'])
    h, w = img.shape[:2]
    if noise_type == 'gaussian':
        sigma = random.uniform(6.0, 18.0)
        gauss = np.random.normal(0, sigma, img.shape).astype(np.float32)
        noisy = np.clip(img.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
        return noisy
    else:
        noisy = img.copy()
        num_sp = int(random.uniform(0.001, 0.004) * h * w)
        for _ in range(num_sp):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            val = 255 if random.random() < 0.5 else 0
            if len(img.shape) == 3:
                noisy[y, x] = [val, val, val]
            else:
                noisy[y, x] = val
        return noisy

def apply_random_shadow(img):
    h, w = img.shape[:2]
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)
    angle = random.uniform(0, 2 * math.pi)
    proj = xx * math.cos(angle) + yy * math.sin(angle)
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
    
    shadow_intensity = random.uniform(0.60, 0.85)
    shadow_mask = 1.0 - (1.0 - shadow_intensity) * proj
    if len(img.shape) == 3:
        shadow_mask = shadow_mask[:, :, np.newaxis]
        
    shadowed = np.clip(img.astype(np.float32) * shadow_mask, 0, 255).astype(np.uint8)
    return shadowed

# ==================== BIẾN ĐỔI HÌNH HỌC & ĐỒNG BỘ TỌA ĐỘ ====================

def get_geometric_transform_matrix(w, h):
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    angle = math.radians(random.uniform(-7.0, 7.0))
    scale = random.uniform(0.88, 1.12)
    tx = random.uniform(-0.07 * w, 0.07 * w)
    ty = random.uniform(-0.07 * h, 0.07 * h)
    
    cx, cy = w / 2.0, h / 2.0
    cos_a = math.cos(angle) * scale
    sin_a = math.sin(angle) * scale
    
    dst_pts = []
    for pt in src_pts:
        px, py = pt[0] - cx, pt[1] - cy
        rx = cos_a * px - sin_a * py + cx + tx
        ry = sin_a * px + cos_a * py + cy + ty
        
        jitter_x = random.uniform(-0.04 * w, 0.04 * w)
        jitter_y = random.uniform(-0.04 * h, 0.04 * h)
        dst_pts.append([rx + jitter_x, ry + jitter_y])
        
    dst_pts = np.float32(dst_pts)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return M

def transform_point(M, x, y):
    vec = np.array([x, y, 1.0], dtype=np.float32)
    res = M.dot(vec)
    w_norm = res[2] if abs(res[2]) > 1e-6 else 1e-6
    return float(res[0] / w_norm), float(res[1] / w_norm)

def apply_geometric_transform(img, annotations, M):
    h, w = img.shape[:2]
    warped_img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    
    transformed_anns = []
    for ann in annotations:
        cls_id = ann['class_id']
        xmin, ymin, xmax, ymax = ann['bbox']
        
        box_corners = [
            (xmin, ymin), (xmax, ymin),
            (xmax, ymax), (xmin, ymax)
        ]
        trans_box_pts = [transform_point(M, bx, by) for bx, by in box_corners]
        
        trans_kpts = []
        out_of_bounds_count = 0
        for kx, ky, kv in ann['keypoints']:
            if kv == 0:
                trans_kpts.append([kx, ky, 0])
                continue
            tx, ty = transform_point(M, kx, ky)
            if 0 <= tx < w and 0 <= ty < h:
                trans_kpts.append([tx, ty, kv])
            else:
                trans_kpts.append([tx, ty, 0])
                out_of_bounds_count += 1
                
        if out_of_bounds_count > 2:
            return None, None
            
        all_x = [pt[0] for pt in trans_box_pts] + [kp[0] for kp in trans_kpts if kp[2] > 0]
        all_y = [pt[1] for pt in trans_box_pts] + [kp[1] for kp in trans_kpts if kp[2] > 0]
        
        new_xmin = max(0.0, min(all_x))
        new_xmax = min(float(w), max(all_x))
        new_ymin = max(0.0, min(all_y))
        new_ymax = min(float(h), max(all_y))
        
        if (new_xmax - new_xmin) < 0.15 * w or (new_ymax - new_ymin) < 0.15 * h:
            return None, None
            
        transformed_anns.append({
            'class_id': cls_id,
            'bbox': [new_xmin, new_ymin, new_xmax, new_ymax],
            'keypoints': trans_kpts
        })
        
    return warped_img, transformed_anns

# ==================== MÔ PHỎNG CHE KHUẤT (CUTOUT) ====================

def apply_cutout(img, annotations):
    h, w = img.shape[:2]
    num_cuts = random.randint(1, 2)
    img_cut = img.copy()
    
    for _ in range(num_cuts):
        cut_w = random.randint(int(0.04 * w), int(0.08 * w))
        cut_h = random.randint(int(0.04 * h), int(0.08 * h))
        
        if annotations:
            box = annotations[0]['bbox']
            cx = random.uniform(box[0] + cut_w, box[2] - cut_w) if (box[2] - box[0]) > 2 * cut_w else random.uniform(0, w)
            cy = random.uniform(box[1] + cut_h, box[3] - cut_h) if (box[3] - box[1]) > 2 * cut_h else random.uniform(0, h)
        else:
            cx = random.uniform(0, w)
            cy = random.uniform(0, h)
            
        x1 = max(0, int(cx - cut_w / 2))
        y1 = max(0, int(cy - cut_h / 2))
        x2 = min(w, int(cx + cut_w / 2))
        y2 = min(h, int(cy + cut_h / 2))
        
        cut_color = random.choice([0, 50, 120, 180])
        if len(img_cut.shape) == 3:
            img_cut[y1:y2, x1:x2] = [cut_color, cut_color, cut_color]
        else:
            img_cut[y1:y2, x1:x2] = cut_color
            
        for ann in annotations:
            for kp in ann['keypoints']:
                if kp[2] == 2 and (x1 <= kp[0] <= x2) and (y1 <= kp[1] <= y2):
                    kp[2] = 1
                    
    return img_cut, annotations

# ==================== MAIN PIPELINE ====================

def run_augmentation():
    img_files = sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpg")) + glob.glob(os.path.join(TRAIN_IMG_DIR, "*.png")))
    base_img_files = [f for f in img_files if not os.path.basename(f).startswith("aug_")]
    
    print(f"--> Tim thay {len(base_img_files)} anh goc trong tap train.")
    if len(base_img_files) == 0:
        print("Loi: Khong tim thay anh trong thu muc train/images!")
        return

    generated_count = 0
    attempt = 0
    max_attempts = TARGET_COUNT * 10
    
    print(f"--> Bat dau tang cuong {TARGET_COUNT} anh...")

    while generated_count < TARGET_COUNT and attempt < max_attempts:
        attempt += 1
        src_img_path = random.choice(base_img_files)
        base_name = os.path.splitext(os.path.basename(src_img_path))[0]
        src_lbl_path = os.path.join(TRAIN_LBL_DIR, base_name + ".txt")
        
        if not os.path.exists(src_lbl_path):
            continue
            
        img = cv2.imread(src_img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        annotations = parse_yolo_pose_label(src_lbl_path, w, h)
        if not annotations:
            continue
            
        # 1. Biến đổi hình học
        M = get_geometric_transform_matrix(w, h)
        aug_img, aug_anns = apply_geometric_transform(img, annotations, M)
        if aug_img is None or aug_anns is None:
            continue
            
        # 2. Biến đổi quang học
        if random.random() < 0.85:
            aug_img = apply_brightness_contrast(aug_img)
        if random.random() < 0.60:
            aug_img = apply_gamma(aug_img)
        if random.random() < 0.45:
            aug_img = apply_clahe(aug_img)
        if random.random() < 0.35:
            aug_img = apply_blur(aug_img)
        if random.random() < 0.40:
            aug_img = apply_noise(aug_img)
        if random.random() < 0.30:
            aug_img = apply_random_shadow(aug_img)
            
        # 3. Cutout
        if random.random() < 0.35:
            aug_img, aug_anns = apply_cutout(aug_img, aug_anns)
            
        # 4. Xuất file
        generated_count += 1
        out_name = f"aug_{generated_count:04d}"
        out_img_path = os.path.join(TRAIN_IMG_DIR, f"{out_name}.jpg")
        out_lbl_path = os.path.join(TRAIN_LBL_DIR, f"{out_name}.txt")
        
        cv2.imwrite(out_img_path, aug_img)
        
        lbl_content = format_yolo_pose_label(aug_anns, w, h)
        with open(out_lbl_path, 'w', encoding='utf-8') as f:
            f.write(lbl_content + "\n")
            
        if generated_count % 50 == 0 or generated_count == TARGET_COUNT:
            print(f"    [Da tao {generated_count}/{TARGET_COUNT} anh ({generated_count / TARGET_COUNT * 100:.1f}%)]")

    print(f"==> HOAN THANH: Da tao thanh cong {generated_count} anh va file nhan trong train/.")

if __name__ == "__main__":
    run_augmentation()
