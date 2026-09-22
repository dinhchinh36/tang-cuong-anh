import os
import shutil
import random

WORKSPACE = r"d:\TANGCUONGANH"
TRAIN_IMG_DIR = os.path.join(WORKSPACE, "train", "images")
TRAIN_LBL_DIR = os.path.join(WORKSPACE, "train", "labels")

VALID_DIR = os.path.join(WORKSPACE, "valid")
VALID_IMG_DIR = os.path.join(VALID_DIR, "images")
VALID_LBL_DIR = os.path.join(VALID_DIR, "labels")

VAL_DIR = os.path.join(WORKSPACE, "val")
VAL_IMG_DIR = os.path.join(VAL_DIR, "images")
VAL_LBL_DIR = os.path.join(VAL_DIR, "labels")

YAML_PATH = os.path.join(WORKSPACE, "data.yaml")

SEED = 42
VAL_RATIO = 0.15

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

def main():
    print("=" * 60)
    print("PHÂN CHIA TẬP DỮ LIỆU TRAIN / VAL (YOLOv8 POSE)")
    print("=" * 60)

    # Bước 1: Gộp dữ liệu từ valid/ cũ vào train/ (nếu tồn tại valid/)
    if os.path.exists(VALID_DIR):
        print("--> Bước 1: Gộp dữ liệu từ thư mục 'valid/' cũ vào 'train/'...")
        valid_imgs = [f for f in os.listdir(VALID_IMG_DIR) if os.path.splitext(f)[1].lower() in VALID_EXTS] if os.path.exists(VALID_IMG_DIR) else []
        for img_name in valid_imgs:
            base_name = os.path.splitext(img_name)[0]
            lbl_name = base_name + ".txt"
            
            src_img = os.path.join(VALID_IMG_DIR, img_name)
            src_lbl = os.path.join(VALID_LBL_DIR, lbl_name)
            
            dst_img = os.path.join(TRAIN_IMG_DIR, img_name)
            dst_lbl = os.path.join(TRAIN_LBL_DIR, lbl_name)
            
            shutil.move(src_img, dst_img)
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, dst_lbl)
            print(f"    + Đã chuyển: {img_name} và {lbl_name} -> train/")
            
        # Xóa thư mục valid/ cũ sau khi gộp
        shutil.rmtree(VALID_DIR, ignore_errors=True)
        print("    => Đã dọn dẹp thư mục 'valid/' cũ.")
    else:
        print("--> Không tìm thấy thư mục 'valid/' cũ, tiếp tục quét từ 'train/'...")

    # Bước 2: Quét toàn bộ cặp ảnh - nhãn trong train/
    print("\n--> Bước 2: Quét và kiểm tra tính toàn vẹn cặp dữ liệu trong 'train/'...")
    train_imgs = [f for f in os.listdir(TRAIN_IMG_DIR) if os.path.splitext(f)[1].lower() in VALID_EXTS]
    train_lbls = [f for f in os.listdir(TRAIN_LBL_DIR) if f.endswith(".txt")]
    
    img_stems = {os.path.splitext(f)[0]: f for f in train_imgs}
    lbl_stems = {os.path.splitext(f)[0]: f for f in train_lbls}
    
    # Kiểm tra tính toàn vẹn (Integrity Check)
    orphaned_imgs = set(img_stems.keys()) - set(lbl_stems.keys())
    orphaned_lbls = set(lbl_stems.keys()) - set(img_stems.keys())
    
    if orphaned_imgs:
        raise ValueError(f"LỖI: Phát hiện ảnh không có file nhãn: {orphaned_imgs}")
    if orphaned_lbls:
        raise ValueError(f"LỖI: Phát hiện nhãn không có file ảnh: {orphaned_lbls}")
        
    matched_stems = sorted(list(img_stems.keys()))
    total_samples = len(matched_stems)
    print(f"    => Tổng số mẫu hợp lệ (đầy đủ cặp ảnh & nhãn): {total_samples}")

    # Bước 3: Phân chia tập dữ liệu với Seed = 42
    print(f"\n--> Bước 3: Phân chia với seed = {SEED} và tỷ lệ Val = {VAL_RATIO * 100:.0f}%...")
    random.seed(SEED)
    shuffled_stems = matched_stems.copy()
    random.shuffle(shuffled_stems)
    
    val_count = int(round(total_samples * VAL_RATIO))
    train_count = total_samples - val_count
    
    val_stems = set(shuffled_stems[:val_count])
    train_stems = set(shuffled_stems[val_count:])
    
    print(f"    + Số lượng mẫu Train: {train_count} ({train_count / total_samples * 100:.1f}%)")
    print(f"    + Số lượng mẫu Val:   {val_count} ({val_count / total_samples * 100:.1f}%)")

    # Bước 4: Tạo thư mục val/ và di chuyển các cặp thuộc tập val
    print("\n--> Bước 4: Chuyển các cặp dữ liệu sang thư mục 'val/'...")
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_LBL_DIR, exist_ok=True)
    
    for stem in sorted(val_stems):
        img_name = img_stems[stem]
        lbl_name = lbl_stems[stem]
        
        src_img = os.path.join(TRAIN_IMG_DIR, img_name)
        src_lbl = os.path.join(TRAIN_LBL_DIR, lbl_name)
        
        dst_img = os.path.join(VAL_IMG_DIR, img_name)
        dst_lbl = os.path.join(VAL_LBL_DIR, lbl_name)
        
        shutil.move(src_img, dst_img)
        shutil.move(src_lbl, dst_lbl)
        print(f"    [VAL] {img_name} <-> {lbl_name}")

    # Bước 5: Cập nhật file data.yaml
    print("\n--> Bước 5: Cập nhật đường dẫn trong 'data.yaml'...")
    if os.path.exists(YAML_PATH):
        with open(YAML_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            if line.strip().startswith('train:'):
                new_lines.append("train: train/images\n")
            elif line.strip().startswith('val:'):
                new_lines.append("val: val/images\n")
            elif line.strip().startswith('test:'):
                new_lines.append("# test: test/images\n")
            else:
                new_lines.append(line)
                
        with open(YAML_PATH, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print("    => Đã cập nhật data.yaml trỏ tới 'train/images' và 'val/images'.")

    # Bước 6: Kiểm tra nghiệm thu sau phân chia
    final_train_imgs = len(os.listdir(TRAIN_IMG_DIR))
    final_train_lbls = len(os.listdir(TRAIN_LBL_DIR))
    final_val_imgs = len(os.listdir(VAL_IMG_DIR))
    final_val_lbls = len(os.listdir(VAL_LBL_DIR))

    print("\n" + "=" * 60)
    print("BÁO CÁO NGHIỆM THU (VERIFICATION REPORT)")
    print("=" * 60)
    print(f"Thư mục Train:")
    print(f"  - Số ảnh  (train/images/): {final_train_imgs}")
    print(f"  - Số nhãn (train/labels/): {final_train_lbls}")
    print(f"  - Tính toàn vẹn: {'HOÀN HẢO' if final_train_imgs == final_train_lbls else 'CẢNH BÁO LỆCH CẶP'}")
    print(f"Thư mục Val:")
    print(f"  - Số ảnh  (val/images/):   {final_val_imgs}")
    print(f"  - Số nhãn (val/labels/):   {final_val_lbls}")
    print(f"  - Tính toàn vẹn: {'HOÀN HẢO' if final_val_imgs == final_val_lbls else 'CẢNH BÁO LỆCH CẶP'}")
    print("-" * 60)
    print(f"Tổng số mẫu: {final_train_imgs + final_val_imgs}")
    print(f"Tỷ lệ thực tế: Train = {final_train_imgs / (final_train_imgs + final_val_imgs) * 100:.2f}% | Val = {final_val_imgs / (final_train_imgs + final_val_imgs) * 100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
