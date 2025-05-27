import os
import cv2
from ultralytics import YOLO

from utils.drawing import draw_legend, draw_counts_in_corner
from utils.tiling import tile_image
from utils.yolo_inference import visualize_with_seg_and_boxes

# === 設定只處理 input 資料夾、輸出到 output 資料夾 ===
model_path = r"model/best.pt"  # 修改為你的模型路徑
input_folder = "input"
output_folder = "output"
tile_size = 1024

os.makedirs(output_folder, exist_ok=True)
model = YOLO(model_path)

class_colors = {0: (0, 255, 0), 1: (0, 0, 255)}
class_labels = {0: "Surface spalling", 1: "Rebar exposure damage"}

for fname in os.listdir(input_folder):
    if fname.lower().endswith((".jpg", ".png")):
        image_path = os.path.join(input_folder, fname)
        image = cv2.imread(image_path)
        tiles, positions, H, W = tile_image(image, tile_size)
        overlay, class_counter = visualize_with_seg_and_boxes(image, tiles, positions, model, class_colors, tile_size)
        draw_legend(overlay, class_colors, class_labels)
        readable_counts = {class_labels[k]: v for k, v in class_counter.items()}
        draw_counts_in_corner(overlay, readable_counts)
        output_path = os.path.join(output_folder, f"{os.path.splitext(fname)[0]}.jpg")
        cv2.imwrite(output_path, overlay)

        image_name = fname
        total_detections = sum(class_counter.values())
        spalling_count = class_counter[0]
        rebar_count = class_counter[1]
        print(f"[INFO] image_name: {image_name}  detection_number: {total_detections}  Surface spalling: {spalling_count}  Rebar exposure damage: {rebar_count}")
