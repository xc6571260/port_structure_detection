import cv2
import numpy as np

def visualize_with_seg_and_boxes(image, tiles, positions, model, class_colors, tile_size):
    overlay = image.copy()
    class_counter = {0: 0, 1: 0}
    for tile, (x_offset, y_offset) in zip(tiles, positions):
        results = model(tile, verbose=False)[0]
        if results.masks is None or results.boxes is None:
            continue
        masks = results.masks.data.cpu().numpy().astype(np.uint8)
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for mask, cls_id in zip(masks, classes):
            class_counter[cls_id] += 1
            mask_h, mask_w = mask.shape
            target_h = min(tile_size, image.shape[0] - y_offset)
            target_w = min(tile_size, image.shape[1] - x_offset)
            mask = mask[:target_h, :target_w]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 50:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                abs_x, abs_y = x + x_offset, y + y_offset
                color = class_colors[cls_id]
                cv2.rectangle(overlay, (abs_x, abs_y), (abs_x + w, abs_y + h), color, 2)
                mask_bool = mask.astype(bool)
                region = overlay[y_offset:y_offset+target_h, x_offset:x_offset+target_w]
                region[mask_bool] = (0.7 * region[mask_bool] + 0.3 * np.array(color)).astype(np.uint8)
                overlay[y_offset:y_offset+target_h, x_offset:x_offset+target_w] = region
    return overlay, class_counter
