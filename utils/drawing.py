import cv2

def draw_legend(image, class_colors, class_labels):
    scale = 5
    legend_height = 60 * scale
    legend_width = 300 * scale
    square_size = 20 * scale
    font_scale = 0.6 * scale / 2.5
    font_thickness = int(2 * scale / 5)
    start_x = 20
    start_y = image.shape[0] - legend_height - 40

    cv2.rectangle(image, (start_x, start_y),
                  (start_x + legend_width, start_y + legend_height),
                  (255, 255, 255), -1)

    for i, class_id in enumerate(class_colors):
        color = class_colors[class_id]
        label = class_labels[class_id]
        y_offset = start_y + 40 + i * (square_size + 40)
        cv2.rectangle(image,
                      (start_x + 40, y_offset),
                      (start_x + 40 + square_size, y_offset + square_size),
                      color, -1)
        cv2.putText(image, label,
                    (start_x + 40 + square_size + 40, y_offset + square_size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

def draw_counts_in_corner(image, class_counts):
    font_scale = 1.5
    font_thickness = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 40
    margin = 20
    text_lines = [f"{label}: {count}" for label, count in class_counts.items()]
    max_width = max([cv2.getTextSize(t, font, font_scale, font_thickness)[0][0] for t in text_lines])
    total_height = len(text_lines) * line_height
    start_x = image.shape[1] - max_width - margin
    start_y = image.shape[0] - total_height - margin

    cv2.rectangle(image, (start_x - 10, start_y - 10),
                  (image.shape[1] - margin + 10, start_y + total_height),
                  (255, 255, 255), -1)
    for i, text in enumerate(text_lines):
        y = start_y + (i + 1) * line_height - 10
        cv2.putText(image, text, (start_x, y), font, font_scale, (0, 0, 0), font_thickness)
