import cv2
import numpy as np
#import torch

# To make detections and get required outputs
def YOLO_Detection(model, frame, conf=0.20):
    # Perform inference on an image
    results = model.predict(frame, conf=conf)
    # Extract bounding boxes, classes, names, and confidences
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()

    return boxes, classes, names, confidences

## Draw YOLOv8 detections function
def label_detection(frame, text, left, top, bottom, right, tbox_color=(30, 155, 50), fontFace=1, fontScale=0.8,
                    fontThickness=1):
    # Draw Bounding Box
    cv2.rectangle(frame, (int(left), int(top)), (int(bottom), int(right)), tbox_color, 2)
    # Draw and Label Text
    textSize = cv2.getTextSize(text, fontFace, fontScale, fontThickness)
    text_w = textSize[0][0]
    text_h = textSize[0][1]
    y_adjust = 10
    cv2.rectangle(frame, (int(left), int(top) + text_h + y_adjust), (int(left) + text_w + y_adjust, int(top)),
                  tbox_color, -1)
    cv2.putText(frame, text, (int(left) + 5, int(top) + 10), fontFace, fontScale, (255, 255, 255), fontThickness,
                cv2.LINE_AA)

# Updated drawPolygons function to avoid changing to green if detection inside
def drawPolygons(frame, points_list, detection_in_polygon=False, blink_state=False, alpha=0.2):
    # Color for blinking red effect
    polygon_color_inside = (0, 0, 255) if blink_state else (0, 0, 0)  # Toggle red/transparent for inside polygons
    polygon_color_outside = (30, 50, 250)  # Default color for outside polygons

    # Create a transparent overlay for the polygons
    overlay = frame.copy()

    for area in points_list:
        # Reshape the flat tuple to an array of shape (4, 1, 2)
        area_np = np.array(area, np.int32)

        # Draw filled polygons for detections inside with blinking effect
        if detection_in_polygon:
            # Blinking red for polygons with detections inside
            cv2.fillPoly(overlay, [area_np], polygon_color_inside)
        else:
            # Draw the polygon boundary (no fill) for outside detections
            cv2.polylines(overlay, [area_np], isClosed=True, color=polygon_color_outside, thickness=3)

    # Blend the overlay with the original frame
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return frame


def draw_background_rectangle(frame, rect_dimensions):
    """Draw a rectangle for the danger alert."""
    rect_width, rect_height = rect_dimensions
    frame_height, frame_width, _ = frame.shape
    top_left = (int((frame_width - rect_width) / 2), 10)
    bottom_right = (top_left[0] + rect_width, top_left[1] + rect_height)

    cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), thickness=-1)
    return top_left, rect_height


def overlay_icon(frame, icon, position):
    """Overlay the danger icon on the frame."""
    icon_x_offset, icon_y_offset = position
    frame[icon_y_offset:icon_y_offset + icon.shape[0],
    icon_x_offset:icon_x_offset + icon.shape[1]] = icon


# def setup_device():
#     """Check if CUDA is available and set the device."""
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
#     return device

