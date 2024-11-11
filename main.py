import os
import time
import cv2
from ultralytics import YOLO
from utilis import YOLO_Detection, label_detection, draw_background_rectangle, overlay_icon


def load_danger_icon(icon_path, height):
    """Load and resize the danger icon."""
    icon = cv2.imread(icon_path)
    aspect_ratio = icon.shape[1] / icon.shape[0]
    new_width = int(height * aspect_ratio)
    return cv2.resize(icon, (new_width, height))


def load_yolo_model(device):
    """Load the YOLO model and configure it."""
    model = YOLO("model_weights/best.pt")
    model.to(device)
    model.nms = 0.7
    print(f"Model classes: {model.names}")
    return model


def process_frame(model, frame, blink_interval, last_blink_time, blink_state):
    """Process a single frame to detect objects and apply labels."""

    current_time = time.time()

    boxes, classes, names, confidences = YOLO_Detection(model, frame, conf=0.5)

    for box, cls in zip(boxes, classes):
        if int(cls) == 1:
            label_detection(frame=frame, text=f"{names[int(cls)]}", tbox_color=(255, 144, 30),
                            left=box[0], top=box[1], bottom=box[2], right=box[3])
        else:
            label_detection(frame=frame, text=f"{names[int(cls)]}", tbox_color=(0, 0, 230),
                            left=box[0], top=box[1], bottom=box[2], right=box[3])

    # Overlay danger icon if blinking
    if len(boxes) > 0:  # Check if a person is detected (class == 1)
        if current_time - last_blink_time > blink_interval:
            blink_state = not blink_state  # Toggle blink state
            last_blink_time = current_time

    return last_blink_time, blink_state


def main(source, output_path="output_video.mp4"):

    model = load_yolo_model(device="cuda")

    danger_icon = load_danger_icon('icon.png', height=40)

    # Initialize VideoWriter if the source is a video file
    if os.path.isfile(source) and not source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        cap = cv2.VideoCapture(source)

        # Get video properties to set up the VideoWriter
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID' or others
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        blink_interval = 0.4
        last_blink_time = time.time()
        blink_state = False
        frame_tracker = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            last_blink_time, blink_state = process_frame(model, frame, blink_interval, last_blink_time, blink_state)

            # Overlay danger icon if blinking
            if blink_state:
                frame_tracker += 1
                if frame_tracker > 0:
                    # Draw alert rectangle when person is detected
                    rect_position, rect_height = draw_background_rectangle(frame, (165, 50))

                    text = f"DANGER" if blink_state else ""

                    overlay_icon(frame, danger_icon, (rect_position[0] + 5, rect_position[1] + 8))
                    font = cv2.FONT_HERSHEY_TRIPLEX
                    text_x = rect_position[0] + danger_icon.shape[1] + 10
                    text_y = rect_position[1] + 35
                    cv2.putText(frame, text, (text_x, text_y), font, 0.8, (0, 0, 250), thickness=2)
            elif not blink_state and frame_tracker > 0:
                frame_tracker -= 1
            else:
                pass


            out.write(frame)  # Write the processed frame to output video

            # Display the frame (press 'q' to quit early)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()  # Release VideoWriter object after the loop

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example: Change this to the desired image/video source or camera index
    main(source="4.mp4", output_path="output_video/output_video.mp4")  # Use an image path (e.g., "image.jpg") or video path
