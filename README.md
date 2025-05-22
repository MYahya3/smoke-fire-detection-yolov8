# Real-time Smoke & Fire Detection

This project is a computer vision application built using **YOLOv8** to enable real-time smoke and fire detection in video streams. It is designed to identify potential hazards and provide visual alerts, making it suitable for early warning systems in surveillance, industrial monitoring, and safety automation environments.

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/05319bc2-ecdc-4c32-a56a-97f49e14b614" alt="Smoke and Fire Detection Preview" width="600"/>
</p>

---

## Features

- **Real-time object detection** for smoke and fire using YOLOv8.
- **Visual alerts** with blinking danger indicators and icons when threats are detected.
- **Video processing** support with automatic annotation and frame-by-frame analysis.
- **Custom icon overlays** to highlight hazards in the video.
- **Support for custom-trained models** using YOLOv8 for specific detection needs.

---

## Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt

## Project Structure
```bash
├── model_weights/
│   └── best.pt              # YOLOv8 trained weights
├── utils/
│   └── utilis.py            # Custom utility functions (detection, overlays, etc.)
├── icon.png                 # Danger icon displayed on detection
├── 4.mp4                    # Input video file
├── output_video/
│   └── output_video.mp4     # Output video with annotations
├── detector.py              # Main detection script
└── README.md                # This file
```

## How to Run
Run the detection script:
```bash
python detector.py

Modify the main() call to change the input video or output path:
main(source="your_input.mp4", output_path="your_output.mp4")

```

## Model Training

This project assumes you already have a YOLOv8 model trained to detect:
```
1. Smoke
2. Fire
```

### To Achieve Better Results
Detection Threshold: Adjust confidence thresholds via YOLO_Detection(conf=0.5)

--> Blink Speed: Modify blink_interval to change how fast the danger indicator blinks.

--> Icon Design: Replace icon.png with your own PNG icon


## What can you do?

1. You can train a custom model with custom classes
2. The detection includes blinking danger alerts to grab attention during high-risk scenarios.
3. Frame-by-frame processing is optimized for both performance and usability.
4. For real-time use with cameras, extend main() to accept webcam input.
 

### **License**
This project is open for educational and research use. For commercial applications, please ensure compliance with the respective licenses of YOLO and the datasets used for training.

