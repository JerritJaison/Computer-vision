# AI Crowd Monitoring

A real-time AI-powered crowd monitoring system using YOLO, OpenCV, and PyTorch for people detection, tracking, and heatmap generation.

---

## Features

- Real-time people detection using YOLO
- Centroid tracking for object movement
- Crowd density heatmap generation with seaborn
- OpenCV-powered video frame processing
- YOLO model loading via PyTorch 

---

## Project Structure
AI-Crowd-Monitoring/ │ ├── main.py # Main application script ├── yolo.pt # YOLO trained weights (not included) ├── heatmap.py # Heatmap creation ├── tracker.py # Object tracking logic ├── requirements.txt # Project dependencies ├── README.md # Project documentation └── venv/ # Virtual environment (excluded)

## Setup

1. Clone the repository
   git clone https://github.com/JerritJaison/Realtime_AI-Crowd-Monitoring_and_weight_detection_using_YOLO.git
   cd AI-Crowd-Monitoring
2. Create and activate a virtual environment
    python -m venv venv
    venv\Scripts\activate   # On Windows
    # or
    source venv/bin/activate  # On macOS/Linux
3. Install Dependencies
   pip install -r requirements.txt

## Notes

Ensure your webcam is connected and working if using live input.
You can modify detection confidence thresholds and source options in main.py
## Sample Output

![Screenshot of AI Crowd Monitoring in action](screenshot.png)
