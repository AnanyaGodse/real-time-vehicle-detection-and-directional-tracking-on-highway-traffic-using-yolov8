### Project Title: Real-Time Vehicle Detection and Directional Tracking on Highway Traffic Using YOLOv8

#### Introduction:
 This project involves real-time vehicle detection and tracking on a highway using a
 pre-trained YOLO (You Only Look Once) object detection model. The system
 identifies and counts vehicles such as cars, buses, and trucks moving up and down
 in two lanes, providing valuable data for traffic monitoring and analysis.

 #### Objectives:
 ● To detect different types of vehicles (cars, buses, trucks) in a video stream.
 ● To trackthe direction of each vehicle (up or down) using reference lines.
 ● To countthevehicles moving in each direction, distinguishing between
 types.
 ● To display the FPS (Frames Per Second) to assess real-time processing
 performance.

 #### Code Structure and Components:
 The project consists of two files: main.py and tracker.py.
 ● YOLO Model Loading: The YOLOv8 pre-trained model is loaded for object
 detection, enabling accurate and fast detection of vehicles in each frame.
 ● Mouse Position Tracking: A utility to track the mouse position over the
 OpenCV window for testing and debugging.
 ● Vehicle Tracking: Custom tracking logic with unique IDs for each detected
 vehicle using a Tracker class, which updates vehicle positions across
 frames.
 ● Direction Thresholds: Two horizontal lines (green and red) serve as
 boundaries to track whether vehicles are moving up or down.
 ● FPS Calculation: The FPS calculation displays the frame processing speed on
 each frame to assess system performance in real-time.

 
