# Long Range Object Detection

This project aims to detect objects such as humans, vehicles (cars), and drones from significant distances with high accuracy, even in challenging environmental conditions. The Deep Learning models are designed for deployment on edge devices to handle real-time detection tasks efficiently with suffciently low computational power requirements.

## Problem Statement

**Long Range Object Detection**:  
- Detect objects (humans, vehicles, drones) from long distances with high precision.  
- The model needs to operate effectively on various devices, with a focus on handling environmental challenges such as:
  - Weather conditions (rain, fog, etc.)
  - Low visibility (nighttime, low light)
  - Motion (object or camera movement)

### Key Features:
1. **Object Detection**:
   - Detect and classify drones, humans, and vehicles in video streams or images.
   - Support for real-time video analysis.
    
2. **Open Source Resources**:
   - Utilizes YOLO (You Only Look Once), a popular open-source object detection algorithm known for its speed and accuracy. This allows for easy customization and improvement of the detection models.
   - The project is built using widely adopted open-source libraries such as OpenCV, PyTorch, and NumPy, ensuring accessibility and community support for further development.

3. **Low Computational Power**:
   - Designed to run efficiently on devices with limited computational resources, making it suitable for deployment on edge devices like NVIDIA Jetson, Raspberry Pi, or similar platforms.
   - The models are optimized for CPU execution, enabling performance even in low-power environments, making it feasible to analyze video streams without requiring high-end hardware.

---

## Demo

https://github.com/user-attachments/assets/2fdc9f96-2d09-4f53-b764-7baded555aa5

---

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Model](#model)

---

## Installation

### Prerequisites:

- Python 3.8 or higher
- PyTorch
- OpenCV

### Steps to Install:

1. Clone the repository.
  ```
    git clone https://github.com/rishn/Drone-Detection.git 
  ```

2. Navigate to the project directory:
  ```
   cd Drone-Detection
  ```

3. Install the required packages:
  ```
   pip install -r requirements.txt
  ```
4. Ensure the pre-trained YOLOv5 and custom drone models (`best.pt` and `yolov5s.pt`) are available in the project directory.


## Usage

1. Place your input video file in the root directory.
   
2. Run the detection script.
  ```
   python3 Advanced_Drone_Detection.py
  ```

3. The output video with detected objects will be saved to the specified output file.

### Video Input:
- Ensure your input video is saved in the same directory as the code.
- The script prompts for the input video file and output file names.

## Model

### YOLOv5 and Custom Drone Model

- The project utilizes two object detection models:
  1. **YOLOv5**: Pre-trained on the COCO dataset to detect vehicles and persons.
  2. **Custom Drone Model**: A specialized YOLOv5 model fine-tuned to detect drones (`best.pt`).

### How It Works:
1. Both models process each video frame to detect objects.
2. Results are filtered using **Non-Maximum Suppression** (NMS) to reduce overlapping bounding boxes.
3. Bounding boxes for vehicles, humans, and drones are drawn on the video output with corresponding confidence scores.

---
