import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from models.common import DetectMultiBackend  # Import DetectMultiBackend for loading the model

# NMS function to filter overlapping boxes
def non_max_suppression(boxes, confidences, iou_threshold=0.5):
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, score_threshold=0.25, nms_threshold=iou_threshold
    )
    return indices.flatten() if len(indices) > 0 else []

# Path to YOLOv5 repo and models
yolo_repo_path = Path('./yolov5')  # Path to the YOLOv5 repo
best_model_path = Path('best.pt')  # Path to your local model for drone detection
yolov5_model_path = Path('yolov5s.pt')  # Use the YOLOv5 model for person and vehicle detection

# Load the drone detection model (best.pt)
try:
    drone_model = DetectMultiBackend(best_model_path, device='cpu')  # Adjust 'cpu' or 'cuda' based on your system
    print("Drone model loaded successfully.")
except Exception as e:
    print("Error loading drone model:", e)

# Load the YOLOv5 model (yolov5.pt)
try:
    vehicle_model = DetectMultiBackend(yolov5_model_path, device='cpu')  # Adjust 'cpu' or 'cuda' based on your system
    print("Vehicle model loaded successfully.")
except Exception as e:
    print("Error loading vehicle model:", e)

# Set both models to evaluation mode
drone_model.eval()
vehicle_model.eval()

# Load class names from YOLOv5 model
names = vehicle_model.names  # This assumes the model has a 'names' attribute containing class names

# Set video source (webcam or video file) and convert path to string
video_path = Path(input('\nEnsure video file is saved in save directory as code.\nEnter file name of video to be opened (with extension): '))
cap = cv2.VideoCapture(video_path.as_posix())

# Define class indices for YOLOv5 (vehicle and person)
vehicle_classes = ['car', 'person']

# Get the original frame dimensions to set up the VideoWriter
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object to save the output video
output_path = Path(input("\nEnter output video file name (with extension): "))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi
out = cv2.VideoWriter(output_path.as_posix(), fourcc, fps, (frame_width, frame_height))

# Loop over frames
while True:
    # Read frame from video source
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Get the original frame dimensions
    orig_h, orig_w = frame.shape[:2]

    # Resize the frame for both models
    img_resized_drone = cv2.resize(frame, (640, 640))
    img_resized_vehicle = cv2.resize(frame, (640, 640))

    # Convert the frame to RGB format and then to PIL Image for drone model
    img_drone = Image.fromarray(img_resized_drone[..., ::-1])
    img_vehicle = Image.fromarray(img_resized_vehicle[..., ::-1])

    # Convert the images to PyTorch tensors (as the models expect tensors)
    img_tensor_drone = torch.from_numpy(np.array(img_drone)).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1] range
    img_tensor_vehicle = torch.from_numpy(np.array(img_vehicle)).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1] range

    img_tensor_drone = img_tensor_drone.unsqueeze(0)  # Add batch dimension
    img_tensor_vehicle = img_tensor_vehicle.unsqueeze(0)  # Add batch dimension

    # Run inference on the frame using the drone model
    drone_results = drone_model(img_tensor_drone)  # Inference results for drone model

    # Run inference on the frame using the vehicle model
    vehicle_results = vehicle_model(img_tensor_vehicle)  # Inference results for vehicle model

    # Process drone detections
    if drone_results[0] is not None and len(drone_results[0]) > 0:
        # Convert detections to numpy array for easier processing
        drone_detections = drone_results[0].detach().cpu().numpy()

        # Prepare lists for NMS
        drone_boxes = []
        drone_confidences = []

        # Loop through the outermost array, which contains multiple 2D arrays (each representing bounding boxes)
        for batch in drone_detections:
            # Loop through each result in the batch (each result is a 1D array representing one bounding box)
            for result in batch:
                x1, y1, width, height, conf, cls = result[:6]  # Extract center (x1, y1), width, height, confidence, and class

                # Ensure conf is a scalar float
                conf = float(conf)

                # Prepare bounding box for NMS
                if conf > 0.25:  # Confidence threshold
                    x_min = int((x1 - width / 2) * orig_w / 640)
                    y_min = int((y1 - height / 2) * orig_h / 640)
                    x_max = int((x1 + width / 2) * orig_w / 640)
                    y_max = int((y1 + height / 2) * orig_h / 640)

                    drone_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
                    drone_confidences.append(conf)

        # Apply Non-Maximum Suppression
        indices = non_max_suppression(drone_boxes, drone_confidences)

        # Draw filtered drone detections
        for i in indices:
            x_min, y_min, w, h = drone_boxes[i]
            conf = drone_confidences[i]

            # Draw bounding box for drone
            cv2.rectangle(frame, (x_min, y_min), (x_min + w, y_min + h), (0, 0, 255), 2)
            cv2.putText(frame, f"drone: {(conf * 100):.2f}%", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Process vehicle/person detections
    if vehicle_results[0] is not None and len(vehicle_results[0]) > 0:
        # Convert detections to numpy array for easier processing
        vehicle_detections = vehicle_results[0].detach().cpu().numpy()

        # Prepare lists for NMS
        vehicle_boxes = []
        vehicle_confidences = []
        vehicle_classes_detected = []

        # Loop through the outermost array, which contains multiple 2D arrays (each representing bounding boxes)
        for batch in vehicle_detections:
            # Loop through each result in the batch (each result is a 1D array representing one bounding box)
            for result in batch:
                x1, y1, width, height, conf, *cls_probs = result  # Extract center (x1, y1), width, height, confidence, and class probabilities

                # Ensure conf is a scalar float
                conf = float(conf)

                # Find the class index with the highest probability
                cls_idx = np.argmax(cls_probs)  # Get the index of the highest class probability
                class_name = names[cls_idx]

                # Prepare bounding box for NMS
                if conf > 0.75 and class_name in vehicle_classes:  # Confidence threshold and class filter
                    x_min = int((x1 - width / 2) * orig_w / 640)
                    y_min = int((y1 - height / 2) * orig_h / 640)
                    x_max = int((x1 + width / 2) * orig_w / 640)
                    y_max = int((y1 + height / 2) * orig_h / 640)

                    vehicle_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
                    vehicle_confidences.append(conf)
                    vehicle_classes_detected.append(class_name)

        # Apply Non-Maximum Suppression
        indices = non_max_suppression(vehicle_boxes, vehicle_confidences)

        # Draw filtered vehicle/person detections
        for i in indices:
            x_min, y_min, w, h = vehicle_boxes[i]
            conf = vehicle_confidences[i]
            class_name = vehicle_classes_detected[i]

            # Choose color based on class
            color = (200, 200, 0) if class_name == 'car' else (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_min + w, y_min + h), color, 2)
            cv2.putText(frame, f"{class_name}: {(conf * 100):.2f}%", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Write the frame with the drawn bounding boxes
    out.write(frame)

    # Show the output frame
    cv2.imshow('Drone and Vehicle Detection', frame)

    # Stop the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
