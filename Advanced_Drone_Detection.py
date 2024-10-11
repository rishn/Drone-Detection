import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from models.common import DetectMultiBackend  # Import DetectMultiBackend for loading the model

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
video_path = Path(input('Ensure video file is saved in save directory as code.\nEnter file name of video to be opened (with extension): '))
cap = cv2.VideoCapture(video_path.as_posix())

# Define class indices for YOLOv5 (vehicle and person)
vehicle_classes = ['car', 'motorbike', 'bus', 'truck', 'person']

# Get the original frame dimensions to set up the VideoWriter
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object to save the output video
output_path = Path('output.avi')
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

        # Initialize variables to track the best detection (highest confidence)
        best_detection = None
        highest_conf = 0

        # Loop through the outermost array, which contains multiple 2D arrays (each representing bounding boxes)
        for batch in drone_detections:
            # Loop through each result in the batch (each result is a 1D array representing one bounding box)
            for result in batch:
                x1, y1, width, height, conf, cls = result[:6]  # Extract center (x1, y1), width, height, confidence, and class

                # Ensure conf is a scalar float
                conf = float(conf)

                # Check if confidence is higher than the previous best and class is 'Drone'
                if conf > highest_conf and conf > 0.25 and int(cls) == 0:
                    highest_conf = conf
                    best_detection = (x1, y1, width, height, conf, cls)

        # If a best detection was found, scale and draw it
        if best_detection is not None:
            x1, y1, width, height, conf, cls = best_detection

            # Scale the bounding box dimensions back to the original frame size
            scale_x = orig_w / 640
            scale_y = orig_h / 640

            x1 = int(x1 * scale_x)  # Center x-coordinate
            y1 = int(y1 * scale_y)  # Center y-coordinate
            width = int(width * scale_x)  # Bounding box width
            height = int(height * scale_y)  # Bounding box height

            # Calculate the corners of the bounding box from the center
            x_min = int(x1 - width / 2)  # Top-left x-coordinate
            y_min = int(y1 - height / 2)  # Top-left y-coordinate
            x_max = int(x1 + width / 2)  # Bottom-right x-coordinate
            y_max = int(y1 + height / 2)  # Bottom-right y-coordinate

            # Draw the bounding box for drone
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            # Display the confidence score above the box
            text_conf = "{:.2f}%".format(highest_conf * 100)
            cv2.putText(frame, f"drone: {text_conf}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Process vehicle detections
    if vehicle_results[0] is not None and len(vehicle_results[0]) > 0:
        # Convert detections to numpy array for easier processing
        vehicle_detections = vehicle_results[0].detach().cpu().numpy()

        # Initialize variables to track the best detection (highest confidence)
        best_detection = None
        highest_conf = 0

        # Loop through the outermost array, which contains multiple 2D arrays (each representing bounding boxes)
        for batch in vehicle_detections:
            # Loop through each result in the batch (each result is a 1D array representing one bounding box)
            for result in batch:
                x1, y1, width, height, conf, cls = result[:6]  # Extract center (x1, y1), width, height, confidence, and class

                # Ensure conf is a scalar float
                conf = float(conf)

                # Check if confidence is higher than the previous best and class is 'Drone'
                if conf > highest_conf and conf > 0.75 and names[int(cls)] in vehicle_classes:
                    highest_conf = conf
                    best_detection = (x1, y1, width, height, conf, cls)

        # If a best detection was found, scale and draw it
        if best_detection is not None:
            x1, y1, width, height, conf, cls = best_detection  # Extract center (x1, y1), width, height, confidence, and class

            # Scale the bounding box dimensions back to the original frame size
            scale_x = orig_w / 640
            scale_y = orig_h / 640

            x1 = int(x1 * scale_x)  # Center x-coordinate
            y1 = int(y1 * scale_y)  # Center y-coordinate
            width = int(width * scale_x)  # Bounding box width
            height = int(height * scale_y)  # Bounding box height

            # Calculate the corners of the bounding box from the center
            x_min = int(x1 - width / 2)  # Top-left x-coordinate
            y_min = int(y1 - height / 2)  # Top-left y-coordinate
            x_max = int(x1 + width / 2)  # Bottom-right x-coordinate
            y_max = int(y1 + height / 2)  # Bottom-right y-coordinate

            # Draw the bounding box for vehicle
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Display the confidence score above the box
            text_conf = "{:.2f}%".format(highest_conf * 100)
            cv2.putText(frame, f"{names[int(cls)]}: {text_conf}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the processed frame to the output video
    out.write(frame)

    # Optionally display the output in a window (can be removed if not needed)
    cv2.imshow('Video', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()  # Save the output video
cv2.destroyAllWindows()
