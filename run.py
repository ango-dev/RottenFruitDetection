import cv2
import pandas as pd
from ultralytics import YOLO
import os
import time
import shutil

# --- Configuration ---
# Choose your pre-trained model (e.g., 'yolov8n.pt' for nano, 'yolov8s.pt' for small)
MODEL_NAME = 'rotten-fruit-detection.pt'
# 0 is usually the default webcam. Change if you have multiple cameras.
WEBCAM_SOURCE = 0
# Directory to save the cropped images
SAVE_DIR = 'frame_images'
# Confidence threshold filter
CONFIDENCE_THRESHOLD = 0.7
# Change the model call to leverage a GPU if available (e.g., device='0' for the first GPU)
# If you don't have a GPU, stick to device='cpu' or omit the device argument
DEVICE_TO_USE = '0' # Or 'cpu'
IMAGE_SIZE = 640
# Initializing the frame counter for unique image filenames
frame_count = 0 
# Initializing the detection counter for unique DataFrame indices
detection_id = 0


os.makedirs(SAVE_DIR, exist_ok=True)

try:
    model = YOLO(MODEL_NAME, task='detect')
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Please ensure you have installed ultralytics and the model file is accessible.")
    exit()

# Open the webcam
cap = cv2.VideoCapture(WEBCAM_SOURCE)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

columns = ['frame', 'detection_id', 'x1', 'y1', 'x2', 'y2', 'confidence', 'class_id', 'class_name', 'crop_filename']
all_detections_df = pd.DataFrame(columns=columns)

print("Starting webcam object detection. Press 'q' to quit.")

# This loop will iterate over every frame
# press q to exit
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Webcam frame is empty or end of stream.")
        break

    results = model(frame, conf=CONFIDENCE_THRESHOLD, imgsz=IMAGE_SIZE, stream=False, device=DEVICE_TO_USE, verbose=False)
    current_frame_detections = []
    
    if results and results[0].boxes is not None:
        boxes = results[0].boxes.cpu().numpy() 
        
        for i, box in enumerate(boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            confidence = float(boxes.conf[i])
            class_id = int(boxes.cls[i])
            class_name = model.names[class_id]
            
            try:
                cropped_img = frame[y1:y2, x1:x2]
                
                timestamp = int(time.time() * 1000)
                crop_filename = f'frame_{frame_count}_det_{detection_id}_{class_name}_{timestamp}.jpg'
                crop_path = os.path.join(SAVE_DIR, crop_filename)
                
                cv2.imwrite(crop_path, cropped_img)
                
            except Exception as crop_error:
                print(f"Error cropping/saving image: {crop_error}")
                crop_filename = 'ERROR'

            detection_data = {
                'frame': frame_count,
                'detection_id': detection_id,
                'x1': x1, 
                'y1': y1, 
                'x2': x2, 
                'y2': y2, 
                'confidence': confidence,
                'class_id': class_id,
                'class_name': class_name,
                'crop_filename': crop_filename
            }
            current_frame_detections.append(detection_data)
            detection_id += 1 # Increment detection ID for the next detection

    if current_frame_detections:
        frame_df = pd.DataFrame(current_frame_detections, columns=columns)
        all_detections_df = pd.concat([all_detections_df, frame_df], ignore_index=True)

    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Object Detection", annotated_frame)
    
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

shutil.rmtree('frame_images')
os.makedirs('frame_images')
cap.release()
cv2.destroyAllWindows()

# csv_filename = 'object_detections_log.csv'
# all_detections_df.to_csv(csv_filename, index=False)

print("\n--- Script Finished ---")
print(f"Total frames processed: {frame_count}")
print(f"Total detections logged: {len(all_detections_df)}")
# print(f"Bounding box data saved to: {os.path.abspath(csv_filename)}")
# print(f"Cropped images saved to folder: {os.path.abspath(SAVE_DIR)}")