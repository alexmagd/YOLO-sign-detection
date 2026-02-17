import os
import cv2
from ultralytics import YOLO

# Define the class-to-name mapping dictionary
class_dict = {
    0: "Regulation",
    1: "Warning",
    2: "Mandatory Direction",
    3: "Priority Give/Take"
}

# Load the YOLO model
model_path = r'Thesis\New_YOLO\detect\train2\weights\best.pt'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = YOLO(model_path)

# Detection confidence threshold
threshold = 0.7

# Desired output video dimensions
width, height = 640, 480

def preprocess_frame(frame):
    """
    Resize the frame to the desired dimensions.
    """
    return cv2.resize(frame, (width, height))

# Input video file path
input_video_path = r"A:\Codes\Thesis\traffic_signs2.mp4"
if not os.path.exists(input_video_path):
    raise FileNotFoundError(f"Input video not found: {input_video_path}")

cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    raise Exception("Could not open video capture.")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video setup
output_video_path = "output_video_yolo2.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or failed to grab frame.")
        break

    # Preprocess the frame
    frame = preprocess_frame(frame)

    # Perform object detection
    results = model(frame)[0]  # YOLO results for the current frame

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            # Get class name from the dictionary
            class_name = class_dict.get(int(class_id), "Unknown")
            label = f"{class_name} {score:.2f}"

            # Draw the bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Draw the label
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the processed frame to the output video
    out.write(frame)

    # Optionally display the frame
    cv2.imshow('YOLO Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved to {output_video_path}")
