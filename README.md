import cv2
import numpy as np
import time

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Pre-trained model
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels (COCO dataset)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Start video capture (0 = webcam, or path to video)
cap = cv2.VideoCapture("traffic_video.mp4")

# Track previous positions of cars
vehicle_positions = {}

def detect_collision(positions, threshold=50):
    for id1, (x1, y1) in positions.items():
        for id2, (x2, y2) in positions.items():
            if id1 != id2:
                dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if dist < threshold:
                    print(f"[ALERT] Possible collision detected between {id1} and {id2}")
                    return True
    return False

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width, _ = frame.shape
    frame_id += 1

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    current_positions = {}
    object_id = 0

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Focus on vehicles only
            if confidence > 0.5 and classes[class_id] in ['car', 'truck', 'bus', 'motorbike']:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Draw bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Store vehicle position
                current_positions[object_id] = (center_x, center_y)
                object_id += 1

    # Check for possible collisions
    if detect_collision(current_positions):
        cv2.putText(frame, "Collision Risk Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("AI Traffic Monitoring", frame)

    if cv2.waitKey(1) == 27:  # Press 'Esc' to quit
        break

cap.release()
cv2.destroyAllWindows()
