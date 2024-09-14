import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from ultralytics import YOLO

# Initializing yolo model
model = YOLO('yolov8n.pt')

# Opening live stream 
stream_url='https://5a78c55e99e82.streamlock.net/ButtimKavsagi/smil:ButtimKavsagi/chunklist_w775435106_b700000_tkd293emF0b2tlbmVuZHRpbWU9MTcyNTU0MzMxOSZ3b3d6YXRva2VuaGFzaD1SSHA4QXp5alhnRXJ6VmdscU1XWENsYXM2N2ZKTjFXZTVkM281NndJenlvSUdJT09VekZrM1lubVNyRjQ5QjcxJndvd3phdG9rZW5zdGFydHRpbWU9MTcyNTU0MTUxOQ==.m3u8'
cap = cv2.VideoCapture(stream_url)

# To keep track of the frame numbers for plotting
frame_count = 0

# Storing traffic data for analysis - with 100 frames, when it exceeds 100 it will remove the oldest one
traffic_data_right_lane = deque(maxlen=100)
traffic_data_left_lane = deque(maxlen=100)

# Defining target classes for detection (Car, Motorcycle, Bus, Truck)
target_classes = [2, 3, 5, 7]

# Initializing a figure and axis for the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 1.2)
ax1.set_xlabel("Frame")
ax1.set_ylabel("Traffic Density")
ax1.set_title("Traffic Density Right Lane")

ax2.set_xlim(0, 100)
ax2.set_ylim(0, 1.2)
ax2.set_xlabel("Frame")
ax2.set_ylabel("Traffic Density")
ax2.set_title("Traffic Density Left Lane")

plt.subplots_adjust(hspace=0.5)
plt.ion()

# Function to update and display the plot
def update_plot():
    if len(traffic_data_right_lane) > 0 and len(traffic_data_left_lane) > 0:
        frames_right, densities_right = zip(*traffic_data_right_lane)
        frames_left, densities_left = zip(*traffic_data_left_lane)
        
        ax1.clear()
        ax1.plot(frames_right, densities_right, label="Right Lane Density", color='blue')
        ax1.set_xlim(max(0, frame_count - 100), frame_count + 10)
        ax1.set_ylim(0, 1.2)
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Traffic Density")
        ax1.set_title("Traffic Density Right Lane")
        ax1.legend()
        
        ax2.clear()
        ax2.plot(frames_left, densities_left, label="Left Lane Density", color='green')
        ax2.set_xlim(max(0, frame_count - 100), frame_count + 10)
        ax2.set_ylim(0, 1.2)
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Traffic Density")
        ax2.set_title("Traffic Density Left Lane")
        ax2.legend()
        
        plt.draw()
        plt.pause(0.01)

# Adjust brightness
def adjust_brightness_contrast(image, alpha=1.5, beta=-50):
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

# Sharpen image
def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

# Calculate traffic density
def calculate_density(roi, objects):
    occupied_area = 0
    for obj in objects:
        x1, y1, x2, y2 = obj
        if x1 < roi[2] and x2 > roi[0] and y1 < roi[3] and y2 > roi[1]:
            occupied_area += (x2 - x1) * (y2 - y1)
    
    roi_area = (roi[2] - roi[0]) * (roi[3] - roi[1])
    density = occupied_area / roi_area
    return density

# Processing the video frames
if not cap.isOpened():
    print("Error: Video could not be opened.")
else:
    while True:
        # Reading a frame from the stream
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Resizing the frame for display
        frame_resized = cv2.resize(frame, (1280, 720))  

        # Adjust brightness and contrast and sharpen the frame
        frame_resized = adjust_brightness_contrast(frame_resized)
        frame_resized = sharpen_image(frame_resized)

        # Define large rectangular ROIs for the left and right lanes with doubled height
# Define the initial height

        original_height_left = 700 - 350  # For the left lane
        original_height_right = 700 - 350  # For the right lane

# Double the height by adding the original height to y2

        roi_left_lane = (50, 350, 600, 700 + original_height_left)
        roi_right_lane = (680, 350, 1230, 700 + original_height_right)


        # Detect objects using YOLO
        results = model(frame_resized)
        detected_objects_right_lane = []
        detected_objects_left_lane = []

        # Loop through detection results
        for result in results:
            boxes = result.boxes.xyxy
            classes = result.boxes.cls

            for i in range(len(boxes)):
                class_id = int(classes[i])

                if class_id in target_classes:
                    x1, y1, x2, y2 = boxes[i]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    if x1 < roi_left_lane[2] and x2 > roi_left_lane[0] and y1 < roi_left_lane[3] and y2 > roi_left_lane[1]:
                        detected_objects_left_lane.append((x1, y1, x2, y2))
                    
                    if x1 < roi_right_lane[2] and x2 > roi_right_lane[0] and y1 < roi_right_lane[3] and y2 > roi_right_lane[1]:
                        detected_objects_right_lane.append((x1, y1, x2, y2))

                    # Draw rectangle around detected object
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Calculate the number of cars in each ROI
        num_cars_left_lane = len(detected_objects_left_lane)
        num_cars_right_lane = len(detected_objects_right_lane)

        # Draw ROIs on the frame
        cv2.rectangle(frame_resized, (roi_left_lane[0], roi_left_lane[1]), (roi_left_lane[2], roi_left_lane[3]), (0, 0, 255), 2)
        cv2.rectangle(frame_resized, (roi_right_lane[0], roi_right_lane[1]), (roi_right_lane[2], roi_right_lane[3]), (0, 0, 255), 2)
        cv2.putText(frame_resized, f"Number of Cars Left Lane: {num_cars_left_lane}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame_resized, f"Number of Cars Right Lane: {num_cars_right_lane}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Display the frame
        cv2.imshow("Frame", frame_resized)

        # Store traffic density data for plotting
        density_left_lane = calculate_density(roi_left_lane, detected_objects_left_lane)
        density_right_lane = calculate_density(roi_right_lane, detected_objects_right_lane)
        traffic_data_left_lane.append((frame_count, density_left_lane))
        traffic_data_right_lane.append((frame_count, density_right_lane))

        # Update plot
        update_plot()

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release video capture object
cap.release()
cv2.destroyAllWindows()
