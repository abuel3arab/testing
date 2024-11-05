import cv2
import numpy as np
import serial
import time

# Initialize serial connection to Arduino
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)

# Camera configuration
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initial Region of Interest (ROI) settings
roi_top, roi_bottom = 300, 480
roi_left, roi_right = 0, 640

# HSV threshold for detecting white lane stripes
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 50, 255])

# Hough transform parameters
rho, theta, threshold = 2, np.pi / 180, 50
min_line_length, max_line_gap = 50, 80

# State variables for external object detection
object_detected = False  # True if green box (object) is detected
traffic_light_green = False  # True if green traffic light is detected

# Steering state
offset, steer_intensity = 0, 0
prev_steer_intensity = None  # To avoid redundant commands

# Weighted moving average for smoothing offset adjustments
alpha = 0.7

while True:
    ret, color_image = cap.read()
    if not ret:
        continue

    # Convert frame to HSV and create mask for white color
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

    # Limit mask to the region of interest
    roi_mask = np.zeros_like(white_mask)
    roi_mask[roi_top:roi_bottom, roi_left:roi_right] = white_mask[roi_top:roi_bottom, roi_left:roi_right]

    # Detect lines in the ROI mask
    lines = cv2.HoughLinesP(roi_mask, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    # Calculate steering offset based on detected lines
    if lines is not None:
        line_positions = [(line[0][0] + line[0][2]) / 2 for line in lines]
        mean_position = np.mean(line_positions)
        
        # Weighted moving average to smooth offset adjustments
        new_offset = (mean_position - 320) / 320
        offset = alpha * new_offset + (1 - alpha) * offset  # Smoothed offset
        
        # Calculate steer intensity based on smoothed offset
        steer_intensity = int(np.clip(offset * 100, -100, 100))

        # Dynamically adjust ROI to focus around the detected lane center
        roi_left = max(0, int(mean_position - 100))
        roi_right = min(640, int(mean_position + 100))
    else:
        # Reset ROI if no line is detected
        roi_left, roi_right = 0, 640
        steer_intensity = 0  # Keep it neutral if no lines are detected

    # Send commands to Arduino based on object detection state and steer intensity changes
    try:
        if object_detected:
            arduino.write("STOP\n".encode())  # Send STOP if an obstacle is detected
        elif traffic_light_green or not object_detected:
            # Send steer intensity if it's different from the previous value
            if steer_intensity != prev_steer_intensity:
                arduino.write(f"{steer_intensity}\n".encode())
                prev_steer_intensity = steer_intensity
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        arduino.close()
        arduino.open()
        continue

    # Display ROI mask and main camera feed for debugging
    cv2.imshow("ROI Mask", roi_mask)
    cv2.imshow("RC Car View", color_image)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
arduino.close()
