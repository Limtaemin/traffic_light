#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from std_msgs.msg import String

# 신호등의 색상을 HSV(Hue, Saturation, Value) 색상 공간에서 범위로 정의합니다.
red_lower = np.array([157, 54, 156])
red_upper = np.array([255, 255, 255])
orange_lower = np.array([20, 38, 227])
orange_upper = np.array([76, 255, 255])
green_lower = np.array([78, 65, 221])
green_upper = np.array([129, 210, 255])

# 색상 감지 카운트 임계값
count_threshold = 70

def nothing(x):
    pass

cv2.namedWindow("Threshold Trackbars")
cv2.createTrackbar("Red Threshold", "Threshold Trackbars", 3, 1000, nothing)
cv2.createTrackbar("Orange Threshold", "Threshold Trackbars", 1300, 3000, nothing)
cv2.createTrackbar("Green Threshold", "Threshold Trackbars", 3, 3000, nothing)

def count_color_pixels(mask, x, y, window_radius):
    mask_circle = np.zeros_like(mask)
    cv2.circle(mask_circle, (x, y), window_radius, 1, thickness=-1)
    return cv2.countNonZero(mask & mask_circle)

# ROS 노드를 초기화합니다.
rospy.init_node('traffic_light_detector', anonymous=True)

# 각각의 색상에 대한 토픽을 설정합니다.
red_pub = rospy.Publisher('/red_detected', String, queue_size=10)
orange_pub = rospy.Publisher('/orange_detected', String, queue_size=10)
green_pub = rospy.Publisher('/green_detected', String, queue_size=10)
none_pub = rospy.Publisher('/none_detected', String, queue_size=10)

video_path = 'output.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

red_count = orange_count = green_count = none_count = 0

while not rospy.is_shutdown():
    ret, frame = cap.read()
    
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(frame, (640, 480))
    roi = frame[0:int(frame.shape[0] / 2), int(frame.shape[1] / 5):int(frame.shape[1]*4/5)]

    blurred_roi = cv2.GaussianBlur(roi, (7, 7), 2)
    hsv_roi = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv_roi, red_lower, red_upper)
    orange_mask = cv2.inRange(hsv_roi, orange_lower, orange_upper)
    green_mask = cv2.inRange(hsv_roi, green_lower, green_upper)

    red_threshold = cv2.getTrackbarPos("Red Threshold", "Threshold Trackbars")
    orange_threshold = cv2.getTrackbarPos("Orange Threshold", "Threshold Trackbars")
    green_threshold = cv2.getTrackbarPos("Green Threshold", "Threshold Trackbars")

    window_radius = 15
    step_size = 35

    max_red = max_orange = max_green = 0
    best_red_position = best_orange_position = best_green_position = None

    for y in range(window_radius, roi.shape[0] - window_radius, step_size):
        for x in range(window_radius, roi.shape[1] - window_radius, step_size):
            red_count_temp = count_color_pixels(red_mask, x, y, window_radius)
            orange_count_temp = count_color_pixels(orange_mask, x, y, window_radius)
            green_count_temp = count_color_pixels(green_mask, x, y, window_radius)

            if red_count_temp > max_red and red_count_temp > red_threshold:
                max_red = red_count_temp
                best_red_position = (x, y)
            if orange_count_temp > max_orange and orange_count_temp > orange_threshold:
                max_orange = orange_count_temp
                best_orange_position = (x, y)
            if green_count_temp > max_green and green_count_temp > green_threshold:
                max_green = green_count_temp
                best_green_position = (x, y)

    detected = False

    if best_red_position:
        cv2.circle(roi, best_red_position, window_radius, (0, 0, 255), 2)
        cv2.putText(roi, "Red", (best_red_position[0] - 50, best_red_position[1] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        red_count += 1
        detected = True
        if red_count >= count_threshold:
            red_pub.publish("Red detected")
            red_count = 0

    if best_orange_position:
        cv2.circle(roi, best_orange_position, window_radius, (0, 165, 255), 2)
        cv2.putText(roi, "Orange", (best_orange_position[0] - 50, best_orange_position[1] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        orange_count += 1
        detected = True
        if orange_count >= count_threshold:
            orange_pub.publish("Orange detected")
            orange_count = 0

    if best_green_position:
        cv2.circle(roi, best_green_position, window_radius, (0, 255, 0), 2)
        cv2.putText(roi, "Green", (best_green_position[0] - 50, best_green_position[1] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        green_count += 1
        detected = True
        if green_count >= count_threshold:
            green_pub.publish("Green detected")
            green_count = 0

    if not detected:
        none_count += 1
        if none_count >= count_threshold:
            none_pub.publish("None detected")
            none_count = 0

    cv2.imshow("Original", frame)
    cv2.imshow("ROI", roi)

    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()
