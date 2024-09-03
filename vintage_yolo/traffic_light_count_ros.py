#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# 신호등의 색상을 HSV(Hue, Saturation, Value) 색상 공간에서 범위로 정의합니다.
red_lower = np.array([157, 54, 156])
red_upper = np.array([255, 255, 255])
orange_lower = np.array([20, 38, 227])
orange_upper = np.array([76, 255, 255])
green_lower = np.array([78, 65, 221])
green_upper = np.array([129, 210, 255])

# 색상 감지 카운트 임계값
count_threshold = 70

# CvBridge 객체 생성
bridge = CvBridge()

# 트랙바의 값을 변경할 때 호출되는 콜백 함수입니다. 여기서는 아무것도 하지 않습니다.
def nothing(x):
    pass

cv2.namedWindow("Threshold Trackbars")
cv2.createTrackbar("Red Threshold", "Threshold Trackbars", 100, 1000, nothing)
cv2.createTrackbar("Orange Threshold", "Threshold Trackbars", 100, 3000, nothing)
cv2.createTrackbar("Green Threshold", "Threshold Trackbars", 100, 3000, nothing)

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

# 색상 감지 카운트 초기화
red_count = orange_count = green_count = none_count = 0

def image_callback(msg):
    global red_count, orange_count, green_count, none_count

    try:
        # ROS 이미지 메시지를 OpenCV 이미지로 변환합니다.
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # 비디오 프레임 크기를 640x480으로 제한합니다.
        frame = cv2.resize(cv_image, (640, 480))

        # 이미지 상단 절반 영역을 ROI(Region of Interest)로 설정합니다.
        roi = frame[0:int(frame.shape[0] / 2), int(frame.shape[1] / 5):int(frame.shape[1]*4/5)]

        # 노이즈 제거를 위해 Gaussian Blur를 적용합니다.
        blurred_roi = cv2.GaussianBlur(roi, (7, 7), 2)

        # 이미지를 HSV 색상 공간으로 변환합니다.
        hsv_roi = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2HSV)

        # 각 색상 범위에 따라 마스크를 생성합니다.
        red_mask = cv2.inRange(hsv_roi, red_lower, red_upper)
        orange_mask = cv2.inRange(hsv_roi, orange_lower, orange_upper)
        green_mask = cv2.inRange(hsv_roi, green_lower, green_upper)

        # 트랙바에서 임계값 값을 가져옵니다.
        red_threshold = cv2.getTrackbarPos("Red Threshold", "Threshold Trackbars")
        orange_threshold = cv2.getTrackbarPos("Orange Threshold", "Threshold Trackbars")
        green_threshold = cv2.getTrackbarPos("Green Threshold", "Threshold Trackbars")

        # 윈도우 크기와 이동 스텝을 설정합니다.
        window_radius = 15
        step_size = 35

        # 각 색상에 대해 가장 큰 픽셀 수와 해당 위치를 저장할 변수를 초기화합니다.
        max_red = max_orange = max_green = 0
        best_red_position = best_orange_position = best_green_position = None

        # ROI 영역 내에서 일정 간격으로 윈도우를 이동하며 색상 검출을 수행합니다.
        for y in range(window_radius, roi.shape[0] - window_radius, step_size):
            for x in range(window_radius, roi.shape[1] - window_radius, step_size):
                # 각 위치에서 색상 픽셀 수를 계산합니다.
                red_count_temp = count_color_pixels(red_mask, x, y, window_radius)
                orange_count_temp = count_color_pixels(orange_mask, x, y, window_radius)
                green_count_temp = count_color_pixels(green_mask, x, y, window_radius)

                # 각 색상에 대해 임계값을 초과하는 최대 픽셀 수와 위치를 저장합니다.
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
            rospy.signal_shutdown('User pressed Esc')
            cap.release()
            cv2.destroyAllWindows()

    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")

# 이미지 토픽을 구독하고 콜백 함수를 지정합니다.
image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)

# ROS 루프를 시작합니다.
rospy.spin()

# 프로그램 종료 시 모든 창 닫기
cv2.destroyAllWindows()
