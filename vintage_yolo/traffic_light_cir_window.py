#!/usr/bin/env python3  # 파이썬 스크립트를 실행할 때 사용하는 기본 인터프리터 경로를 지정합니다.

import rospy  # ROS (Robot Operating System)와 통신하기 위해 rospy 모듈을 불러옵니다.
from sensor_msgs.msg import Image  # 이미지 메시지를 사용하기 위해 ROS 메시지 타입을 불러옵니다.
from cv_bridge import CvBridge, CvBridgeError  # ROS 이미지를 OpenCV 이미지로 변환하기 위해 CvBridge를 사용합니다.
import cv2  # OpenCV 라이브러리, 이미지 및 비디오 처리에 사용됩니다.
import numpy as np  # NumPy 라이브러리, 배열 계산 및 이미지 처리에 사용됩니다.

# ROS 노드를 초기화합니다. 노드 이름은 'circle_detector'이며 익명으로 설정되어 중복 노드가 생기지 않습니다.
rospy.init_node('circle_detector', anonymous=True)

# CvBridge 객체를 생성합니다. ROS 이미지 메시지를 OpenCV 이미지로 변환하는 데 사용됩니다.
bridge = CvBridge()

# 신호등의 색상을 HSV(Hue, Saturation, Value) 색상 공간에서 범위로 정의합니다.
red_lower = np.array([173, 34, 165])  # 빨간색의 HSV 범위 하한값
red_upper = np.array([197, 255, 255])  # 빨간색의 HSV 범위 상한값
orange_lower = np.array([21, 79, 136])  # 주황색의 HSV 범위 하한값
orange_upper = np.array([63, 255, 255])  # 주황색의 HSV 범위 상한값
green_lower = np.array([78, 76, 143])  # 녹색의 HSV 범위 하한값
green_upper = np.array([115, 217, 252])  # 녹색의 HSV 범위 상한값

# 트랙바의 값을 변경할 때 호출되는 콜백 함수입니다. 여기서는 아무것도 하지 않습니다.
def nothing(x):
    pass

# 트랙바를 표시할 창을 생성합니다.
cv2.namedWindow("Threshold Trackbars")

# 트랙바를 생성합니다. 각 트랙바는 색상 검출 시 사용되는 임계값을 설정할 수 있습니다.
cv2.createTrackbar("Red Threshold", "Threshold Trackbars", 400, 1000, nothing)  # 빨간색 임계값 트랙바 (최대 1000)
cv2.createTrackbar("Orange Threshold", "Threshold Trackbars", 1300, 3000, nothing)  # 주황색 임계값 트랙바 (최대 3000)
cv2.createTrackbar("Green Threshold", "Threshold Trackbars", 1700, 3000, nothing)  # 녹색 임계값 트랙바 (최대 3000)

# 주어진 마스크에서 특정 위치(x, y)를 중심으로 원형 윈도우 내의 픽셀 수를 계산합니다.
def count_color_pixels(mask, x, y, window_radius):
    mask_circle = np.zeros_like(mask)  # 원형 마스크를 초기화합니다. 마스크와 동일한 크기의 배열을 만듭니다.
    cv2.circle(mask_circle, (x, y), window_radius, 1, thickness=-1)  # 주어진 위치에 반지름이 window_radius인 원을 그립니다.
    return cv2.countNonZero(mask & mask_circle)  # 원 내에서 마스크가 활성화된 (1로 설정된) 픽셀 수를 반환합니다.

# ROS 이미지 메시지를 수신할 때 호출되는 콜백 함수입니다.
def image_callback(msg):
    try:
        # ROS 이미지 메시지를 OpenCV 이미지로 변환합니다.
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")  # "bgr8"은 OpenCV에서 사용하는 기본 이미지 형식입니다.

        # 이미지를 640x480 크기로 조정합니다.
        frame = cv2.resize(cv_image, (640, 480))

        # 이미지 상단 1/3 영역을 ROI(Region of Interest)로 설정합니다. 신호등이 이 영역에 위치할 가능성이 높기 때문입니다.
        roi = frame[0:int(frame.shape[0] * 1/3), :]

        # 노이즈 제거를 위해 Gaussian Blur를 적용합니다. 커널 크기는 (5, 5)이며 표준 편차는 2입니다.
        blurred_roi = cv2.GaussianBlur(roi, (5, 5), 2)

        # 이미지를 HSV 색상 공간으로 변환합니다. 이 색상 공간은 색상 검출에 더 적합합니다.
        hsv_roi = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2HSV)

        # 각 색상 범위에 따라 마스크를 생성합니다. 마스크는 해당 색상의 픽셀만을 포함합니다.
        red_mask = cv2.inRange(hsv_roi, red_lower, red_upper)  # 빨간색 마스크 생성
        orange_mask = cv2.inRange(hsv_roi, orange_lower, orange_upper)  # 주황색 마스크 생성
        green_mask = cv2.inRange(hsv_roi, green_lower, green_upper)  # 녹색 마스크 생성

        # 트랙바에서 임계값 값을 가져옵니다. 사용자가 실시간으로 임계값을 조정할 수 있습니다.
        red_threshold = cv2.getTrackbarPos("Red Threshold", "Threshold Trackbars")
        orange_threshold = cv2.getTrackbarPos("Orange Threshold", "Threshold Trackbars")
        green_threshold = cv2.getTrackbarPos("Green Threshold", "Threshold Trackbars")

        # 윈도우 크기와 이동 스텝을 설정합니다. 신호등의 크기와 화면 내 위치를 고려하여 설정합니다.
        window_radius = 30  # 원형 윈도우의 반지름 설정
        step_size = 20  # 윈도우 이동 스텝 크기 설정

        # 각 색상에 대해 가장 큰 픽셀 수와 해당 위치를 저장할 변수를 초기화합니다.
        max_red = 0  # 빨간색 픽셀 수 최대값
        max_orange = 0  # 주황색 픽셀 수 최대값
        max_green = 0  # 녹색 픽셀 수 최대값
        best_red_position = None  # 빨간색 픽셀 최대값 위치
        best_orange_position = None  # 주황색 픽셀 최대값 위치
        best_green_position = None  # 녹색 픽셀 최대값 위치

        # ROI 영역 내에서 일정 간격으로 윈도우를 이동하며 색상 검출을 수행합니다.
        for y in range(window_radius, roi.shape[0] - window_radius, step_size):  # y 방향으로 이동
            for x in range(window_radius, roi.shape[1] - window_radius, step_size):  # x 방향으로 이동
                # 각 위치에서 색상 픽셀 수를 계산합니다.
                red_count = count_color_pixels(red_mask, x, y, window_radius)
                orange_count = count_color_pixels(orange_mask, x, y, window_radius)
                green_count = count_color_pixels(green_mask, x, y, window_radius)

                # 각 색상에 대해 임계값을 초과하는 최대 픽셀 수와 위치를 저장합니다.
                if red_count > max_red and red_count > red_threshold:  # 빨간색 임계값 체크
                    max_red = red_count
                    best_red_position = (x, y)
                if orange_count > max_orange and orange_count > orange_threshold:  # 주황색 임계값 체크
                    max_orange = orange_count
                    best_orange_position = (x, y)
                if green_count > max_green and green_count > green_threshold:  # 녹색 임계값 체크
                    max_green = green_count
                    best_green_position = (x, y)

        # 각 색상에 대해 감지된 위치에 원을 그립니다.
        if best_red_position:
            cv2.circle(roi, best_red_position, window_radius, (0, 0, 255), 2)  # 빨간색 원 그리기
            cv2.putText(roi, "Red", (best_red_position[0] - 50, best_red_position[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # "Red" 텍스트 추가
        if best_orange_position:
            cv2.circle(roi, best_orange_position, window_radius, (0, 165, 255), 2)  # 주황색 원 그리기
            cv2.putText(roi, "Orange", (best_orange_position[0] - 50, best_orange_position[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)  # "Orange" 텍스트 추가
        if best_green_position:
            cv2.circle(roi, best_green_position, window_radius, (0, 255, 0), 2)  # 녹색 원 그리기
            cv2.putText(roi, "Green", (best_green_position[0] - 50, best_green_position[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # "Green" 텍스트 추가

        # 결과 화면을 출력합니다.
        cv2.imshow("Original", frame)  # 원본 이미지를 표시합니다.
        cv2.imshow("ROI", roi)  # ROI 영역만을 별도로 표시합니다.

        # 'Esc' 키를 누르면 프로그램을 종료합니다.
        if cv2.waitKey(1) == 27:
            rospy.signal_shutdown('User pressed Esc')  # ROS 노드를 안전하게 종료합니다.
            cv2.destroyAllWindows()  # 모든 OpenCV 창을 닫습니다.

    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")  # CvBridge에서 오류가 발생하면 로그에 기록합니다.

# ROS에서 /usb_cam/image_raw 토픽을 구독하여 이미지 데이터를 수신합니다. 수신된 이미지는 image_callback 함수에서 처리됩니다.
image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)

# ROS 루프를 시작하여 노드를 계속 실행 상태로 유지합니다.
rospy.spin()

# ROS 노드가 종료될 때 모든 OpenCV 창을 닫습니다.
cv2.destroyAllWindows()
