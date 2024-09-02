#!/usr/bin/env python3  # 파이썬 스크립트를 실행할 때 사용하는 기본 인터프리터 경로를 지정합니다.

import rospy  # ROS (Robot Operating System)와 통신하기 위해 rospy 모듈을 불러옵니다.
from sensor_msgs.msg import Image  # ROS 이미지 메시지 타입을 사용하기 위해 불러옵니다.
from cv_bridge import CvBridge, CvBridgeError  # ROS 이미지를 OpenCV 이미지로 변환하기 위해 CvBridge를 사용합니다.
import cv2  # OpenCV 라이브러리, 이미지 및 비디오 처리에 사용됩니다.
import numpy as np  # NumPy 라이브러리, 배열 계산 및 이미지 처리에 사용됩니다.

# CvBridge 객체 생성. ROS 이미지 메시지를 OpenCV 이미지로 변환하는 데 사용됩니다.
bridge = CvBridge()

# 트랙바 콜백 함수입니다. 트랙바 값이 변경될 때 호출되지만, 여기서는 아무런 작업도 하지 않습니다.
def nothing(x):
    pass

# 트랙바를 위한 창을 생성합니다.
cv2.namedWindow("Trackbars")

# 트랙바를 생성합니다. 각 트랙바는 색상 검출을 위한 HSV 범위 값을 설정하는 데 사용됩니다.
# 빨간색에 대한 트랙바 생성
cv2.createTrackbar("L - H (Red)", "Trackbars", 0, 255, nothing)  # 낮은 Hue 값 설정
cv2.createTrackbar("L - S (Red)", "Trackbars", 100, 255, nothing)  # 낮은 Saturation 값 설정
cv2.createTrackbar("L - V (Red)", "Trackbars", 100, 255, nothing)  # 낮은 Value 값 설정
cv2.createTrackbar("U - H (Red)", "Trackbars", 10, 255, nothing)  # 높은 Hue 값 설정
cv2.createTrackbar("U - S (Red)", "Trackbars", 255, 255, nothing)  # 높은 Saturation 값 설정
cv2.createTrackbar("U - V (Red)", "Trackbars", 255, 255, nothing)  # 높은 Value 값 설정

# 주황색에 대한 트랙바 생성
cv2.createTrackbar("L - H (Orange)", "Trackbars", 10, 255, nothing)
cv2.createTrackbar("L - S (Orange)", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("L - V (Orange)", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("U - H (Orange)", "Trackbars", 25, 255, nothing)
cv2.createTrackbar("U - S (Orange)", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V (Orange)", "Trackbars", 255, 255, nothing)

# 초록색에 대한 트랙바 생성
cv2.createTrackbar("L - H (Green)", "Trackbars", 40, 255, nothing)
cv2.createTrackbar("L - S (Green)", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("L - V (Green)", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("U - H (Green)", "Trackbars", 70, 255, nothing)
cv2.createTrackbar("U - S (Green)", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V (Green)", "Trackbars", 255, 255, nothing)

# ROS 이미지 메시지를 수신할 때 호출되는 콜백 함수입니다.
def image_callback(msg):
    try:
        # ROS 이미지 메시지를 OpenCV 이미지로 변환합니다.
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")  # "bgr8"은 OpenCV에서 사용하는 기본 이미지 형식입니다.

        # 이미지를 HSV 색상 공간으로 변환합니다. 이 색상 공간은 색상 검출에 더 적합합니다.
        hsv_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # 트랙바에서 설정된 값 가져오기 (빨간색)
        l_h_red = cv2.getTrackbarPos("L - H (Red)", "Trackbars")
        l_s_red = cv2.getTrackbarPos("L - S (Red)", "Trackbars")
        l_v_red = cv2.getTrackbarPos("L - V (Red)", "Trackbars")
        u_h_red = cv2.getTrackbarPos("U - H (Red)", "Trackbars")
        u_s_red = cv2.getTrackbarPos("U - S (Red)", "Trackbars")
        u_v_red = cv2.getTrackbarPos("U - V (Red)", "Trackbars")

        # 트랙바에서 설정된 값 가져오기 (주황색)
        l_h_orange = cv2.getTrackbarPos("L - H (Orange)", "Trackbars")
        l_s_orange = cv2.getTrackbarPos("L - S (Orange)", "Trackbars")
        l_v_orange = cv2.getTrackbarPos("L - V (Orange)", "Trackbars")
        u_h_orange = cv2.getTrackbarPos("U - H (Orange)", "Trackbars")
        u_s_orange = cv2.getTrackbarPos("U - S (Orange)", "Trackbars")
        u_v_orange = cv2.getTrackbarPos("U - V (Orange)", "Trackbars")

        # 트랙바에서 설정된 값 가져오기 (초록색)
        l_h_green = cv2.getTrackbarPos("L - H (Green)", "Trackbars")
        l_s_green = cv2.getTrackbarPos("L - S (Green)", "Trackbars")
        l_v_green = cv2.getTrackbarPos("L - V (Green)", "Trackbars")
        u_h_green = cv2.getTrackbarPos("U - H (Green)", "Trackbars")
        u_s_green = cv2.getTrackbarPos("U - S (Green)", "Trackbars")
        u_v_green = cv2.getTrackbarPos("U - V (Green)", "Trackbars")

        # 색상 범위를 설정합니다.
        red_lower = np.array([l_h_red, l_s_red, l_v_red])  # 빨간색의 HSV 범위 하한값 설정
        red_upper = np.array([u_h_red, u_s_red, u_v_red])  # 빨간색의 HSV 범위 상한값 설정
        orange_lower = np.array([l_h_orange, l_s_orange, l_v_orange])  # 주황색의 HSV 범위 하한값 설정
        orange_upper = np.array([u_h_orange, u_s_orange, u_v_orange])  # 주황색의 HSV 범위 상한값 설정
        green_lower = np.array([l_h_green, l_s_green, l_v_green])  # 초록색의 HSV 범위 하한값 설정
        green_upper = np.array([u_h_green, u_s_green, u_v_green])  # 초록색의 HSV 범위 상한값 설정

        # 각 색상에 대한 마스크를 생성합니다. 마스크는 해당 색상에 해당하는 픽셀만을 포함합니다.
        red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)  # 빨간색 마스크 생성
        orange_mask = cv2.inRange(hsv_frame, orange_lower, orange_upper)  # 주황색 마스크 생성
        green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)  # 초록색 마스크 생성

        # 각 색상 마스크를 결합하여 하나의 마스크로 만듭니다.
        combined_mask = cv2.bitwise_or(red_mask, orange_mask)
        combined_mask = cv2.bitwise_or(combined_mask, green_mask)

        # 원본 이미지에 마스크를 적용하여 색상이 검출된 부분만 남깁니다.
        masked_frame = cv2.bitwise_and(cv_image, cv_image, mask=combined_mask)

        # 결과 화면을 출력합니다.
        cv2.imshow("Original", cv_image)  # 원본 이미지를 표시합니다.
        cv2.imshow("Red Mask", red_mask)  # 빨간색 마스크를 표시합니다.
        cv2.imshow("Orange Mask", orange_mask)  # 주황색 마스크를 표시합니다.
        cv2.imshow("Green Mask", green_mask)  # 초록색 마스크를 표시합니다.
        cv2.imshow("Masked Image", masked_frame)  # 마스크가 적용된 결과 이미지를 표시합니다.

        # 'Esc' 키를 누르면 프로그램을 종료합니다.
        if cv2.waitKey(1) == 27:
            rospy.signal_shutdown('User pressed Esc')  # ROS 노드를 안전하게 종료합니다.
            cv2.destroyAllWindows()  # 모든 OpenCV 창을 닫습니다.

    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")  # CvBridge에서 오류가 발생하면 로그에 기록합니다.

# ROS 노드를 초기화합니다. 노드 이름은 'color_detector'입니다.
rospy.init_node('color_detector', anonymous=True)

# ROS에서 /usb_cam/image_raw 토픽을 구독하여 이미지 데이터를 수신합니다. 수신된 이미지는 image_callback 함수에서 처리됩니다.
image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)

# ROS 루프를 시작하여 노드를 계속 실행 상태로 유지합니다.
rospy.spin()

# 프로그램 종료 시 모든 창을 닫습니다.
cv2.destroyAllWindows()
