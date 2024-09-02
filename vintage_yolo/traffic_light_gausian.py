#!/usr/bin/env python3  # 파이썬 스크립트를 실행할 때 사용하는 기본 인터프리터 경로를 지정합니다.

import rospy  # ROS (Robot Operating System)와 통신하기 위해 rospy 모듈을 불러옵니다.
from sensor_msgs.msg import Image  # ROS 이미지 메시지 타입을 사용하기 위해 불러옵니다.
from cv_bridge import CvBridge, CvBridgeError  # ROS 이미지를 OpenCV 이미지로 변환하기 위해 CvBridge를 사용합니다.
import cv2  # OpenCV 라이브러리, 이미지 및 비디오 처리에 사용됩니다.
import numpy as np  # NumPy 라이브러리, 배열 계산 및 이미지 처리에 사용됩니다.

# ROS 노드를 초기화합니다. 노드 이름은 'circle_detector'입니다.
rospy.init_node('circle_detector', anonymous=True)

# CvBridge 객체를 생성합니다. ROS 이미지 메시지를 OpenCV 이미지로 변환하는 데 사용됩니다.
bridge = CvBridge()

# 트랙바 콜백 함수입니다. 트랙바 값이 변경될 때 호출되지만, 여기서는 아무런 작업도 하지 않습니다.
def nothing(x):
    pass

# 트랙바를 위한 창을 생성합니다.
cv2.namedWindow("Trackbars")

# 트랙바를 생성합니다. 각 트랙바는 원 검출을 위한 HSV 범위 값을 설정하는 데 사용됩니다.
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)  # 낮은 Hue 값 설정
cv2.createTrackbar("L - S", "Trackbars", 45, 255, nothing)  # 낮은 Saturation 값 설정
cv2.createTrackbar("L - V", "Trackbars", 182, 255, nothing)  # 낮은 Value 값 설정
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)  # 높은 Hue 값 설정
cv2.createTrackbar("U - S", "Trackbars", 222, 255, nothing)  # 높은 Saturation 값 설정
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)  # 높은 Value 값 설정

# ROS 이미지 메시지를 수신할 때 호출되는 콜백 함수입니다.
def image_callback(msg):
    try:
        # ROS 이미지 메시지를 OpenCV 이미지로 변환합니다.
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")  # "bgr8"은 OpenCV에서 사용하는 기본 이미지 형식입니다.

        # 이미지를 640x480 크기로 조정합니다.
        frame = cv2.resize(cv_image, (640, 480))

        # Gaussian Blur를 적용하여 노이즈를 줄입니다. 커널 크기는 (5, 5)이며, 표준 편차는 2로 설정됩니다.
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 2)

        # 이미지를 HSV 색상 공간으로 변환합니다. 이 색상 공간은 색상 기반 마스킹에 더 적합합니다.
        hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

        # 트랙바에서 설정된 값을 가져옵니다.
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        # 트랙바에서 가져온 값으로 HSV 범위를 설정합니다.
        lower = np.array([l_h, l_s, l_v])  # 하한 HSV 값 설정
        upper = np.array([u_h, u_s, u_v])  # 상한 HSV 값 설정
        mask = cv2.inRange(hsv_frame, lower, upper)  # 설정된 HSV 범위에 맞는 픽셀만 포함하는 마스크 생성

        # Canny 에지 검출기를 사용하여 에지를 검출합니다. 임계값은 100과 200으로 설정됩니다.
        edges = cv2.Canny(mask, 100, 200)

        filtered_circles = []  # 검출된 원 정보를 저장할 빈 리스트

        # 원 검출을 수행합니다.
        circles = cv2.HoughCircles(
            edges,                   # 입력 이미지 (에지 이미지)
            cv2.HOUGH_GRADIENT,      # 허프 변환 방법 사용
            dp=1,                    # 해상도 비율
            minDist=20,              # 원의 중심들 간의 최소 거리 설정
                                     # param1이 높으면 노이즈가 줄어들지만 약한 원은 놓칠 수 있습니다.
                                     # param2가 높으면 정확한 원만 검출되지만, 검출된 원의 수가 줄어들 수 있습니다.
            param1=50,               # Canny 에지 검출기의 높은 임계값
            param2=30,               # 원 검출 임계값
            minRadius=0,             # 검출할 원의 최소 반지름
            maxRadius=1000           # 검출할 원의 최대 반지름
        )

        # 원이 검출되면, 검출된 원의 반지름 크기를 기준으로 필터링하고, 선명한 원 5개만 표시합니다.
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")  # 검출된 원 정보를 정수로 변환

            for (x, y, r) in circles:
                if 10 <= r <= 100:  # 반지름 크기를 필터링
                    if y - r >= 0 and y + r < mask.shape[0] and x - r >= 0 and x + r < mask.shape[1]:
                        circle_roi = edges[y-r:y+r, x-r:x+r]  # 원의 ROI (Region of Interest) 설정
                        if circle_roi.size > 0:  # ROI가 유효한지 확인
                            edge_count = np.sum(circle_roi) / 255  # 에지 픽셀의 개수 계산
                            filtered_circles.append((x, y, r, edge_count))  # 필터링된 원 정보를 리스트에 추가

            # 선명도 기준으로 상위 5개의 원 선택
            filtered_circles = sorted(filtered_circles, key=lambda x: x[3], reverse=True)[:5]

            # 선택된 원을 이미지에 그리기
            for (x, y, r, _) in filtered_circles:
                cv_image = cv2.circle(frame, (x, y), r, (0, 255, 0), 2)  # 원을 그리기
                cv_image = cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)  # 원의 중심 표시

        # 결과 화면 출력
        cv_image = cv2.putText(frame, f"Circles detected: {len(filtered_circles)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 원의 개수 표시
        cv2.imshow("Original", frame)  # 원본 이미지를 표시
        cv2.imshow("Masked Image", mask)  # 마스크 이미지를 표시
        cv2.imshow("Edges", edges)  # 에지 이미지를 표시

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

# ROS 노드 종료 시 모든 OpenCV 창을 닫습니다.
cv2.destroyAllWindows()
