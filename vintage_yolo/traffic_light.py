#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

# ROS 노드 초기화
rospy.init_node('circle_detector', anonymous=True)

# CvBridge 객체 생성
bridge = CvBridge()

# 트랙바 콜백 함수 (필수는 아니지만 필요함)
def nothing(x):
    pass

# 트랙바 윈도우 창 만들기
cv2.namedWindow("Trackbars")

# 트랙바 구성요소 만들기
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# 이미지 콜백 함수
def image_callback(msg):
    try:
        # ROS 이미지 메시지를 OpenCV 이미지로 변환
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

        # 이미지 크기를 조정 (선택사항)
        frame = cv2.resize(cv_image, (640, 480))

        # HSV 변환
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 트랙바에서 설정된 값 가져오기
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        lower = np.array([l_h, l_s, l_v])
        upper = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv_frame, lower, upper)

        filtered_circles = []  # 빈 리스트로 초기화

        # 원 검출
        circles = cv2.HoughCircles(
            mask,                     # 입력 이미지 (마스크된 이미지)
            cv2.HOUGH_GRADIENT,       # 허프 변환 방법
            dp=1,                     # 해상도 비율
            minDist=200,               # 원의 중심들 간의 최소 거리
            param1=100,               # Canny 에지 검출기의 높은 임계값
            param2=30,                # 원 검출 임계값
            minRadius=10,             # 최소 반지름
            maxRadius=100             # 최대 반지름
        )

        # 원이 검출되면 반지름 크기를 기준으로 필터링하고, 선명한 원 5개만 표시
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            for (x, y, r) in circles:
                if 10 <= r <= 100:  # 반지름 크기 필터링
                    # 원이 이미지 경계를 벗어나지 않도록 조건 추가
                    if y - r >= 0 and y + r < mask.shape[0] and x - r >= 0 and x + r < mask.shape[1]:
                        circle_roi = mask[y-r:y+r, x-r:x+r]
                        if circle_roi.size > 0:  # ROI가 유효한지 확인
                            edges = cv2.Canny(circle_roi, 50, 150)
                            edge_count = np.sum(edges) / 255  # 에지 픽셀의 개수 계산
                            filtered_circles.append((x, y, r, edge_count))

            # 선명도 기준으로 상위 5개의 원 선택
            filtered_circles = sorted(filtered_circles, key=lambda x: x[3], reverse=True)[:5]

            # 선택된 원 그리기
            for (x, y, r, _) in filtered_circles:
                cv_image = cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                cv_image = cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)  # 중심 표시

        # 결과 화면 출력
        cv_image = cv2.putText(frame, f"Circles detected: {len(filtered_circles)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Original", frame)
        cv2.imshow("Masked Image", mask)

        # 'Esc' 키를 누르면 종료
        if cv2.waitKey(1) == 27:
            rospy.signal_shutdown('User pressed Esc')
            cv2.destroyAllWindows()

    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")

# 이미지 토픽 구독자 설정
image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)

# ROS 루프 시작
rospy.spin()

# ROS 노드 종료 시 OpenCV 창 닫기
cv2.destroyAllWindows()
