import numpy as np
import cv2
import cv2.aruco as aruco
import time

ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.02
MARKER_LENGTH = 0.01
LENGTH_PX = 640   # total length of the page in pixels
MARGIN_PX = 20    # size of the margin in pixels

def camera_calibration():
    # ArUco 마커 사전 정의
    # ChAruco board variables
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    
    # img = cv2.aruco.CharucoBoard.generateImage(board, (LENGTH_PX, int(LENGTH_PX*size_ratio)), marginSize=MARGIN_PX)
    # cv2.imshow("img", img)
    # cv2.waitKey(2000)
    
    # 캘리브레이션을 위한 변수 초기화
    all_corners = []
    all_ids = []
    counter = 0
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    print("camera open")
    # ArUco 검출기 생성
    # charuco_param = cv2.aruco.CharucoParameters()
    detector_param = cv2.aruco.DetectorParameters()
    # detector = cv2.aruco.CharucoDetector(board, charuco_param, detector_param)
    
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 마커 검출
        # charucoCorners, charucoIds, markerCorners, markerIds = detector.detectBoard(gray)
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=detector_param)
        display = frame.copy()
        cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)

        # 현재 시간을 기준으로 2초마다 한번씩 저장
        current_time = time.time()
        if current_time - last_time >= 2:
            try:
                if len(marker_ids) > 0:
                    charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)
                    if charuco_retval and len(charuco_corners) > 3:
                        all_corners.append(charuco_corners)
                        all_ids.append(charuco_ids)
                        counter += 1
                        print(f"저장된 프레임 수: {counter}")
            except:
                pass
            last_time = current_time

        cv2.imshow('Frame', display)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("exp_charuco.png", frame)
            print("exp_charuco.png saved")

        # 충분한 수의 프레임이 저장되면 캘리브레이션 수행
        if counter >= 20:
            print("캘리브레이션 수행 중...")
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                all_corners, all_ids, board, gray.shape[::-1], None, None)
            
            print("\n카메라 행렬:")
            print(camera_matrix)
            print("\n왜곡 계수:")
            print(dist_coeffs)
            print("이미지 크기:\n",gray.shape[::-1])
            cv2.imwrite("calibration_image.png", frame)
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return camera_matrix, dist_coeffs

if __name__ == "__main__":
    camera_matrix, dist_coeffs = camera_calibration()
    
    
