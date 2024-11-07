import cv2
import numpy as np
import time
# ChArUco 보드 생성 함수
def create_charuco_board():
    # squares_x = 5  # x축 정사각형 수
    # squares_y = 7  # y축 정사각형 수
    # square_length = 0.04  # 정사각형 길이 (미터 단위 예시)
    # marker_length = 0.02  # 마커 크기 (미터 단위 예시)

    # aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    # board = cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, aruco_dict)
    
    ARUCO_DICT = cv2.aruco.DICT_6X6_250
    SQUARES_VERTICALLY = 7
    SQUARES_HORIZONTALLY = 5
    SQUARE_LENGTH = 0.02
    MARKER_LENGTH = 0.01
    LENGTH_PX = 640   # total length of the page in pixels
    MARGIN_PX = 20    # size of the margin in pixels

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    # size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    # img = cv2.aruco.CharucoBoard.generateImage(board, (LENGTH_PX, int(LENGTH_PX*size_ratio)), marginSize=MARGIN_PX)
    
    return board

# 카메라 캘리브레이션 함수
# def calibrate_camera_with_charuco(board, cap, num_images=20):
#     all_corners = []
#     all_ids = []
#     image_size = None
#     count = 0

#     while count < num_images:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         corners, ids, _ = cv2.aruco.detectMarkers(gray, board.dictionary)

#         if ids is not None:
#             _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
#             if charuco_corners is not None and charuco_ids is not None:
#                 all_corners.append(charuco_corners)
#                 all_ids.append(charuco_ids)
#                 count += 1
#                 cv2.aruco.drawDetectedMarkers(frame, corners, ids)

#                 if image_size is None:
#                     image_size = gray.shape[::-1]

#                 cv2.imshow("ChArUco Calibration", frame)
#                 if cv2.waitKey(500) & 0xFF == ord('q'):  # 'q'를 누르면 종료
#                     break

#     # 캘리브레이션 수행
#     if len(all_corners) > 0:
#         ret, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
#             charucoCorners=all_corners,
#             charucoIds=all_ids,
#             board=board,
#             imageSize=image_size,
#             cameraMatrix=None,
#             distCoeffs=None
#         )
#         if ret:
#             print("Calibration successful.")
#             return camera_matrix, dist_coeffs
#     print("Calibration failed.")
#     return None, None

def calibrate_camera_with_charuco(board, cap, num_images=20):
    # 캘리브레이션을 위한 변수 초기화
    all_corners = []
    all_ids = []
    counter = 0
    
    # ArUco 검출기 생성
    detector_param = cv2.aruco.DetectorParameters()
    
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 마커 검출
        # charucoCorners, charucoIds, markerCorners, markerIds = detector.detectBoard(gray)
        aruco_dict = board.getDictionary()
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_param)
        cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

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

        cv2.imshow('Frame', frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        # 충분한 수의 프레임이 저장되면 캘리브레이션 수행
        if counter >= 20:
            print("캘리브레이션 수행 중...")
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                all_corners, all_ids, board, gray.shape[::-1], None, None)
            
            print("\n카메라 행렬:")
            print(camera_matrix)
            print("\n왜곡 계수:")
            print(dist_coeffs)
    
            return camera_matrix, dist_coeffs

# bird-eye view 변환 함수
def charuco_bird_eye_view(image, board, camera_matrix, dist_coeffs):
    if camera_matrix is not None and dist_coeffs is not None:
        image = cv2.undistort(image, camera_matrix, dist_coeffs)

    parameters = cv2.aruco.DetectorParameters()
    aruco_dict = board.getDictionary()
    corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

    # 디버깅: 마커 검출 결과 확인
    print("검출된 마커 수:", len(ids) if ids is not None else 0)
    
    if ids is not None:
        # 검출된 마커 시각화
        debug_image = image.copy()
        cv2.aruco.drawDetectedMarkers(debug_image, corners, ids)
        cv2.imshow("Detected Markers", debug_image)
        cv2.waitKey(1)

        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, image, board)

        # 디버깅: ChArUco 코너 검출 결과 확인
        print("검출된 ChArUco 코너 수:", len(charuco_ids) if charuco_ids is not None else 0)

        if charuco_ids is not None and len(charuco_ids) >= 4:
            src_points = charuco_corners.reshape(-1, 2)
            
            # 픽셀 단위로 크기 조정
            PIXEL_SIZE = 100  # 각 정사각형의 픽셀 크기
            dst_points = []
            for y in range(board.getChessboardSize()[1]-1):
                for x in range(board.getChessboardSize()[0]-1):
                    dst_points.append([
                        PIXEL_SIZE + PIXEL_SIZE*x,
                        PIXEL_SIZE + PIXEL_SIZE*y
                    ])
            dst_points = np.array(dst_points, dtype=np.float32)
            
            # 디버깅: 포인트 수 확인
            print("src_points shape:", src_points.shape)
            print("dst_points shape:", dst_points.shape)
            
            transform_matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
            
            if transform_matrix is not None:
                # 출력 이미지 크기 계산
                max_width = int(np.max(dst_points[:, 0]) + PIXEL_SIZE)
                max_height = int(np.max(dst_points[:, 1]) + PIXEL_SIZE)
                
                print("출력 이미지 크기:", max_width, "x", max_height)
                
                bird_eye_view = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))
                # bird_eye_view = cv2.warpPerspective(image, transform_matrix, (1920, 1080))

                return bird_eye_view, transform_matrix
            else:
                print("Homography 행렬 계산 실패")
    else:
        print("마커를 찾을 수 없습니다")

    return None

if __name__ == "__main__":
    # ChArUco 보드와 캡처 장치 준비
    board = create_charuco_board()
    
    img = cv2.imread("exp_charuco.png")
    if img is None:
        print("이미지를 불러올 수 없습니다")
    else:
        print("이미지 크기:", img.shape)
        cv2.imshow("Original", img)
        bird_eye_view, transform_matrix = charuco_bird_eye_view(img, board, None, None)
        
        if bird_eye_view is not None:
            cv2.imshow("Bird-Eye View", bird_eye_view)
            print("transform_matrix:", transform_matrix.shape)
            cv2.waitKey(0)
        else:
            print("버드아이뷰 변환 실패")

    # print("opening camera...")
    # cap = cv2.VideoCapture(0)  # 웹캠 사용
    # print("camera opened")

    # camera_matrix, dist_coeffs = None, None

    # # 카메라 매트릭스 및 왜곡 계수 계산
    # # camera_matrix, dist_coeffs = calibrate_camera_with_charuco(board, cap, num_images=20)

    # # 실시간 Bird-eye view 변환
    # print("Starting bird-eye view transformation. Press 'q' to exit.")
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     bird_eye_view = charuco_bird_eye_view(frame, board, camera_matrix, dist_coeffs)
    #     if bird_eye_view is not None:
    #         cv2.imshow("Bird-Eye View", bird_eye_view)
    #     else:
    #         cv2.imshow("Original", frame)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'를 누르면 종료
    #         break
    #     elif cv2.waitKey(1) & 0xFF == ord('s'):
    #         cv2.imwrite("exp_charuco.png",frame)

    # cap.release()
    # cv2.destroyAllWindows()
