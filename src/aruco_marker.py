import cv2
import cv2.aruco as aruco
import numpy as np

class MarkerDetector:
    def __init__(self, debug=False):
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parameters = aruco.DetectorParameters()
        self.debug = debug

    def detect_marker(self, frame):
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # 마커 검출
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        if ids is not None:
            # aruco.drawDetectedMarkers(frame, corners, ids)
            top_left = tuple(corners[0][0][0].astype(int))
            top_right = tuple(corners[0][0][1].astype(int))
            bottom_right = tuple(corners[0][0][2].astype(int))
            bottom_left = tuple(corners[0][0][3].astype(int))
            top_length = ((top_left[0] - top_right[0]) ** 2 + (top_left[1] - top_right[1]) ** 2) ** 0.5
            left_length = ((top_left[0] - bottom_left[0]) ** 2 + (top_left[1] - bottom_left[1]) ** 2) ** 0.5

            # 각 마커에 대해 ID 표시
            if self.debug:
                for i, corner in enumerate(corners):
                    marker_id = ids[i][0]
                    cv2.circle(frame, top_left, 5, (0, 0, 255), -1)      # 빨간색
                    cv2.circle(frame, top_right, 5, (0, 255, 0), -1)     # 초록색 
                    cv2.circle(frame, bottom_right, 5, (255, 0, 0), -1)  # 파란색
                    cv2.circle(frame, bottom_left, 5, (0, 255, 255), -1) # 노란색
                
                cv2.line(frame, top_left, top_right, (0, 0, 255), 2)
                cv2.putText(frame, str(top_length).format(2.2), ((top_left[0]+top_right[0])//2, (top_left[1]+top_right[1])//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.line(frame, top_left, bottom_left, (0, 255, 255), 2)
                cv2.putText(frame, str(left_length).format(2.2), ((top_left[0]+bottom_left[0])//2, (top_left[1]+bottom_left[1])//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.imshow('Frame', frame)

            return top_left, top_right, bottom_left, bottom_right
        else:
            if self.debug:
                cv2.imshow('Frame', frame)
            return None, None, None, None

    def draw_marker(self, frame, corners, ids):
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

class CharucoDetector:
    def __init__(self, debug=False):
        self.debug = debug
        self.ARUCO_DICT = cv2.aruco.DICT_4X4_100
        self.SQUARES_VERTICALLY = 7
        self.SQUARES_HORIZONTALLY = 5
        self.SQUARE_LENGTH = 60 
        self.MARKER_LENGTH = 45

        self.dictionary = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT)
        self.board = cv2.aruco.CharucoBoard((self.SQUARES_HORIZONTALLY, self.SQUARES_VERTICALLY), self.SQUARE_LENGTH, self.MARKER_LENGTH, self.dictionary)
        
        self.detector_param = cv2.aruco.DetectorParameters()

        
        # w_offset = 1920 // 2
        # h_offset = 1080 // 2
        # self.dst_points = []
        # for y in range(self.board.getChessboardSize()[1]-1):
        #     for x in range(self.board.getChessboardSize()[0]-1):
        #         self.dst_points.append([
        #                 w_offset + self.PIXEL_SIZE + self.PIXEL_SIZE*x,
        #                 h_offset + self.PIXEL_SIZE + self.PIXEL_SIZE*y
        #             ])
        # self.dst_points = np.array(self.dst_points, dtype=np.float32)

        self.frame_width = 2048 
        self.frame_height = 3058
        
        # 마커들의 코너 좌표 생성
        self.dst_points_aruco = np.zeros((77, 4, 2), dtype=np.float32)
        
        for n in range(4):
            id_offset = n * 20
            count = 0
            id = 0
            x_offset = (n % 2)* (self.SQUARE_LENGTH * self.SQUARES_HORIZONTALLY)
            y_offset = (n // 2)* (self.SQUARE_LENGTH * self.SQUARES_VERTICALLY)
            for i in range(self.SQUARES_VERTICALLY):

                for j in range(self.SQUARES_HORIZONTALLY):
                    count += 1
                    if count % 2 == 1: continue

                    idx = id + id_offset
                    id += 1
                    x = x_offset + j * self.SQUARE_LENGTH
                    y = y_offset + i * self.SQUARE_LENGTH
                    
                    # 각 마커의 4개 코너점 설정
                    top_left = [x+(self.SQUARE_LENGTH - self.MARKER_LENGTH)//2, y+(self.SQUARE_LENGTH - self.MARKER_LENGTH)//2]
                    self.dst_points_aruco[idx] = np.array([
                        top_left,
                        [top_left[0] + self.MARKER_LENGTH, top_left[1]], # 우상단
                        [top_left[0] + self.MARKER_LENGTH, top_left[1] + self.MARKER_LENGTH], # 우하단
                        [top_left[0], top_left[1] + self.MARKER_LENGTH] # 좌하단
                    ])

    def detect_charuco(self, frame):
        '''
        return:
            src_points: 원본이미지 좌표계에서의 charuco 좌표
        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.detector_param)
        
        if marker_ids is not None:
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, self.board)
            
            if self.debug:
                cv2.putText(frame, f"num of detected marker: {len(marker_ids) if marker_ids is not None else 0}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"num of detected charuco: {len(charuco_ids) if charuco_ids is not None else 0}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 마커 그리기
                cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
                
                # charuco 코너가 검출된 경우에만 그리기
                if charuco_corners is not None and charuco_ids is not None:
                    charuco_corners = charuco_corners.reshape(-1, 1, 2)  # reshape to correct format
                    cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
                
                cv2.imshow('Frame', frame)
            
            if charuco_ids is not None and len(charuco_ids) >= 4:
                src_points = charuco_corners.reshape(-1, 2)
                return src_points
            else:
                if self.debug:
                    print("최소 4개의 charuco코너가 검출되지 않았습니다.", end='\r', flush=True)
                return None
        
        else:
            if self.debug:
                print("aruco 마커가 검출되지 않았습니다.")
                cv2.imshow('Frame', frame)
            return None # aruco 마커가 검출되지 않으면 None 반환


    def detect_aruco(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # OpenCV 4.x 버전용 ArUco 검출 코드
        detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_param)
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        
        if marker_ids is not None:
            # num_markers = len(marker_ids)
            # corners_array = np.zeros((num_markers, 4, 2), dtype=int)
            
            # for i in range(num_markers):
            #     corners = marker_corners[i][0]
            #     corners_array[i] = np.array([
            #         tuple(corners[0].astype(int)),  # top_left
            #         tuple(corners[1].astype(int)),  # top_right 
            #         tuple(corners[2].astype(int)),  # bottom_right
            #         tuple(corners[3].astype(int))   # bottom_left
            #     ])
            
            # return corners_array.reshape(num_markers, 4, 2)
            return marker_corners, marker_ids
        else:
            return None


    def find_homography(self, src_points, dst_points):
        '''
        return:
            transform_matrix: 원본이미지(frame) 좌표계 -> charuco 좌표계(self.MARKER_LENGTH = 2cm)
        '''
        if src_points is None:
            return None
        
        try:
            # src_points와 dst_points를 2D 형태로 변환
            src_points = np.array(src_points, dtype=np.float32).reshape(-1, 2)  # (N*4, 2) 형태로 변환
            dst_points = np.array(dst_points, dtype=np.float32).reshape(-1, 2)  # (N*4, 2) 형태로 변환
            
            # 디버깅을 위한 출력
            # print("src_points shape:", src_points.shape)
            # print("dst_points shape:", dst_points.shape)
            
            transform_matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
            return transform_matrix
        except cv2.error as e:
            print("Homography 계산 중 오류 발생:", e)
            return None
        except Exception as e:
            print("예상치 못한 오류 발생:", e)
            return None
        
def save_aruco_marker():
    # 마커 크기 (픽셀 단위)
    marker_size = 200
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    # 여러 마커 생성 및 저장
    for marker_id in range(10):  # 0부터 9까지 10개의 마커 생성
        marker_image = aruco.generateImageMarker(dictionary, marker_id, marker_size)
        cv2.imwrite(f"aruco_marker_{marker_id}.png", marker_image)

    print("마커가 생성되어 저장되었습니다.")

def save_charuco_boards():
    # A4 해상도 설정 (300 DPI 기준)
    a4_width, a4_height = 2480, 3508

    # 보드 설정 값 (필요에 따라 조정)
    board_settings = [
        {'squaresX': 5, 'squaresY': 7, 'squareLength': 60, 'markerLength': 45, 'start_id': 0},
        {'squaresX': 5, 'squaresY': 7, 'squareLength': 60, 'markerLength': 45, 'start_id': 20},
        {'squaresX': 5, 'squaresY': 7, 'squareLength': 60, 'markerLength': 45, 'start_id': 40},
        {'squaresX': 5, 'squaresY': 7, 'squareLength': 60, 'markerLength': 45, 'start_id': 60}
    ]
    # 각기 다른 보드 생성 및 저장
    for idx, setting in enumerate(board_settings):
        squaresX = setting['squaresX']
        squaresY = setting['squaresY']
        squareLength = setting['squareLength']
        markerLength = setting['markerLength']
        start_id = setting['start_id']
        
        # ArUco 딕셔너리 생성 (DICT_4X4_100 사용)
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
        
        # 새로운 딕셔너리 생성
        new_dict = aruco.Dictionary(
            _markerSize=4,  # 마커의 X/Y 차원 (DICT_4X4의 경우 4)
            bytesList=np.empty(shape = (squaresX * squaresY, 2, 4), dtype = np.uint8)  # shape을 (2,4)로 변경
        )
        
        # 기존 딕셔너리에서 새로운 ID로 마커 복사
        for i in range(squaresX * squaresY):
            marker_id = start_id + i
            if marker_id < aruco_dict.bytesList.shape[0]:
                new_dict.bytesList[i] = aruco_dict.bytesList[marker_id]
        
        # ChArUco 보드 생성 (새로운 딕셔너리 사용)
        charuco_board = aruco.CharucoBoard(
           (squaresX, squaresY), squareLength, markerLength, new_dict
        )
        
        # 보드 이미지 생성
        board_image = charuco_board.generateImage((a4_width, a4_height))
        
        # 이미지 저장
        filename = f"charuco_board_{idx + 1}.png"
        cv2.imwrite(filename, board_image)
        print(f"{filename} 저장 완료")

if __name__ == "__main__":
    # print("opening camera...")
    # cap = cv2.VideoCapture(1)
    # print("start")

    cap = cv2.VideoCapture(1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    print(f"원본 비디오 크기: {width}x{height}, total_frames: {total_frames}")
            
    if not cap.isOpened():
        print("카메라를 열 수 없습니다!")
        exit()

    # marker_detector = MarkerDetector(debug=True)
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     result = marker_detector.detect_marker(frame.copy())
    #     if result is not None:
    #         top_left, top_right, bottom_left, bottom_right = result
    #         print(f"top_left: {top_left}, top_right: {top_right}, bottom_left: {bottom_left}, bottom_right: {bottom_right}")
    #     # 'q' 키를 누르면 종료
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #     elif cv2.waitKey(1) & 0xFF == ord('s'):
    #         cv2.imwrite("exp_marker.png", frame)

    charuco_detector = CharucoDetector(debug=True)
    # 비디오 저장을 위한 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 비디오의 시작 위치로 되돌림
            break
            
        # 프레임 저장
        out.write(frame)
        
        marker_corners, marker_ids = charuco_detector.detect_aruco(frame)
        try:    
            transform_matrix = charuco_detector.find_homography(marker_corners, charuco_detector.dst_points_aruco[marker_ids])
        except Exception as e:
            print(f"homography error: {e}")

        if transform_matrix is not None:
            if not hasattr(charuco_detector, 'transform_matrices'):
                charuco_detector.transform_matrices = []
            charuco_detector.transform_matrices.append(transform_matrix)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # while 루프가 끝난 후
    if hasattr(charuco_detector, 'transform_matrices') and len(charuco_detector.transform_matrices) > 0:
        # 모든 변환 행렬의 평균 계산
        avg_transform_matrix = np.mean(charuco_detector.transform_matrices, axis=0)
        print("평균 변환 행렬:")
        print(avg_transform_matrix)
    
    cap.release()
    cv2.destroyAllWindows()