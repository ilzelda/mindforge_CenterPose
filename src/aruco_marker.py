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
        self.ARUCO_DICT = cv2.aruco.DICT_6X6_250
        self.SQUARES_VERTICALLY = 7
        self.SQUARES_HORIZONTALLY = 5
        self.SQUARE_LENGTH = 0.02
        self.MARKER_LENGTH = 0.01
        self.LENGTH_PX = 640   # total length of the page in pixels
        self.MARGIN_PX = 20    # size of the margin in pixels

        self.dictionary = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT)
        self.board = cv2.aruco.CharucoBoard((self.SQUARES_VERTICALLY, self.SQUARES_HORIZONTALLY), self.SQUARE_LENGTH, self.MARKER_LENGTH, self.dictionary)
        
        self.detector_param = cv2.aruco.DetectorParameters()

        self.PIXEL_SIZE = 100  # 각 정사각형의 픽셀 크기
        
        w_offset = 340
        h_offset = 240
        self.dst_points = []
        for y in range(self.board.getChessboardSize()[1]-1):
            for x in range(self.board.getChessboardSize()[0]-1):
                self.dst_points.append([
                        w_offset + self.PIXEL_SIZE + self.PIXEL_SIZE*x,
                        h_offset + self.PIXEL_SIZE + self.PIXEL_SIZE*y
                    ])
        self.dst_points = np.array(self.dst_points, dtype=np.float32)

        self.dst_points_aruco = []
        for i in range(2):
            for j in range(2):
                self.dst_points_aruco.append([w_offset + self.PIXEL_SIZE*i, h_offset + self.PIXEL_SIZE*j])
        self.dst_points_aruco = np.array(self.dst_points_aruco, dtype=np.float32)
        
    def detect_charuco(self, frame):
        '''
        return:
            src_points: 원본이미지 좌표계에서의 charuco 좌표
        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.detector_param)
        if self.debug:
            print("검출된 마커 수:", len(marker_ids) if marker_ids is not None else 0)
        
        if marker_ids is not None:
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, self.board)
            if self.debug:
                print("검출된 ChArUco 코너 수:", len(charuco_ids) if charuco_ids is not None else 0)
            
            if charuco_ids is not None and len(charuco_ids) >= 4:
                src_points = charuco_corners.reshape(-1, 2)
                return src_points
            else:
                return None # 최소 4개의 charuco코너가 검출되지 않으면 None 반환
        else:
            return None # aruco 마커가 검출되지 않으면 None 반환

    def detect_aruco(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.detector_param)
            
        if marker_ids is not None:
            top_left = tuple(marker_corners[0][0][0].astype(int))
            top_right = tuple(marker_corners[0][0][1].astype(int))
            bottom_right = tuple(marker_corners[0][0][2].astype(int))
            bottom_left = tuple(marker_corners[0][0][3].astype(int))
            return np.array([top_left, top_right, bottom_left, bottom_right])
        else:
            return None


    def find_homography(self, src_points, dst_points):
        '''
        return:
            transform_matrix: 원본이미지(frame) 좌표계 -> charuco 좌표계(self.PIXEL_SIZE = 2cm)
        '''
        if src_points is None:
            return None
        
        # src_points와 dst_points의 개수를 맞춤
        num_points = min(len(src_points), len(self.dst_points))
        src_points = src_points[:num_points]
        dst_points = dst_points[:num_points]
        
        # 디버깅을 위한 출력
        # print("src_points shape:", src_points.shape) # 4,2
        # print("dst_points shape:", dst_points.shape) # 4,2
        
        try:
            transform_matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
            return transform_matrix
        except cv2.error as e:
            print("Homography 계산 중 오류 발생:", e)
            return None

if __name__ == "__main__":
    # # 마커 크기 (픽셀 단위)
    # marker_size = 200

    # # 여러 마커 생성 및 저장
    # for marker_id in range(10):  # 0부터 9까지 10개의 마커 생성
    #     marker_image = aruco.generateImageMarker(dictionary, marker_id, marker_size)
    #     cv2.imwrite(f"aruco_marker_{marker_id}.png", marker_image)

    # print("마커가 생성되어 저장되었습니다.")

    cap = cv2.VideoCapture(0)
    print("camera open")
    marker_detector = MarkerDetector(debug=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = marker_detector.detect_marker(frame.copy())
        if result is not None:
            top_left, top_right, bottom_left, bottom_right = result
            print(f"top_left: {top_left}, top_right: {top_right}, bottom_left: {bottom_left}, bottom_right: {bottom_right}")
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("exp_marker.png", frame)

    cap.release()
    cv2.destroyAllWindows()