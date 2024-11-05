import cv2
import cv2.aruco as aruco


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
        result = marker_detector.detect_marker(frame)
        if result is not None:
            top_left, top_right, bottom_left, bottom_right = result
            print(f"top_left: {top_left}, top_right: {top_right}, bottom_left: {bottom_left}, bottom_right: {bottom_right}")
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()