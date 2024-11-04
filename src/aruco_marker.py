import cv2
import cv2.aruco as aruco

dictionary = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

# 웹캠 초기화
cap = cv2.VideoCapture(1)
print("camera open")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 마커 검출
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    try:
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            # 각 마커에 대해 ID 표시
            for i, corner in enumerate(corners):
                marker_id = ids[i][0]
                top_left = tuple(corner[0][0].astype(int))
                cv2.putText(frame, str(marker_id), top_left, cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)
    except:
        pass

    cv2.imshow('Frame', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

