import cv2

# 웹캠 열기
cap = cv2.VideoCapture(1)  # 0번 카메라 (기본 장치)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 프레임 처리 (예: 흑백 변환)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 프레임 출력
    cv2.imshow("Original", frame)
    cv2.imshow("Gray", gray_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
