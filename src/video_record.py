import cv2

def main():
    print("opening webcam...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    recording = False
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        cv2.imshow('Webcam', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            if not recording:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
                recording = True
                print("Recording started.")
            else:
                recording = False
                out.release()
                print("Recording stopped.")
        elif key == ord('q'):
            break

        if recording:
            out.write(frame)

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()