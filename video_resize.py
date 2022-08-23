import cv2

video_path = "./output/shin_vc.avi"

capture = cv2.VideoCapture(video_path)

while True:
    ret, frame = capture.read()

    if not ret:
        break

    resize_frame = cv2.resize(frame, (256, 144), interpolation=cv2.INTER_AREA)

    cv2.imshow("resize_frame", resize_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()

cv2.destoryAllWindows()