import cv2

def stream_video_frames(path, stride=10):
    cap = cv2.VideoCapture(path)
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            yield frame
        idx += 1



