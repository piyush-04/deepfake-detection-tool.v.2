import cv2
from modalities.image.infer import analyze_image

def analyze_video(path, fps=1):
    cap = cv2.VideoCapture(path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(video_fps // fps) if video_fps > 0 else 1

    scores = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            score = analyze_image(frame)
            scores.append(score)

        idx += 1

    cap.release()

    if not scores:
        return {"score": 0.0, "frames": 0}

    return {
        "score": sum(scores) / len(scores),
        "frames": len(scores),
        "raw_scores": scores
    }



