def verdict_from_scores(image_score, audio_score, text_score=None):
    """
    Scores are probabilities in [0,1]
    Convention:
      0 → real
      1 → fake
    """

    TH = 0.5

    image_fake = image_score > TH
    audio_fake = audio_score > TH
    text_fake = text_score > TH if text_score is not None else False

    # ---- main logic ----
    if image_fake and audio_fake:
        return {
            "verdict": "Fully Synthetic",
            "confidence": max(image_score, audio_score),
            "reason": "Both visual and audio streams show synthetic artifacts"
        }

    if image_fake and not audio_fake:
        return {
            "verdict": "Video Deepfake",
            "confidence": image_score,
            "reason": "Visual manipulation detected, audio appears authentic"
        }

    if not image_fake and audio_fake:
        return {
            "verdict": "Audio Deepfake",
            "confidence": audio_score,
            "reason": "Synthetic voice detected with authentic visuals"
        }

    if not image_fake and not audio_fake:
        if text_fake:
            return {
                "verdict": "Scripted / AI-Assisted Content",
                "confidence": text_score,
                "reason": "Language patterns indicate AI-generated script"
            }
        return {
            "verdict": "Likely Authentic",
            "confidence": 1 - max(image_score, audio_score),
            "reason": "No significant synthetic artifacts detected"
        }
