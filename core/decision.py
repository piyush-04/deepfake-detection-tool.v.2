def decide(scores: dict):
    """
    scores = {"text":0.8, "audio":0.2, ...}
    """
    if not scores:
        return "NO_EVIDENCE"

    if any(v > 0.7 for v in scores.values()):
        return "LIKELY_SYNTHETIC"

    if all(v < 0.4 for v in scores.values()):
        return "LIKELY_AUTHENTIC"

    return "UNCERTAIN"



