def build_report(scores, verdict):
    return {
        "verdict": verdict,
        "scores": scores,
        "confidence": max(scores.values()) if scores else 0.0
    }



