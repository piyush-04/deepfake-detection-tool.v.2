def localize(scores, threshold=0.6):
    return [i for i,s in enumerate(scores) if s > threshold]



