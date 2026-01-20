import mimetypes

def route_input(path_or_text):
    if isinstance(path_or_text, str) and "\n" in path_or_text:
        return ["text"]

    mime, _ = mimetypes.guess_type(path_or_text)
    if mime is None:
        return []

    if mime.startswith("image"):
        return ["image"]
    if mime.startswith("audio"):
        return ["audio"]
    if mime.startswith("video"):
        return ["video"]
    if mime.startswith("text"):
        return ["text"]

    return []



