import torch
import torchaudio
from modalities.audio.model import AudioForensicsModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = AudioForensicsModel().to(DEVICE)
model.load_state_dict(torch.load("audio_model.pt", map_location=DEVICE))
model.eval()

def analyze_audio(path: str) -> float:
    waveform, sr = torchaudio.load(path)
    waveform = waveform.to(DEVICE)

    with torch.no_grad():
        logit = model(waveform)
    return torch.sigmoid(logit).mean().item()



