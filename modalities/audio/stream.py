import os
import torchaudio
import torch

AUDIO_DIR = "audio_samples"

def stream_real_audio():
    if not os.path.exists(AUDIO_DIR):
        raise RuntimeError(
            "audio_samples folder missing at project root."
        )

    for fname in os.listdir(AUDIO_DIR):
        if not fname.lower().endswith(".wav"):
            continue

        path = os.path.join(AUDIO_DIR, fname)
        waveform, sr = torchaudio.load(path)

        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if waveform.shape[1] >= sr * 2:
            yield waveform, sr



