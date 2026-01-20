import torch
import torchaudio
import torch.nn.functional as F
from modalities.audio.model import AudioForensicsModel
from modalities.audio.stream import stream_real_audio
from modalities.audio.fake import generate_fake_audio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Configuration
# =========================
SAMPLE_RATE = 16000
TARGET_SECONDS = 2
TARGET_LEN = SAMPLE_RATE * TARGET_SECONDS
STEPS = 300
LR = 1e-4

# =========================
# Utilities
# =========================
def crop_or_pad(waveform, target_len):
    """
    waveform: Tensor [1, T]
    returns:  Tensor [1, target_len]
    """
    T = waveform.shape[1]

    if T > target_len:
        return waveform[:, :target_len]
    elif T < target_len:
        pad = target_len - T
        return F.pad(waveform, (0, pad))
    return waveform


# =========================
# Model / Optimizer / Loss
# =========================
model = AudioForensicsModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.head.parameters(), lr=LR)
loss_fn = torch.nn.BCEWithLogitsLoss()

# =========================
# Data Stream
# =========================
stream = stream_real_audio()

# =========================
# Training Loop
# =========================
for step in range(STEPS):
    # ---- real audio ----
    try:
        real_wave, sr = next(stream)
    except StopIteration:
        stream = stream_real_audio()
        real_wave, sr = next(stream)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
        orig_freq=sr,
        new_freq=SAMPLE_RATE
    )
    real_wave = resampler(real_wave)

    # ---- fake audio ----
    fake_wave = generate_fake_audio(real_wave, sr)

    # ---- normalize length ----
    real_wave = crop_or_pad(real_wave, TARGET_LEN)
    fake_wave = crop_or_pad(fake_wave, TARGET_LEN)

    # ---- batch ----
    x = torch.cat([real_wave, fake_wave], dim=0).to(DEVICE)  # [2, T]

    # labels: 0 = real, 1 = fake
    y = torch.tensor([[0.0], [1.0]], device=DEVICE)

    # ---- forward ----
    logits = model(x)
    loss = loss_fn(logits, y)

    # ---- backward ----
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"[AUDIO] step {step} loss {loss.item():.4f}")

# =========================
# Save Model
# =========================
torch.save(model.state_dict(), "audio_model.pt")
print("Saved audio_model.pt")

# Clean shutdown (avoids HF iterator warnings)
del stream



