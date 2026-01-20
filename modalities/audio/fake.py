import torch

def generate_fake_audio(waveform, sr):
    stretched = torch.nn.functional.interpolate(
        waveform.unsqueeze(0),
        scale_factor=1.03,
        mode="linear",
        align_corners=False
    ).squeeze(0)

    noise = torch.randn_like(stretched) * 0.004
    return stretched + noise



