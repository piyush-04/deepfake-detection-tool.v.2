import torch
from torchvision import transforms
from modalities.image.stream import stream_real_images
from modalities.image.fake import generate_fake_image
from modalities.image.model import ImageForensicsModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = ImageForensicsModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4)
loss_fn = torch.nn.BCEWithLogitsLoss()

# 🔥 PIL → Tensor happens HERE (single place)
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])


stream = stream_real_images()

for step in range(300):
    # PIL images
    real_img = next(stream)
    fake_img = generate_fake_image(real_img)

    # Convert to tensors
    real = transform(real_img).unsqueeze(0).to(DEVICE)
    fake = transform(fake_img).unsqueeze(0).to(DEVICE)

    x = torch.cat([real, fake], dim=0)   # [2, 3, 224, 224]

    # 🔥 Correct label shape
    y = torch.tensor([[0.0], [1.0]], device=DEVICE)

    logits = model(x)                    # [2, 1]
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"[IMAGE] step {step} loss {loss.item():.4f}")

torch.save(model.state_dict(), "image_model.pt")
print("Saved image_model.pt")



