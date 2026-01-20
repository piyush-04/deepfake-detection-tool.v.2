import torch
from transformers import AutoTokenizer
from modalities.text.stream import stream_real_text
from modalities.text.fake import TextFaker
from modalities.text.model import TextForensicsModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = TextForensicsModel().to(DEVICE)
faker = TextFaker(DEVICE)

opt = torch.optim.AdamW(model.head.parameters(), lr=2e-4)
loss_fn = torch.nn.BCEWithLogitsLoss()

stream = stream_real_text()

for step in range(300):
    real = next(stream)
    fake = faker.generate(real)

    texts = [real, fake]
    labels = torch.tensor([0,1], dtype=torch.float32).to(DEVICE)

    batch = tok(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    logits = model(batch["input_ids"], batch["attention_mask"])

    loss = loss_fn(logits, labels)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 50 == 0:
        print("step", step, "loss", loss.item())

torch.save(model.state_dict(), "text_model.pt")



