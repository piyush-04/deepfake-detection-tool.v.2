import torch
from transformers import AutoTokenizer
from modalities.text.model import TextForensicsModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = TextForensicsModel().to(DEVICE)
model.load_state_dict(torch.load("text_model.pt", map_location=DEVICE))
model.eval()

def analyze_text(text: str) -> float:
    batch = tok(text, return_tensors="pt", truncation=True).to(DEVICE)
    with torch.no_grad():
        logit = model(batch["input_ids"], batch["attention_mask"])
        return torch.sigmoid(logit).item()




