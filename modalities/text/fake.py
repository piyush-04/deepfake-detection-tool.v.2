from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class TextFaker:
    def __init__(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        self.model.eval()

        # GPT-2 has no pad token by default
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = device
        self.MAX_CONTEXT = 900      # < 1024 (safe margin)
        self.MAX_NEW_TOKENS = 80

    @torch.no_grad()
    def generate(self, text: str) -> str:
        # Tokenize and truncate hard
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.MAX_CONTEXT,
            return_tensors="pt"
        ).to(self.device)

        output = self.model.generate(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            max_new_tokens=self.MAX_NEW_TOKENS,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id
        )

        return self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )



