import torch
from transformers import CLIPTokenizer, CLIPTextModel, T5TokenizerFast, T5EncoderModel


class TextEncoder:
    def __init__(self, device="cuda", dtype=torch.bfloat16, max_length=256):
        self.device = device
        self.dtype = dtype
        self.max_length = max_length

        self.tokenizer_clip = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.encoder_clip = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype).to(device)

        self.tokenizer_t5 = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl")
        self.encoder_t5 = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl", torch_dtype=dtype).to(device)

        print("Caching null embeddings for classifier-free guidance...")
        with torch.no_grad():
            inputs_clip = self.tokenizer_clip([""], padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
            self.null_pooled = self.encoder_clip(inputs_clip.input_ids).pooler_output
            inputs_t5 = self.tokenizer_t5([""], padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt").to(device)
            self.null_prompt = self.encoder_t5(inputs_t5.input_ids)[0]

    @torch.no_grad()
    def get_null_info(self, batch_size):
        null_prompt = self.null_prompt.expand(batch_size, -1, -1).to(self.device, dtype=self.dtype)
        null_pooled = self.null_pooled.expand(batch_size, -1).to(self.device, dtype=self.dtype)
        null_kwargs = {
            "prompt_embeds": null_prompt,
            "pooled_embeds": null_pooled,
            "res": torch.zeros(batch_size, device=self.device, dtype=self.dtype),
            "lon": torch.zeros(batch_size, device=self.device, dtype=self.dtype),
            "lat": torch.zeros(batch_size, device=self.device, dtype=self.dtype),
        }
        return null_kwargs

    @torch.no_grad()
    def get_null_info_flux2(self, batch_size):
        null_prompt = self.null_prompt.expand(batch_size, -1, -1).to(self.device, dtype=self.dtype)
        null_pooled = self.null_pooled.expand(batch_size, -1).to(self.device, dtype=self.dtype)
        null_kwargs = {
            "ctx": null_prompt,
            "y": null_pooled,
            "res": torch.zeros(batch_size, device=self.device, dtype=self.dtype),
            "lon": torch.zeros(batch_size, device=self.device, dtype=self.dtype),
            "lat": torch.zeros(batch_size, device=self.device, dtype=self.dtype),
        }
        return null_kwargs

    @torch.no_grad()
    def __call__(self, prompt: str | list[str], max_length=256):
        if isinstance(prompt, str):
            prompt = [prompt]

        batch_size = len(prompt)

        prompt_embeds = torch.empty((batch_size, max_length, 4096), device=self.device, dtype=self.dtype)
        pooled_embeds = torch.empty((batch_size, 768), device=self.device, dtype=self.dtype)
        text_ids = torch.zeros((batch_size, max_length, 3), device=self.device, dtype=self.dtype)
        seq_idx = torch.arange(max_length, device=self.device)
        text_ids[:, :, 0] = seq_idx

        valid_indices = [i for i, p in enumerate(prompt) if p != ""]
        empty_indices = [i for i, p in enumerate(prompt) if p == ""]

        if empty_indices:
            prompt_embeds[empty_indices] = self.null_prompt.expand(len(empty_indices), -1, -1)
            pooled_embeds[empty_indices] = self.null_pooled.expand(len(empty_indices), -1)

        if valid_indices:
            valid_prompts = [prompt[i] for i in valid_indices]
            inputs_clip = self.tokenizer_clip(valid_prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(self.device)
            pooled_embeds[valid_indices] = self.encoder_clip(inputs_clip.input_ids).pooler_output
            inputs_t5 = self.tokenizer_t5(valid_prompts, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt").to(self.device)
            prompt_embeds[valid_indices] = self.encoder_t5(inputs_t5.input_ids)[0]

        return prompt_embeds, pooled_embeds, text_ids


# === usage example ===
if __name__ == "__main__":
    extractor = TextEncoder()

    my_prompt = "A high-resolution satellite image of a circular irrigation field in the desert."

    p_emb, pooled_emb, t_ids = extractor(my_prompt)

    print(f"T5 Prompt Embeds (Sequence): {p_emb.shape}")  # [1, 256, 4096]
    print(f"CLIP Pooled Embeds (Global): {pooled_emb.shape}")  # [1, 768]
    print(f"Text IDs (RoPE): {t_ids.shape}")  # [1, 256, 3]