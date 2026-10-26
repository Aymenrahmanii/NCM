"""Phase 4: autoregressive baselines with the same prepare -> run -> finish interface as NCM.

Hugging Face `generate` uses the KV cache by default, so the baseline gets its normal speedups.
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PRECISIONS = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

BASELINES = {
    "xsum": "morenolq/bart-base-xsum",       # BART-base fine-tuned on XSum, 140M
    "wmt14": "Helsinki-NLP/opus-mt-de-en",   # MarianMT De->En, 74M
}


class HFSeq2Seq:
    def __init__(self, task: str, model_id: str | None = None, device: str = "cuda",
                 precision: str = "fp32", batch_size: int = 16, max_new_tokens: int = 64,
                 num_beams: int = 1, max_input_tokens: int = 512):
        self.task = task
        self.model_id = model_id or BASELINES[task]
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.max_input_tokens = max_input_tokens
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id, torch_dtype=PRECISIONS[precision]).to(device).eval()

    def prepare(self, texts: list[str]):
        return self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=self.max_input_tokens).to(self.device)

    @torch.no_grad()
    def run(self, inputs) -> torch.Tensor:
        return self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                   num_beams=self.num_beams, do_sample=False, use_cache=True)

    def finish(self, out: torch.Tensor) -> list[str]:
        return [s.strip() for s in self.tok.batch_decode(out, skip_special_tokens=True)]

    def generate(self, texts: list[str]) -> list[str]:
        res = []
        for i in range(0, len(texts), self.batch_size):
            res += self.finish(self.run(self.prepare(texts[i : i + self.batch_size])))
        return res

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
