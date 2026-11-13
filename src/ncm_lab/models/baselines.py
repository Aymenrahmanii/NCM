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
                 num_beams: int = 1, max_input_tokens: int = 512, compiled: bool = False):
        self.task = task
        self.model_id = model_id or BASELINES[task]
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.max_input_tokens = max_input_tokens
        self.compiled = compiled
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id, torch_dtype=PRECISIONS[precision]).to(device).eval()
        if compiled:
            # A static cache makes generate() compile the decoding step (CUDA graphs).
            self.model.generation_config.cache_implementation = "static"
            import torch._dynamo.config as dynamo_config

            # Raise instead of silently timing eager code once the graph budget runs out.
            dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, 128)
            dynamo_config.fail_on_recompile_limit_hit = True

    def prepare(self, texts: list[str]):
        # Compiled graphs are shape-specialized: bucket source lengths to limit recompiles.
        bucket = 64 if self.compiled else None
        return self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=self.max_input_tokens,
                        pad_to_multiple_of=bucket).to(self.device)

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


class HFCausalLM:
    """Unconditional samples from a causal LM, with the prepare -> run -> finish interface.

    Pure ancestral sampling (temperature 1, no top-k / top-p), the setting of the released
    unconditional NCM checkpoints. `forced=True` writes exactly `length` tokens (EOS is
    suppressed) so latency can be measured at a fixed output length; otherwise generation
    stops at EOS like a normal sample.
    """

    def __init__(self, model_id: str, length: int = 128, device: str = "cuda",
                 precision: str = "bf16", forced: bool = False, compiled: bool = False):
        from transformers import AutoModelForCausalLM

        self.model_id, self.length, self.device = model_id, length, device
        self.forced, self.compiled = forced, compiled
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=PRECISIONS[precision]).to(device).eval()
        bos = self.tok.bos_token_id
        self.bos = bos if bos is not None else self.tok.eos_token_id
        if compiled:
            self.model.generation_config.cache_implementation = "static"
            import torch._dynamo.config as dynamo_config

            dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, 128)
            dynamo_config.fail_on_recompile_limit_hit = True

    def prepare(self, items: list) -> torch.Tensor:
        return torch.full((len(items), 1), self.bos, device=self.device)

    @torch.no_grad()
    def run(self, ids: torch.Tensor) -> torch.Tensor:
        n = self.length
        return self.model.generate(
            ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=1.0,
            top_k=0, top_p=1.0, max_new_tokens=n, min_new_tokens=n if self.forced else 0,
            pad_token_id=self.tok.eos_token_id)[:, 1:]

    def finish(self, out: torch.Tensor) -> list[str]:
        texts = []
        for row in out.tolist():
            if not self.forced and self.tok.eos_token_id in row:
                row = row[: row.index(self.tok.eos_token_id)]
            texts.append(self.tok.decode(row, skip_special_tokens=True).strip())
        return texts

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
