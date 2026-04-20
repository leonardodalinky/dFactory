"""Runner for iCoder (LLaDA2 block diffusion) models."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lcb_runner.runner.base_runner import BaseRunner


class iCoderRunner(BaseRunner):
    def __init__(self, args, model):
        super().__init__(args, model)
        model_path = args.local_model_path or model.model_name
        tokenizer_path = args.tokenizer_path or model_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        self.model_instance = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model_instance.eval()

        self.gen_length = args.gen_length
        self.block_length = args.block_length
        self.diffusion_steps = args.diffusion_steps
        self.threshold = args.icoder_threshold
        self.editing_threshold = args.editing_threshold

    def _run_single(self, prompt: list[dict[str, str]]) -> list[str]:
        input_ids = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            interleaved=True,
            tokenize=True,
            return_tensors="pt",
        ).to(self.model_instance.device)

        outputs = []
        for _ in range(self.args.n):
            with torch.no_grad():
                generated = self.model_instance.generate(
                    inputs=input_ids,
                    eos_early_stop=True,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    steps=self.diffusion_steps,
                    threshold=self.threshold,
                    editing_threshold=self.editing_threshold,
                    temperature=self.args.temperature,
                )
            raw_text = self.tokenizer.decode(
                generated[0], skip_special_tokens=False
            )
            outputs.append(raw_text)

        return outputs
