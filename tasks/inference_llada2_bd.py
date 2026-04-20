import argparse
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./separater_expert_model/global_step_47501")
    parser.add_argument("--tokenizer-path", type=str, default="./tokenizers/icoder")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing question files")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory to save result files")
    parser.add_argument("--gen-length", type=int, default=2048)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, device_map=device, torch_dtype=torch.bfloat16
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    os.makedirs(args.output_dir, exist_ok=True)

    question_files = sorted(
        f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))
    )
    print(f"Found {len(question_files)} question files in {args.input_dir}")

    for filename in question_files:
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)

        with open(input_path, "r") as f:
            prompt = f.read()

        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")

        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            interleaved=True,
        )
        generated_tokens = model.generate(
            inputs=input_ids,
            eos_early_stop=True,
            gen_length=args.gen_length,
            block_length=args.block_length,
            steps=args.steps,
            # Quality mode
            threshold=0.7,
            editing_threshold=0.5,
            temperature=args.temperature,
        )
        generated_answer = tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=False,
        )

        with open(output_path, "w") as f:
            f.write(generated_answer)

        print(f"Saved result to {output_path}")

    print(f"\nAll done. {len(question_files)} results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
