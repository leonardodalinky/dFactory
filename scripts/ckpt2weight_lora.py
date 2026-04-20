"""Convert a LoRA DCP checkpoint to merged HuggingFace weights.

Usage:
    python scripts/ckpt2weight_lora.py \
        --ckpt_path ./llada2_mini_bd_sft_lora_outputs/checkpoints/global_step_1000 \
        --output_dir ./merged_hf_weights \
        --model_assets_dir ./llada2_mini_bd_sft_lora_outputs/model_assets \
        --lora_alpha 256 --lora_rank 128
"""

import argparse
import os
import shutil

import torch
from veomni.checkpoint import ckpt_to_state_dict
from veomni.models import save_model_weights


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA checkpoint into HuggingFace weights")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the DCP checkpoint directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for merged HF weights")
    parser.add_argument(
        "--model_assets_dir", type=str, default=None,
        help="Path to model_assets directory (config + tokenizer). "
             "Defaults to <ckpt_path>/../../model_assets",
    )
    parser.add_argument("--lora_alpha", type=float, default=256.0)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--ckpt_manager", type=str, default="dcp", choices=["dcp", "omnistore", "bytecheckpoint"])
    args = parser.parse_args()

    # Resolve model_assets_dir: default to output_dir/../../model_assets
    if args.model_assets_dir is None:
        args.model_assets_dir = os.path.join(os.path.dirname(os.path.dirname(args.ckpt_path)), "model_assets")
    if not os.path.isdir(args.model_assets_dir):
        print(f"Warning: model_assets_dir not found at {args.model_assets_dir}, skipping asset copy")
        args.model_assets_dir = None

    # Step 1: Load DCP checkpoint into state dict
    print(f"Loading checkpoint from {args.ckpt_path} ...")
    # ckpt_to_state_dict needs an output_dir for some backends; use a temp location
    model_state_dict = ckpt_to_state_dict(
        save_checkpoint_path=args.ckpt_path,
        output_dir=args.output_dir,
        ckpt_manager=args.ckpt_manager,
    )
    print(f"Loaded state dict with {len(model_state_dict)} keys")

    # Step 2: Merge LoRA deltas into base weights
    scaling = args.lora_alpha / args.lora_rank
    lora_a_suffix = ".lora_A.default.weight"
    lora_b_suffix = ".lora_B.default.weight"
    lora_a_keys = [k for k in model_state_dict if k.endswith(lora_a_suffix)]
    print(f"Found {len(lora_a_keys)} LoRA adapter pairs to merge (scaling={scaling})")

    for a_key in lora_a_keys:
        prefix = a_key[: -len(lora_a_suffix)]
        b_key = prefix + lora_b_suffix
        base_key = prefix + ".base_layer.weight"
        assert b_key in model_state_dict, f"LoRA merge: missing lora_B key {b_key}"
        assert base_key in model_state_dict, f"LoRA merge: missing base weight key {base_key}"
        lora_a = model_state_dict[a_key].float()
        lora_b = model_state_dict[b_key].float()
        model_state_dict[base_key] = model_state_dict[base_key].float() + scaling * (lora_b @ lora_a)
        model_state_dict[base_key] = model_state_dict[base_key].to(torch.bfloat16)

    # Step 3: Rename base_layer keys back to original names, drop lora keys
    merged_state_dict = {}
    for k, v in model_state_dict.items():
        if "lora_A" in k or "lora_B" in k:
            continue
        new_key = k.replace(".base_layer.", ".")
        merged_state_dict[new_key] = v
    print(f"LoRA merge done. Final state dict has {len(merged_state_dict)} keys")

    del model_state_dict

    # Step 4: Save merged weights
    os.makedirs(args.output_dir, exist_ok=True)
    save_model_weights(args.output_dir, merged_state_dict)
    print(f"Saved merged weights to {args.output_dir}")

    # Step 5: Copy model config and tokenizer files from model_assets
    if args.model_assets_dir:
        for fname in os.listdir(args.model_assets_dir):
            src = os.path.join(args.model_assets_dir, fname)
            dst = os.path.join(args.output_dir, fname)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
        print(f"Copied model assets from {args.model_assets_dir}")

    print("Done!")


if __name__ == "__main__":
    main()
