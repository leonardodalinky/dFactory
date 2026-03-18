import json
import os
import sys
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch


# Make VeOmni importable when running from repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
VEOMNI_ROOT = REPO_ROOT / "VeOmni"
if str(VEOMNI_ROOT) not in sys.path:
    sys.path.insert(0, str(VEOMNI_ROOT))

from veomni.models import build_tokenizer
from veomni.data import build_iterative_dataset, build_mapping_dataset
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args
from veomni.models.registry import ModelRegistry

ModelRegistry.register_modeling_path("models.llada2_moe")

from dataset.data_transform import (
    apply_chat_template_mdm,
    process_mdm_sft_example,
    process_mdm_tokenized_example,
)
from dataset import build_local_dataset


@dataclass
class LLaDA2ModelArguments(ModelArguments):
    attn_implementation: Optional[Literal["eager", "sdpa", "flex_attention"]] = field(
        default="sdpa",
        metadata={"help": "Attention implementation to use."},
    )


@dataclass
class LLaDA2DataArguments(DataArguments):
    data_type: Literal["conversation", "tokenid"] = field(
        default="conversation",
        metadata={"help": "Type of the training data."},
    )
    datasets_type: Literal["mapping", "local", "iterable"] = field(
        default="mapping",
        metadata={"help": "Type of the datasets."},
    )
    text_keys: str = field(
        default="messages",
        metadata={"help": "Key to get text from the training data."},
    )
    noise_range_low: float = field(
        default=0.3,
        metadata={"help": "Noise level for random flip input_ids to mask_ids"}
    )
    noise_range_high: float = field(
        default=0.8,
        metadata={"help": "Noise level for random flip input_ids to mask_ids"}
    )

    def __post_init__(self):
        super().__post_init__()
        if self.noise_range_low > self.noise_range_high:
            raise ValueError(
                f"noise_range_low ({self.noise_range_low}) "
                f"cannot be greater than noise_range_high ({self.noise_range_high})."
            )

        if not (0.0 <= self.noise_range_low <= 1.0):
            raise ValueError(
                f"noise_range_low must be between 0.0 and 1.0, but got {self.noise_range_low}."
            )

        if not (0.0 <= self.noise_range_high <= 1.0):
            raise ValueError(
                f"noise_range_high must be between 0.0 and 1.0, but got {self.noise_range_high}."
            )


@dataclass
class LLaDA2TrainingArguments(TrainingArguments):
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW optimizer beta1."},
    )
    beta2: float = field(
        default=0.999,
        metadata={"help": "AdamW optimizer beta2"},
    )
    block_diffusion_mode: bool = field(
        default=False,
        metadata={"help": "If train MDM in block_diffusion mode. True: use block_diffusion, False: full_attention"}
    )
    block_size: int = field(
        default=32,
        metadata={"help": "The block size for block diffusion block size"}
    )
    same_token_labels: bool = field(
        default=False,
        metadata={"help": "If use same token location labels. True: no shift, False: use next-token prediction shift."}
    )
    log_steps: int = field(
        default=1,
        metadata={"help": "Logging interval in steps."},
    )


@dataclass
class DebugArguments:
    namespace: str = field(
        default="train",
        metadata={"help": "Dataset split for local dataset mode."},
    )
    n_samples: int = field(
        default=3,
        metadata={"help": "How many examples to inspect."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducible noise."},
    )
    show_tokens: bool = field(
        default=False,
        metadata={"help": "Show per-token strings for first print_tokens tokens."},
    )
    print_tokens: int = field(
        default=50,
        metadata={"help": "How many tokens to print."},
    )
    mask_token_id: int = field(
        default=156895,
        metadata={"help": "Mask token id used in transform."},
    )


@dataclass
class Arguments:
    model: "LLaDA2ModelArguments" = field(default_factory=LLaDA2ModelArguments)
    data: "LLaDA2DataArguments" = field(default_factory=LLaDA2DataArguments)
    train: "LLaDA2TrainingArguments" = field(default_factory=LLaDA2TrainingArguments)
    debug: "DebugArguments" = field(default_factory=DebugArguments)


def normalize_text_keys(text_keys: str) -> Union[str, List[str]]:
    keys = [x.strip() for x in text_keys.split(",") if x.strip()]
    if not keys:
        raise ValueError("--text_keys is empty after parsing.")
    return keys[0] if len(keys) == 1 else keys


def choose_value(example: Dict[str, Any], text_keys: Union[str, List[str]]) -> Any:
    if isinstance(text_keys, str):
        if text_keys not in example:
            raise KeyError(f"Key '{text_keys}' not found. Available keys: {list(example.keys())}")
        return example[text_keys]
    for key in text_keys:
        if key in example:
            return example[key]
    raise KeyError(f"None of keys {text_keys} found. Available keys: {list(example.keys())}")


def to_list(value: Any) -> List[int]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, list):
        return value
    raise TypeError(f"Unsupported value type for to_list: {type(value)}")


def preview_object(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2) if isinstance(obj, (dict, list)) else str(obj)


def print_tensor_brief(name: str, tensor: torch.Tensor, n: int) -> None:
    arr = tensor.detach().cpu()
    flat = arr.view(-1)
    preview = flat[:n].tolist()
    print(f"{name}: shape={tuple(arr.shape)}, dtype={arr.dtype}, first_{n}={preview}")


def print_decode(tokenizer: Any, ids: Sequence[int], n: int, name: str, show_tokens: bool) -> None:
    part = list(ids[:n])
    decoded = tokenizer.decode(part, skip_special_tokens=False)
    print(f"{name}.decode(first_{n}, keep_special_tokens=True):")
    print(decoded)
    if show_tokens:
        toks = tokenizer.convert_ids_to_tokens(part)
        print(f"{name}.tokens(first_{n}):")
        print(toks)


def inspect_conversation(
    example: Dict[str, Any],
    tokenizer: Any,
    text_keys: Union[str, List[str]],
    max_seq_len: int,
    noise_range: Tuple[float, float],
    mask_token_id: int,
    print_tokens: int,
    show_tokens: bool,
) -> None:
    messages = choose_value(example, text_keys)
    print("[raw messages preview]")
    print(preview_object(messages))

    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    print("[chat template preview]")
    print("full_text:")
    print(full_text)
    print("prompt_text:")
    print(prompt_text)

    input_ids, prompt_length = apply_chat_template_mdm(messages=messages, tokenizer=tokenizer, max_length=max_seq_len)
    ids_list = to_list(input_ids)
    print(f"prompt_length={prompt_length}, seq_len={len(ids_list)}, max_seq_len={max_seq_len}")
    print_decode(tokenizer, ids_list, print_tokens, "input_ids", show_tokens)

    transformed = process_mdm_sft_example(
        example=example,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        text_keys=text_keys,
        noise_range=noise_range,
        mask_token_id=mask_token_id,
    )
    if not transformed:
        print("Transform output is empty.")
        return

    item = transformed[0]
    print("[transform output]")
    print_tensor_brief("input_ids", item["input_ids"], print_tokens)
    print_tensor_brief("noisy_input_ids", item["noisy_input_ids"], print_tokens)
    print_tensor_brief("labels", item["labels"], print_tokens)

    labels = item["labels"].detach().cpu()
    valid_label_positions = (labels != -100).nonzero(as_tuple=False).view(-1).tolist()
    print(f"valid_label_count={len(valid_label_positions)}, first_positions={valid_label_positions[:30]}")

    noisy_ids = to_list(item["noisy_input_ids"])
    print_decode(tokenizer, noisy_ids, print_tokens, "noisy_input_ids", show_tokens)


def inspect_tokenid(
    example: Dict[str, Any],
    tokenizer: Any,
    text_keys: Union[str, List[str]],
    max_seq_len: int,
    noise_range: Tuple[float, float],
    mask_token_id: int,
    print_tokens: int,
    show_tokens: bool,
) -> None:
    raw_ids = choose_value(example, text_keys)
    prompt_lengths = example.get("prompt_lengths", None)

    print("[raw tokenized fields preview]")
    print(f"prompt_lengths={prompt_lengths}")
    print(f"input_ids_len={len(raw_ids)}")
    print(f"input_ids_first_{print_tokens}={raw_ids[:print_tokens]}")
    print_decode(tokenizer, raw_ids, print_tokens, "raw_input_ids", show_tokens)

    transformed = process_mdm_tokenized_example(
        example=example,
        max_seq_len=max_seq_len,
        text_keys=text_keys,
        noise_range=noise_range,
        mask_token_id=mask_token_id,
    )
    if not transformed:
        print("Transform output is empty.")
        return

    item = transformed[0]
    print("[transform output]")
    print_tensor_brief("input_ids", item["input_ids"], print_tokens)
    print_tensor_brief("noisy_input_ids", item["noisy_input_ids"], print_tokens)
    print_tensor_brief("labels", item["labels"], print_tokens)

    labels = item["labels"].detach().cpu()
    valid_label_positions = (labels != -100).nonzero(as_tuple=False).view(-1).tolist()
    print(f"valid_label_count={len(valid_label_positions)}, first_positions={valid_label_positions[:30]}")

    noisy_ids = to_list(item["noisy_input_ids"])
    print_decode(tokenizer, noisy_ids, print_tokens, "noisy_input_ids", show_tokens)


def main() -> None:
    # Enable standalone debug run without torchrun.
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "1")

    args = parse_args(Arguments)
    text_keys = normalize_text_keys(args.data.text_keys)

    torch.manual_seed(args.debug.seed)

    print("=" * 100)
    print("Debug tokenizer script")
    print(f"cwd={os.getcwd()}")
    print("loaded_args:")
    print(json.dumps(asdict(args), ensure_ascii=False, indent=2))
    print(
        f"data_type={args.data.data_type}, datasets_type={args.data.datasets_type}, "
        f"text_keys={text_keys}, max_seq_len={args.data.max_seq_len}"
    )
    print(
        f"noise_range=({args.data.noise_range_low}, {args.data.noise_range_high}), "
        f"mask_token_id={args.debug.mask_token_id}"
    )
    print("=" * 100)

    tokenizer = build_tokenizer(args.model.tokenizer_path)

    if args.data.data_type == "conversation":
        if not tokenizer.chat_template:
            raise ValueError("No chat template found in the tokenizer.")

        transform = partial(
            process_mdm_sft_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
            noise_range=(args.data.noise_range_low, args.data.noise_range_high),
            mask_token_id=args.debug.mask_token_id,
        )
    elif args.data.data_type == "tokenid":
        transform = partial(
            process_mdm_tokenized_example,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
            noise_range=(args.data.noise_range_low, args.data.noise_range_high),
            mask_token_id=args.debug.mask_token_id,
        )
    else:
        raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")

    if args.data.datasets_type == "mapping":
        dataset = build_mapping_dataset(args.data.train_path)
    elif args.data.datasets_type == "local":
        dataset = build_local_dataset(args.data.train_path, namespace=args.debug.namespace)
    elif args.data.datasets_type == "iterable":
        dataset = build_iterative_dataset(args.data.train_path, seed=args.debug.seed)
    else:
        raise NotImplementedError(f"Unsupported datasets_type: {args.data.datasets_type}")

    dataset_len = len(dataset) if hasattr(dataset, "__len__") else None
    print(f"dataset_len={dataset_len}")
    if hasattr(dataset, "column_names"):
        print(f"dataset_columns={list(dataset.column_names)}")
    elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "column_names"):
        print(f"dataset_columns={list(dataset.dataset.column_names)}")

    n = args.debug.n_samples if dataset_len is None else min(args.debug.n_samples, dataset_len)
    if args.data.datasets_type == "iterable":
        iterator = iter(dataset)

    for i in range(n):
        print("\n" + "#" * 40)
        print(f"Sample index: {i}")
        print("#" * 40)
        if args.data.datasets_type == "iterable":
            try:
                example = next(iterator)
            except StopIteration:
                print("Iterable dataset exhausted early.")
                break
        else:
            example = dataset[i]
        print(f"example_keys={list(example.keys())}")

        try:
            if args.data.data_type == "conversation":
                inspect_conversation(
                    example=example,
                    tokenizer=tokenizer,
                    text_keys=text_keys,
                    max_seq_len=args.data.max_seq_len,
                    noise_range=(args.data.noise_range_low, args.data.noise_range_high),
                    mask_token_id=args.debug.mask_token_id,
                    print_tokens=args.debug.print_tokens,
                    show_tokens=args.debug.show_tokens,
                )
            else:
                inspect_tokenid(
                    example=example,
                    tokenizer=tokenizer,
                    text_keys=text_keys,
                    max_seq_len=args.data.max_seq_len,
                    noise_range=(args.data.noise_range_low, args.data.noise_range_high),
                    mask_token_id=args.debug.mask_token_id,
                    print_tokens=args.debug.print_tokens,
                    show_tokens=args.debug.show_tokens,
                )
        except Exception as exc:
            print(f"[ERROR] sample {i} failed: {repr(exc)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
