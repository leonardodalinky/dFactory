import os
import torch
from typing import  Any, Dict, List, Optional, Sequence, Union, Tuple

# Suppress tokenizer length warnings for long conversations that get truncated later
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore", message="Token indices sequence length is longer than")

def process_mdm_sft_example(
    example: Dict[str, Any],
    tokenizer,
    max_seq_len: int,
    text_keys: Union[str, List[str]] = "messages",
    noise_range: Tuple[float, float] = (0.3, 0.8),
    mask_token_id: int = 156895,
    source_name: Optional[str] = None,
) -> List[Dict[str, "torch.Tensor"]]:
    if isinstance(text_keys, str):
        messages = example[text_keys]
    elif isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                messages = example[key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")
    
    examples = []
    input_ids, prompt_length = apply_chat_template_mdm(messages=messages, tokenizer=tokenizer, max_length=max_seq_len)

    labels = input_ids.clone()
    labels[:prompt_length] = -100

    maskable_mask = torch.arange(max_seq_len) >= prompt_length

    noisy_input_ids = sft_noise_transition(
        input_ids.clone(), 
        noise_range=noise_range, 
        maskable_mask=maskable_mask, 
        mask_token_id=mask_token_id
    )

    loss_mask = noisy_input_ids == mask_token_id
    labels[~loss_mask] = -100

    examples.append({
        "input_ids": input_ids,
        "noisy_input_ids": noisy_input_ids,
        "attention_mask": torch.ones_like(input_ids), # 使用torch.ones_like更简洁
        "labels": labels,
    })
    
    return examples


def process_mdm_tokenized_example(
    example: Dict[str, List[int]],
    max_seq_len: int, 
    text_keys: Union[str, List[str]] = "input_ids",
    noise_range: Tuple[float, float] = (0.3, 0.8),
    mask_token_id: int = 156895,
    source_name: Optional[str] = None,
) -> List[Dict[str, "torch.Tensor"]]:
    examples = []
    if isinstance(text_keys, str):
        input_ids = example[text_keys]
    elif isinstance(text_keys, list):
        for text_key in text_keys:
            if text_key in example:
                input_ids = example[text_key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    prompt_length = example['prompt_lengths']

    input_ids = torch.tensor(input_ids)
    labels = input_ids.clone()
    labels[:prompt_length] = -100

    maskable_mask = torch.arange(max_seq_len) >= prompt_length

    noisy_input_ids = sft_noise_transition(
        input_ids.clone(), 
        noise_range=noise_range, 
        maskable_mask=maskable_mask, 
        mask_token_id=mask_token_id
    )

    loss_mask = noisy_input_ids == mask_token_id
    labels[~loss_mask] = -100

    examples.append({
        "input_ids": input_ids,
        "noisy_input_ids": noisy_input_ids,
        "attention_mask": torch.ones_like(input_ids), # 使用torch.ones_like更简洁
        "labels": labels,
    })
    
    return examples


def sft_noise_transition(x_0, noise_range, maskable_mask, mask_token_id):
    """
    Performs a noise transition by masking tokens.

    Args:
        x_0 (torch.Tensor): The input sequence (batch_size, seq_len).
        noise_range (tuple): A tuple (min, max) for the noise range, from which the masking ratio sigma is sampled.
        maskable_mask (torch.Tensor): A boolean mask indicating which positions are allowed to be masked (batch_size, seq_len).
        mask_token_id (int): The ID of the mask token.

    Returns:
        torch.Tensor: The sequence after masking.
    """

    t_tensor = torch.rand(1) * (noise_range[1] - noise_range[0]) + noise_range[0]
    sigma = t_tensor.item()
    # move_chance = 1 - (-sigma).exp()
    move_chance = sigma
    move_indices = (torch.rand(*x_0.shape) < move_chance) & maskable_mask
    x_t = torch.where(move_indices, mask_token_id, x_0)
    return x_t

def apply_chat_template_mdm(messages, tokenizer, max_length):
    inputs_str = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_str = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

    prompt_ids_unpadded = tokenizer(prompt_str, add_special_tokens=False)['input_ids']
    prompt_length = len(prompt_ids_unpadded)

    tokenized_input = tokenizer(
        inputs_str,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
        add_special_tokens=False
    ).input_ids.squeeze(0)

    return tokenized_input, prompt_length


def compute_response_mask(messages, tokenizer, max_length):
    """Compute a per-position boolean mask indicating assistant response tokens.

    Supports multi-turn conversations: every assistant message segment is marked
    as True (response), all other tokens (user messages, system prompts, special
    template tokens between turns) are marked as False (prompt).

    Args:
        messages: list of dicts with 'role' and 'content' keys.
        tokenizer: tokenizer with apply_chat_template support.
        max_length: sequence length (after padding/truncation).

    Returns:
        response_mask: (max_length,) boolean tensor.
    """
    response_mask = torch.zeros(max_length, dtype=torch.bool)

    for turn_idx in range(len(messages)):
        if messages[turn_idx]["role"] != "assistant":
            continue

        # Tokenize everything up to (but not including) this assistant turn
        # with add_generation_prompt=True to include the assistant prompt header
        prefix_str = tokenizer.apply_chat_template(
            messages[:turn_idx], tokenize=False, add_generation_prompt=True
        )
        start = len(tokenizer(prefix_str, add_special_tokens=False)["input_ids"])

        # Tokenize everything up to and including this assistant turn
        through_str = tokenizer.apply_chat_template(
            messages[: turn_idx + 1], tokenize=False
        )
        end = len(tokenizer(through_str, add_special_tokens=False)["input_ids"])

        # Clamp to max_length
        start = min(start, max_length)
        end = min(end, max_length)
        if start < end:
            response_mask[start:end] = True

    return response_mask


def process_mdm_sft_latent_example(
    example: Dict[str, Any],
    tokenizer,
    max_seq_len: int,
    text_keys: Union[str, List[str]] = "messages",
    noise_range: Tuple[float, float] = (0.3, 0.8),
    mask_token_id: int = 156895,
    source_name: Optional[str] = None,
) -> List[Dict[str, "torch.Tensor"]]:
    """Process a conversation example for latent-block diffusion training.

    Similar to process_mdm_sft_example but:
      - Computes a multi-turn aware response_mask instead of scalar prompt_length.
      - Returns response_mask in the output dict for use by the training script.
      - Latent token insertion is NOT done here (handled in the training loop
        for dynamic block_size support).
    """
    if isinstance(text_keys, str):
        messages = example[text_keys]
    elif isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                messages = example[key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    examples = []
    input_ids, _ = apply_chat_template_mdm(
        messages=messages, tokenizer=tokenizer, max_length=max_seq_len
    )

    # Multi-turn aware response mask
    response_mask = compute_response_mask(messages, tokenizer, max_seq_len)

    labels = input_ids.clone()
    labels[~response_mask] = -100  # only response positions can have valid labels

    # maskable_mask = response positions (only response tokens can be noised)
    maskable_mask = response_mask

    noisy_input_ids = sft_noise_transition(
        input_ids.clone(),
        noise_range=noise_range,
        maskable_mask=maskable_mask,
        mask_token_id=mask_token_id,
    )

    # Only positions that were actually masked contribute to the loss
    loss_mask = noisy_input_ids == mask_token_id
    labels[~loss_mask] = -100

    examples.append({
        "input_ids": input_ids,
        "noisy_input_ids": noisy_input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "response_mask": response_mask,
    })

    return examples