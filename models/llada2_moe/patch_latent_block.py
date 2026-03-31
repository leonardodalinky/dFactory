"""
Latent-Block Diffusion Language Model Patch

Provides utilities for latent token insertion, attention mask construction,
embedding injection, and loss computation for the two-objective latent block
diffusion training scheme.

This module is designed to be non-invasive -- the base LLaDA2 MoE model code
is not modified. All latent block logic is implemented here and consumed by
the training script (train_llada2_bd_latent.py).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Trainable projection modules
# ---------------------------------------------------------------------------


class LatentOutputHead(nn.Module):
    """Bottleneck MLP that projects the model's hidden state at latent positions
    down to the sentence-transformer embedding space for Objective 1 loss.

    Architecture:
        hidden_size -> hidden_size // 2 -> LayerNorm -> GELU -> latent_dim
    """

    def __init__(self, hidden_size: int, latent_dim: int):
        super().__init__()
        bottleneck = hidden_size // 2
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_latents, hidden_size)
        Returns:
            (batch, num_latents, latent_dim)
        """
        return self.mlp(x)


class LatentInputProjector(nn.Module):
    """Projects sentence-transformer embeddings up to the model's hidden size
    for injection as word-embedding replacements at latent positions (Obj 2).

    Architecture:
        latent_dim -> hidden_size
    """

    def __init__(self, latent_dim: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(latent_dim, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_latents, latent_dim)
        Returns:
            (batch, num_latents, hidden_size)
        """
        return self.proj(x)


# ---------------------------------------------------------------------------
# Sentence-transformer ground truth computation
# ---------------------------------------------------------------------------


def compute_latent_ground_truth(
    clean_blocks: torch.Tensor,
    tokenizer,
    sentence_model,
    device: torch.device,
) -> torch.Tensor:
    """Compute sentence-transformer embeddings for each block's content.

    Args:
        clean_blocks: (batch, num_blocks, orig_block_size) token IDs.
        tokenizer: tokenizer for decoding token IDs back to text.
        sentence_model: frozen SentenceTransformer model.
        device: target device for the returned tensor.

    Returns:
        (batch, num_blocks, latent_dim) tensor of ST embeddings.
    """
    B, num_blocks, _ = clean_blocks.shape

    # Decode all blocks to text in one pass
    all_texts: List[str] = []
    for b in range(B):
        for blk in range(num_blocks):
            token_ids = clean_blocks[b, blk].tolist()
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            all_texts.append(text if text.strip() else " ")  # avoid empty string

    # Batch-encode with sentence-transformer (frozen, no grad)
    with torch.no_grad():
        embeddings = sentence_model.encode(
            all_texts,
            batch_size=len(all_texts),
            convert_to_tensor=True,
            show_progress_bar=False,
        )  # (B * num_blocks, latent_dim)

    return embeddings.view(B, num_blocks, -1).to(device)


# ---------------------------------------------------------------------------
# Latent token insertion and coordinate remapping
# ---------------------------------------------------------------------------


def insert_latent_tokens(
    input_ids: torch.Tensor,
    block_size: int,
    latent_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Insert a latent token at the last position of each block.

    Takes raw token sequences and:
      1. Truncates to num_blocks * (block_size - 1) tokens
      2. Reshapes into blocks of (block_size - 1)
      3. Appends latent_token_id at the end of each block

    Args:
        input_ids: (batch, seq_len) raw token IDs (no latent tokens yet).
        block_size: tokens per block (including latent).
        latent_token_id: token ID for <|block_latent|>.

    Returns:
        new_ids: (batch, num_blocks * block_size) with latent tokens inserted.
        clean_blocks: (batch, num_blocks, block_size - 1) original tokens per block.
    """
    B, L = input_ids.shape
    orig_block_size = block_size - 1
    num_blocks = L // block_size  # effective number of blocks
    orig_tokens = num_blocks * orig_block_size

    # Truncate and reshape
    truncated = input_ids[:, :orig_tokens]  # (B, orig_tokens)
    clean_blocks = truncated.view(
        B, num_blocks, orig_block_size
    )  # (B, num_blocks, orig_block_size)

    # Append latent token at the last position of each block
    latent_col = torch.full(
        (B, num_blocks, 1),
        latent_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    blocks_with_latent = torch.cat([clean_blocks, latent_col], dim=2)  # (B, num_blocks, block_size)
    new_ids = blocks_with_latent.reshape(B, num_blocks * block_size)  # (B, max_seq_len)

    return new_ids, clean_blocks


def remap_response_mask(
    response_mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Remap a per-position response_mask from original coordinates to the
    latent-inserted coordinate system.

    Mapping: original position `orig` -> new position
        (orig // (block_size - 1)) * block_size + (orig % (block_size - 1))

    Latent positions (last position of each block) inherit True if ANY token
    in that block is a response token.

    Args:
        response_mask: (batch, orig_seq_len) boolean tensor.
        block_size: tokens per block (including latent).

    Returns:
        (batch, num_blocks * block_size) boolean tensor in the new coordinate system.
    """
    B, L = response_mask.shape
    orig_block_size = block_size - 1
    num_blocks = L // block_size
    orig_tokens = num_blocks * orig_block_size
    new_seq_len = num_blocks * block_size

    # Truncate and reshape into blocks
    truncated = response_mask[:, :orig_tokens]  # (B, orig_tokens)
    blocks = truncated.view(B, num_blocks, orig_block_size)  # (B, num_blocks, orig_block_size)

    # Latent position inherits True if any token in the block is response
    latent_flags = blocks.any(dim=2, keepdim=True)  # (B, num_blocks, 1)

    # Concatenate: [orig tokens | latent flag] per block
    new_blocks = torch.cat([blocks, latent_flags], dim=2)  # (B, num_blocks, block_size)
    return new_blocks.reshape(B, new_seq_len)


def compute_new_prompt_length(orig_prompt_length: int, block_size: int) -> int:
    """Map a scalar prompt_length from original to latent-inserted coordinates.

    Kept for backward compatibility / simple single-turn usage.
    """
    orig_block_size = block_size - 1
    return (orig_prompt_length // orig_block_size) * block_size + (
        orig_prompt_length % orig_block_size
    )


# ---------------------------------------------------------------------------
# Attention mask construction
# ---------------------------------------------------------------------------


def get_latent_positions(num_blocks: int, block_size: int) -> List[int]:
    """Return the list of latent token positions (last pos of each block)."""
    return [(i + 1) * block_size - 1 for i in range(num_blocks)]


def build_obj1_attention_mask(
    max_seq_len: int,
    block_size: int,
    response_masks: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build Objective 1 attention mask for latent generation (S' only).

    Mask values: 0.0 = allow attention, -inf = block attention.

    Allowed attention:
        - prompt <-> prompt
        - latent -> prompt
        - latent <-> latent  (bidirectional among all latents)
    All other positions (non-prompt, non-latent response tokens) are invisible.

    Args:
        max_seq_len: sequence length after latent insertion (= num_blocks * block_size).
        block_size: tokens per block (including latent).
        response_masks: (batch, max_seq_len) boolean. True = response position.
        device: target device.
        dtype: float dtype for the mask.

    Returns:
        (batch, 1, max_seq_len, max_seq_len) attention mask.
    """
    B = response_masks.shape[0]
    num_blocks = max_seq_len // block_size

    # Identify position types
    latent_pos_set = set(get_latent_positions(num_blocks, block_size))
    is_latent = torch.zeros(max_seq_len, dtype=torch.bool, device=device)
    for p in latent_pos_set:
        is_latent[p] = True

    # prompt = not response AND not latent
    prompt_masks = ~response_masks & ~is_latent.unsqueeze(0)  # (B, max_seq_len)

    mask = torch.full((B, 1, max_seq_len, max_seq_len), float("-inf"), dtype=dtype, device=device)

    is_latent_expanded = is_latent.unsqueeze(0).expand(B, -1)  # (B, max_seq_len)

    for b in range(B):
        prompt_idx = prompt_masks[b].nonzero(as_tuple=True)[0]
        latent_idx = is_latent_expanded[b].nonzero(as_tuple=True)[0]

        # prompt <-> prompt
        if prompt_idx.numel() > 0:
            mask[b, 0, prompt_idx.unsqueeze(1), prompt_idx.unsqueeze(0)] = 0.0

        # latent -> prompt
        if latent_idx.numel() > 0 and prompt_idx.numel() > 0:
            mask[b, 0, latent_idx.unsqueeze(1), prompt_idx.unsqueeze(0)] = 0.0

        # latent <-> latent
        if latent_idx.numel() > 0:
            mask[b, 0, latent_idx.unsqueeze(1), latent_idx.unsqueeze(0)] = 0.0

    return mask


def sample_latent_mask(
    num_blocks: int,
    noise_range: Tuple[float, float],
    device: torch.device,
) -> torch.Tensor:
    """Sample a random mask for latent diffusion in Obj1.

    Each latent is independently masked with probability sigma,
    where sigma ~ Uniform(noise_range[0], noise_range[1]).

    Args:
        num_blocks: number of latent tokens.
        noise_range: (low, high) for uniform sampling of mask ratio.
        device: target device.

    Returns:
        (num_blocks,) boolean tensor. True = masked (to predict), False = given as GT.
    """
    sigma = torch.rand(1, device=device) * (noise_range[1] - noise_range[0]) + noise_range[0]
    return torch.rand(num_blocks, device=device) < sigma


def build_obj1_latent_diffusion_mask(
    max_seq_len: int,
    block_size: int,
    num_blocks: int,
    response_masks: torch.Tensor,
    latent_masked: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build Objective 1 attention mask for latent diffusion mode.

    The input sequence is [S' (max_seq_len) | GT_ref (num_unmasked)].
    Masked latents in S' are prediction targets.
    Unmasked latents' GT embeddings are appended as reference conditioning.

    Attention rules:
        - prompt <-> prompt: allowed
        - masked S'_latent -> prompt: allowed
        - masked S'_latent <-> masked S'_latent: allowed (bidirectional)
        - masked S'_latent -> GT_ref: allowed (all unmasked GT visible)
        - unmasked S'_latent: invisible (no query, no key)
        - prompt -> S'_latent / GT: blocked
        - GT positions: never serve as query
        - other response positions: invisible

    Args:
        max_seq_len: S' length (num_blocks * block_size).
        block_size: tokens per block (including latent).
        num_blocks: number of blocks.
        response_masks: (batch, max_seq_len) boolean. True = response position.
        latent_masked: (batch, num_blocks) boolean. True = masked (to predict).
        device: target device.
        dtype: float dtype for the mask.

    Returns:
        (batch, 1, total_len, total_len) mask where total_len = max_seq_len + num_unmasked.
    """
    B = response_masks.shape[0]
    num_unmasked = int((~latent_masked[0]).sum().item())  # same across batch for simplicity
    total_len = max_seq_len + num_unmasked

    latent_pos_list = get_latent_positions(num_blocks, block_size)
    is_latent_sp = torch.zeros(max_seq_len, dtype=torch.bool, device=device)
    for p in latent_pos_list:
        is_latent_sp[p] = True

    prompt_masks_2d = ~response_masks & ~is_latent_sp.unsqueeze(0)  # (B, max_seq_len)

    mask = torch.full((B, 1, total_len, total_len), float("-inf"), dtype=dtype, device=device)

    latent_pos_tensor = torch.tensor(latent_pos_list, device=device)
    gt_ref_start = max_seq_len

    for b in range(B):
        prompt_idx = prompt_masks_2d[b].nonzero(as_tuple=True)[0]
        masked_latent_idx = latent_pos_tensor[latent_masked[b]]  # S' positions of masked latents
        gt_ref_idx = torch.arange(gt_ref_start, gt_ref_start + num_unmasked, device=device)

        # prompt <-> prompt
        if prompt_idx.numel() > 0:
            mask[b, 0, prompt_idx.unsqueeze(1), prompt_idx.unsqueeze(0)] = 0.0

        # masked S'_latent -> prompt
        if masked_latent_idx.numel() > 0 and prompt_idx.numel() > 0:
            mask[b, 0, masked_latent_idx.unsqueeze(1), prompt_idx.unsqueeze(0)] = 0.0

        # masked S'_latent <-> masked S'_latent (bidirectional)
        if masked_latent_idx.numel() > 0:
            mask[b, 0, masked_latent_idx.unsqueeze(1), masked_latent_idx.unsqueeze(0)] = 0.0

        # masked S'_latent -> all GT ref positions
        if masked_latent_idx.numel() > 0 and gt_ref_idx.numel() > 0:
            mask[b, 0, masked_latent_idx.unsqueeze(1), gt_ref_idx.unsqueeze(0)] = 0.0

    return mask


# ---------------------------------------------------------------------------
# Input construction helpers
# ---------------------------------------------------------------------------


def build_noisy_for_obj1(
    clean_with_latent: torch.Tensor,
    response_mask_new: torch.Tensor,
    mask_token_id: int,
    latent_positions: List[int],
    latent_token_id: int = 156901,
) -> torch.Tensor:
    """Build the S' input for Objective 1 (latent generation).

    All non-latent response positions are replaced with <|mask|>.
    Latent positions use <|block_latent|> to explicitly signal their role.
    Prompt positions remain clean.

    Args:
        clean_with_latent: (batch, max_seq_len) clean tokens with latent inserted.
        response_mask_new: (batch, max_seq_len) boolean in latent-inserted coords.
        mask_token_id: <|mask|> token ID.
        latent_positions: list of latent position indices.
        latent_token_id: <|block_latent|> token ID.

    Returns:
        (batch, max_seq_len) noisy input for Obj1.
    """
    noisy = clean_with_latent.clone()
    # Mask all response positions with <|mask|>
    noisy[response_mask_new] = mask_token_id
    # Latent positions use <|block_latent|> instead of <|mask|> so the model
    # knows "this is a latent position" vs "this is a masked response token"
    noisy[:, latent_positions] = latent_token_id
    return noisy


def build_noisy_for_obj2(
    noisy_input_ids: torch.Tensor,
    block_size: int,
    latent_token_id: int,
) -> torch.Tensor:
    """Build the S' input for Objective 2 (token generation | latent).

    Takes the standard noised sequence (without latent tokens), truncates,
    reshapes into blocks, and inserts latent_token_id at the last position.
    The actual ST embedding is injected later via the forward pre-hook.

    Args:
        noisy_input_ids: (batch, seq_len) partially masked token IDs.
        block_size: tokens per block (including latent).
        latent_token_id: placeholder token ID for latent positions.

    Returns:
        (batch, num_blocks * block_size) noisy input with latent placeholders.
    """
    B, L = noisy_input_ids.shape
    orig_block_size = block_size - 1
    num_blocks = L // block_size
    orig_tokens = num_blocks * orig_block_size

    truncated = noisy_input_ids[:, :orig_tokens]
    blocks = truncated.view(B, num_blocks, orig_block_size)
    latent_col = torch.full(
        (B, num_blocks, 1),
        latent_token_id,
        dtype=noisy_input_ids.dtype,
        device=noisy_input_ids.device,
    )
    blocks_with_latent = torch.cat([blocks, latent_col], dim=2)
    return blocks_with_latent.reshape(B, num_blocks * block_size)


def build_labels_with_latent(
    labels: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Insert -100 at latent positions in the label tensor.

    Args:
        labels: (batch, seq_len) label tensor (without latent positions).
        block_size: tokens per block (including latent).

    Returns:
        (batch, num_blocks * block_size) labels with -100 at latent positions.
    """
    B, L = labels.shape
    orig_block_size = block_size - 1
    num_blocks = L // block_size
    orig_tokens = num_blocks * orig_block_size

    truncated = labels[:, :orig_tokens]
    blocks = truncated.view(B, num_blocks, orig_block_size)
    ignore_col = torch.full(
        (B, num_blocks, 1),
        -100,
        dtype=labels.dtype,
        device=labels.device,
    )
    blocks_with_ignore = torch.cat([blocks, ignore_col], dim=2)
    return blocks_with_ignore.reshape(B, num_blocks * block_size)


# ---------------------------------------------------------------------------
# Model patching (non-invasive hooks)
# ---------------------------------------------------------------------------


def patch_model_for_hidden_capture(model) -> None:
    """Register a forward hook on the inner LLaDA2MoeModel to capture
    last_hidden_state after each forward pass.

    The captured tensor is stored as model._captured_last_hidden_state.
    """
    model._captured_last_hidden_state = None

    def _capture_hook(module, input, output):
        # output is MoeModelOutputWithPast; output[0] = last_hidden_state
        if hasattr(output, "last_hidden_state"):
            model._captured_last_hidden_state = output.last_hidden_state
        else:
            model._captured_last_hidden_state = output[0]

    model.model.register_forward_hook(_capture_hook)


def patch_model_for_latent_injection(model) -> None:
    """Register a forward pre-hook on the inner LLaDA2MoeModel that
    replaces word embeddings at latent positions with externally provided
    embeddings.

    Before an Obj2 forward pass, set:
        model.model._latent_injection = (positions, embeddings)
    where positions is a list of int and embeddings is (B, len(positions), hidden_size).

    The hook will:
      1. Call module.word_embeddings(input_ids) to get base embeddings
      2. Replace the specified positions with the provided embeddings
      3. Switch the forward to use inputs_embeds instead of input_ids

    FSDP2 safety: This hook fires after FSDP's own pre-hook has unsharded
    the parameters, so word_embeddings is accessible.
    """
    model.model._latent_injection = None

    def _injection_hook(module, args, kwargs):
        injection = module._latent_injection
        if injection is None:
            return args, kwargs

        positions, embeds = injection
        module._latent_injection = None  # reset after use

        # Resolve input_ids from args or kwargs
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and len(args) > 0:
            input_ids = args[0]

        if input_ids is None:
            return args, kwargs

        # Compute base word embeddings (parameters are unsharded at this point)
        inputs_embeds = module.word_embeddings(input_ids)
        inputs_embeds = inputs_embeds.clone()  # avoid in-place modification

        # Replace latent positions with injected embeddings
        inputs_embeds[:, positions, :] = embeds

        # Switch forward to use inputs_embeds
        kwargs["input_ids"] = None
        kwargs["inputs_embeds"] = inputs_embeds

        # Remove input_ids from positional args if present
        if len(args) > 0:
            args = (None,) + args[1:]

        return args, kwargs

    model.model.register_forward_pre_hook(_injection_hook, with_kwargs=True)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


def compute_latent_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute cosine embedding loss between predicted and target latent embeddings.

    Args:
        predicted: (batch, num_latents, latent_dim) from LatentOutputHead.
        target: (batch, num_latents, latent_dim) sentence-transformer embeddings.
        loss_mask: (batch, num_latents) boolean. True = compute loss for this block.
                   If None, all blocks contribute.

    Returns:
        Scalar loss value.
    """
    B, N, D = predicted.shape

    # Flatten to (B*N, D) for CosineEmbeddingLoss
    pred_flat = predicted.reshape(-1, D)
    target_flat = target.reshape(-1, D)

    # Target label = +1 (we want embeddings to be similar)
    y = torch.ones(pred_flat.shape[0], device=predicted.device)

    if loss_mask is not None:
        mask_flat = loss_mask.reshape(-1)  # (B*N,)
        pred_flat = pred_flat[mask_flat]
        target_flat = target_flat[mask_flat]
        y = y[mask_flat]

    if pred_flat.shape[0] == 0:
        return torch.tensor(0.0, device=predicted.device, requires_grad=True)

    loss_fn = nn.CosineEmbeddingLoss()
    return loss_fn(pred_flat, target_flat, y)


# ---------------------------------------------------------------------------
# Inference: Two-phase latent-block generation (aligned with LLaDA 2.1)
# ---------------------------------------------------------------------------


def _sample_tokens(logits: torch.Tensor, temperature: float, top_k: int, top_p: float):
    """Sample tokens from logits with temperature, top-k, and top-p filtering.

    Returns:
        tokens: (batch, seq_len) sampled token IDs
        probs:  (batch, seq_len) probabilities of the sampled tokens
    """
    orig_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    logits = logits.reshape(-1, vocab_size)

    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, vocab_size)
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask_indices = torch.scatter(
            torch.full_like(logits, False, dtype=torch.bool),
            -1,
            sorted_indices,
            sorted_mask,
        )
        logits = logits.masked_fill(mask_indices, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    if temperature == 0:
        tokens = torch.argmax(probs, dim=-1, keepdim=True)
    else:
        tokens = torch.multinomial(probs, num_samples=1)
    token_prob = torch.gather(probs, -1, tokens)
    return tokens.view(*orig_shape), token_prob.view(*orig_shape)


@torch.no_grad()
def latent_block_generate(
    model,
    latent_output_head: nn.Module,
    latent_input_projector: nn.Module,
    input_ids: torch.Tensor,
    block_size: int = 32,
    gen_length: int = 2048,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    threshold: float = 0.95,
    editing_threshold: float = 0.9,
    max_post_steps: int = 16,
    num_to_transfer: int = 1,
    mask_id: int = 156895,
    latent_token_id: int = 156901,
    eos_id: int = 156892,
    eos_early_stop: bool = False,
    latent_block_causal: bool = False,
) -> torch.Tensor:
    """Two-phase latent-block diffusion generation (aligned with LLaDA 2.1).

    Phase 1 -- Latent Planning:
        Given the prompt, run the model once with an Obj1-style mask to predict
        latent embeddings for ALL generation blocks simultaneously.

    Phase 2 -- Block-wise Token Generation (LLaDA 2.1 style):
        Process blocks left-to-right. For each block, inject the predicted latent
        embedding and iteratively denoise tokens. Uses a while-loop with two
        transfer mechanisms:
          - Mask filling: replace <|mask|> tokens with high-confidence samples
          - Editing: replace already-generated tokens if the model produces a
            different token with confidence above editing_threshold
        After all masks in a block are filled, continue refining for up to
        max_post_steps additional iterations.

    Args:
        model: The LLaDA2MoeModelLM model (with hooks already patched via
               patch_model_for_hidden_capture and patch_model_for_latent_injection).
        latent_output_head: Trained LatentOutputHead module.
        latent_input_projector: Trained LatentInputProjector module.
        input_ids: (1, prompt_length) prompt token IDs.
        block_size: Tokens per block (including latent token).
        gen_length: Number of tokens to generate (excluding prompt).
        temperature: Sampling temperature (0 = greedy).
        top_k: Top-k filtering (0 = disabled).
        top_p: Top-p / nucleus filtering (1.0 = disabled).
        threshold: Confidence threshold for filling masked positions.
        editing_threshold: Confidence threshold for editing already-generated
            tokens. A token is edited only when confidence exceeds this AND the
            sampled token differs from the current one.
        max_post_steps: After all masks in a block are filled, continue
            iterating for up to this many additional refinement steps.
        num_to_transfer: Minimum number of mask tokens to fill per step.
        mask_id: <|mask|> token ID.
        latent_token_id: <|block_latent|> token ID.
        eos_id: End-of-sequence token ID.
        eos_early_stop: Stop generation early if EOS is produced.
        latent_block_causal: If True, Phase 1 generates latents autoregressively
            -- each latent_i is predicted conditioned on prompt + previously
            predicted latents 0..i-1 (projected back via LatentInputProjector).

    Returns:
        Generated token IDs (1, output_length), excluding the prompt.
    """
    device = input_ids.device
    prompt_length = input_ids.shape[1]
    orig_block_size = block_size - 1  # content tokens per block

    # Compute layout
    total_content = prompt_length + gen_length
    num_blocks = math.ceil(total_content / orig_block_size)
    total_length = num_blocks * block_size
    prompt_blocks = prompt_length // orig_block_size

    # ======================================================================
    # Build the initial sequence with latent tokens inserted
    # ======================================================================
    padded_content_len = num_blocks * orig_block_size
    content = torch.full((1, padded_content_len), mask_id, dtype=torch.long, device=device)
    content[:, :prompt_length] = input_ids

    content_blocks = content.view(1, num_blocks, orig_block_size)
    latent_col = torch.full((1, num_blocks, 1), latent_token_id, dtype=torch.long, device=device)
    x = torch.cat([content_blocks, latent_col], dim=2).reshape(1, total_length)

    # Track prompt positions in the latent-inserted coordinate system
    prompt_mask_in_new_coords = torch.zeros(total_length, dtype=torch.bool, device=device)
    for i in range(num_blocks):
        block_start_content = i * orig_block_size
        block_end_content = min((i + 1) * orig_block_size, prompt_length)
        if block_start_content >= prompt_length:
            break
        new_block_start = i * block_size
        num_prompt_in_block = block_end_content - block_start_content
        prompt_mask_in_new_coords[new_block_start : new_block_start + num_prompt_in_block] = True

    latent_positions = get_latent_positions(num_blocks, block_size)
    is_latent = torch.zeros(total_length, dtype=torch.bool, device=device)
    for p in latent_positions:
        is_latent[p] = True

    # ======================================================================
    # Phase 1: Latent Planning
    # ======================================================================
    is_prompt = prompt_mask_in_new_coords & ~is_latent
    position_ids = torch.arange(total_length, device=device).unsqueeze(0)

    if not latent_block_causal:
        # --- Standard mode: predict all latents in one forward pass ---
        obj1_input = x.clone()
        obj1_input[~prompt_mask_in_new_coords.unsqueeze(0).expand_as(obj1_input)] = mask_id
        obj1_input[:, latent_positions] = latent_token_id

        obj1_mask = torch.full(
            (1, 1, total_length, total_length),
            float("-inf"), dtype=torch.bfloat16, device=device,
        )
        prompt_idx = is_prompt.nonzero(as_tuple=True)[0]
        latent_idx = is_latent.nonzero(as_tuple=True)[0]

        if prompt_idx.numel() > 0:
            obj1_mask[0, 0, prompt_idx.unsqueeze(1), prompt_idx.unsqueeze(0)] = 0.0
        if latent_idx.numel() > 0 and prompt_idx.numel() > 0:
            obj1_mask[0, 0, latent_idx.unsqueeze(1), prompt_idx.unsqueeze(0)] = 0.0
        if latent_idx.numel() > 0:
            obj1_mask[0, 0, latent_idx.unsqueeze(1), latent_idx.unsqueeze(0)] = 0.0

        model.model._latent_injection = None
        _ = model(
            input_ids=obj1_input, attention_mask=obj1_mask,
            position_ids=position_ids, use_cache=False,
        )
        last_hidden = model._captured_last_hidden_state
        latent_hidden = last_hidden[:, latent_positions, :]
        predicted_latent = latent_output_head(latent_hidden)
        latent_embeds_for_injection = latent_input_projector(predicted_latent)

        del last_hidden, latent_hidden, obj1_input, obj1_mask
    else:
        # --- Latent diffusion mode: iterative denoising of latent tokens ---
        # Start with all latents masked. Each step: predict all, keep highest
        # confidence ones as "unmasked" GT, re-predict the rest conditioned on them.
        obj1_base = x.clone()
        obj1_base[~prompt_mask_in_new_coords.unsqueeze(0).expand_as(obj1_base)] = mask_id
        obj1_base[:, latent_positions] = latent_token_id

        response_mask_1d = ~is_prompt
        response_masks_2d = response_mask_1d.unsqueeze(0)  # (1, total_length)

        # All latents start as masked (to predict)
        latent_is_masked = torch.ones(1, num_blocks, dtype=torch.bool, device=device)
        # Store predicted latent embeddings (latent_dim) for all blocks
        all_predicted = torch.zeros(1, num_blocks, latent_output_head.mlp[-1].out_features, device=device)

        # Iterative denoising: unmask 1 latent per step (like LLaDA 2.1 token gen)
        for step in range(num_blocks):
            num_still_masked = latent_is_masked.sum().item()
            if num_still_masked == 0:
                break

            unmasked_indices = (~latent_is_masked[0]).nonzero(as_tuple=True)[0]
            num_unmasked = unmasked_indices.numel()

            # Build input: [S' | GT_ref (num_unmasked)]
            if num_unmasked > 0:
                gt_placeholder = torch.full(
                    (1, num_unmasked), latent_token_id, dtype=torch.long, device=device
                )
                obj1_input = torch.cat([obj1_base, gt_placeholder], dim=1)
                gt_positions = list(range(total_length, total_length + num_unmasked))
                gt_embeds = latent_input_projector(
                    all_predicted[:, unmasked_indices, :]
                )
                model.model._latent_injection = (gt_positions, gt_embeds)
            else:
                obj1_input = obj1_base.clone()
                model.model._latent_injection = None

            obj1_mask = build_obj1_latent_diffusion_mask(
                total_length, block_size, num_blocks,
                response_masks_2d, latent_is_masked, device, torch.bfloat16,
            )

            obj1_total_len = total_length + num_unmasked
            obj1_pos = torch.cat([
                position_ids[0, :total_length],
                torch.arange(num_unmasked, dtype=torch.long, device=device),
            ]).unsqueeze(0)

            _ = model(
                input_ids=obj1_input, attention_mask=obj1_mask,
                position_ids=obj1_pos, use_cache=False,
            )
            last_hidden = model._captured_last_hidden_state

            # Extract predictions at masked latent positions
            masked_sp_positions = [latent_positions[i] for i in range(num_blocks) if latent_is_masked[0, i]]
            if len(masked_sp_positions) > 0:
                masked_hidden = last_hidden[:, masked_sp_positions, :]
                masked_pred = latent_output_head(masked_hidden)  # (1, num_masked, latent_dim)

                # Compute confidence as cosine similarity with a reference direction
                # (use L2 norm as a proxy for confidence -- higher norm = more confident)
                confidence = masked_pred.norm(dim=-1)  # (1, num_masked)

                # Update all_predicted for masked positions
                masked_block_indices = latent_is_masked[0].nonzero(as_tuple=True)[0]
                all_predicted[:, masked_block_indices, :] = masked_pred

                # Unmask the highest-confidence latent
                best_idx = confidence[0].argmax().item()
                best_block = masked_block_indices[best_idx].item()
                latent_is_masked[0, best_block] = False

            del last_hidden, obj1_input, obj1_mask

        latent_embeds_for_injection = latent_input_projector(all_predicted)
        del obj1_base, all_predicted

    torch.cuda.empty_cache()

    # ======================================================================
    # Phase 2: Block-wise Token Generation (LLaDA 2.1 style)
    # ======================================================================
    block_mask_2d = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    block_attn_mask = (
        block_mask_2d
        .repeat_interleave(block_size, dim=0)
        .repeat_interleave(block_size, dim=1)
        .unsqueeze(0)
        .unsqueeze(0)
    ).to(torch.bfloat16)

    for block_idx in range(prompt_blocks, num_blocks):
        current_window_end = (block_idx + 1) * block_size
        cur_x = x[:, :current_window_end]
        cur_attn_mask = block_attn_mask[:, :, :current_window_end, :current_window_end]
        cur_position_ids = position_ids[:, :current_window_end]

        cur_latent_positions = [p for p in latent_positions if p < current_window_end]
        cur_latent_embeds = latent_embeds_for_injection[:, : len(cur_latent_positions), :]

        block_content_start = block_idx * block_size

        # Per-block prompt mask (for the orig_block_size content positions only)
        prompt_mask_in_block = torch.zeros(orig_block_size, dtype=torch.bool, device=device)
        for j in range(orig_block_size):
            pos = block_content_start + j
            if pos < total_length and prompt_mask_in_new_coords[pos]:
                prompt_mask_in_block[j] = True

        post_steps = 0
        while True:
            old_block_tokens = cur_x[
                :, block_content_start : block_content_start + orig_block_size
            ].clone()
            active_block_mask = old_block_tokens == mask_id

            # Once all masks are filled, start counting post-refinement steps
            if not active_block_mask.any():
                post_steps += 1
            if post_steps > max_post_steps:
                break

            # Inject latent embeddings and forward
            model.model._latent_injection = (cur_latent_positions, cur_latent_embeds)
            logits = model(
                input_ids=cur_x,
                attention_mask=cur_attn_mask,
                position_ids=cur_position_ids,
                use_cache=False,
            ).logits

            block_logits = logits[
                :, block_content_start : block_content_start + orig_block_size, :
            ]
            x0, x0_p = _sample_tokens(
                block_logits, temperature=temperature, top_k=top_k, top_p=top_p,
            )

            # --- Mask filling ---
            mask_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            if active_block_mask.sum() > 0:
                mask_confidence = torch.where(active_block_mask, x0_p, -torch.inf)
                high_conf_mask = (
                    mask_confidence[0] > threshold
                ) & active_block_mask[0]
                num_high = high_conf_mask.sum().item()

                if num_high >= num_to_transfer:
                    mask_transfer_index[0] = high_conf_mask
                else:
                    num_available = active_block_mask.sum().item()
                    if num_available > 0:
                        _, idx = torch.topk(
                            mask_confidence[0],
                            k=min(num_to_transfer, num_available),
                        )
                        mask_transfer_index[0, idx] = True

            # --- Editing: replace non-masked, non-prompt tokens ---
            editing_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            non_mask_positions = ~active_block_mask
            non_prompt_positions = ~prompt_mask_in_block
            editable_positions = non_mask_positions & non_prompt_positions[None, :]

            editing_confidence = torch.where(editable_positions, x0_p, -torch.inf)
            high_conf_editing = (
                editing_confidence[0] > editing_threshold
            ) & editable_positions[0]

            # Only edit if the token actually changed
            token_changed = x0[0] != old_block_tokens[0]
            editing_transfer_index[0] = high_conf_editing & token_changed

            # Combined transfer
            final_transfer_index = mask_transfer_index | editing_transfer_index

            if final_transfer_index.any():
                cur_x[
                    :, block_content_start : block_content_start + orig_block_size
                ][final_transfer_index] = x0[final_transfer_index]

            # Early termination: no masks left and no edits made
            if active_block_mask.sum() == 0 and not editing_transfer_index.any():
                break

        x[:, :current_window_end] = cur_x

        # EOS early stop after completing a block
        if eos_early_stop:
            generated_part = x[0, :current_window_end]
            non_prompt_non_latent = ~prompt_mask_in_new_coords[:current_window_end] & ~is_latent[:current_window_end]
            gen_tokens = generated_part[non_prompt_non_latent]
            if (gen_tokens == mask_id).sum() == 0:
                if (gen_tokens == eos_id).any():
                    break

    return _extract_generated_content(
        x, prompt_length, orig_block_size, block_size, num_blocks, eos_id,
    )


def _extract_generated_content(
    x: torch.Tensor,
    prompt_length: int,
    orig_block_size: int,
    block_size: int,
    num_blocks: int,
    eos_id: int,
) -> torch.Tensor:
    """Remove latent tokens from the generated sequence and return only the
    generated content (after the prompt, up to EOS).

    Args:
        x: (1, total_length) full sequence with latent tokens.
        prompt_length: original prompt length (before latent insertion).
        orig_block_size: content tokens per block (block_size - 1).
        block_size: full block size including latent.
        num_blocks: total number of blocks.
        eos_id: end-of-sequence token ID.

    Returns:
        (1, gen_length) generated token IDs (latent tokens stripped).
    """
    # Strip latent tokens: keep only content positions from each block
    blocks = x.view(1, num_blocks, block_size)
    content_only = blocks[:, :, :orig_block_size]  # drop last position (latent)
    flat = content_only.reshape(1, -1)  # (1, num_blocks * orig_block_size)

    # Extract generated portion (after prompt)
    generated = flat[:, prompt_length:]

    # Truncate at first EOS
    eos_positions = (generated[0] == eos_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        end = eos_positions[0].item() + 1  # include the EOS token
        generated = generated[:, :end]

    return generated
