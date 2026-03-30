# Latent-Block Diffusion Language Model: Implementation Details

This document describes the full training pipeline for the Latent-Block dLLM,
covering data preparation, latent token insertion, the two training objectives,
attention mask construction, embedding injection, and loss computation.

---

## Table of Contents

1. [Overview](#overview)
2. [Stage 0: Data Preparation (DataLoader)](#stage-0-data-preparation-dataloader)
3. [Stage 1: Shared Preprocessing (Training Loop)](#stage-1-shared-preprocessing-training-loop)
4. [Stage 2: Objective 1 -- Latent Generation](#stage-2-objective-1----latent-generation)
5. [Stage 3: Objective 2 -- Token Generation conditioned on Latent](#stage-3-objective-2----token-generation-conditioned-on-latent)
6. [Stage 4: Optimizer Step](#stage-4-optimizer-step)
7. [Summary: Obj1 vs Obj2 Comparison](#summary-obj1-vs-obj2-comparison)
8. [Key Files](#key-files)

---

## Overview

Each micro-batch goes through **two complete forward+backward passes**, one per objective:

```
Data → Shared Prep → Obj1 forward → Obj1 backward → empty_cache → Obj2 forward → Obj2 backward → optimizer.step()
```

The model is a standard LLaDA2 MoE transformer. Latent block logic is implemented
entirely outside the model via:
- Utility functions in `models/llada2_moe/patch_latent_block.py`
- Forward hooks for hidden state capture and embedding injection
- The training script `tasks/train_llada2_bd_latent.py`

Two small trainable modules are added alongside the main model:
- **LatentOutputHead**: Bottleneck MLP (hidden_size → hidden_size//2 → latent_dim) for Obj1 loss
- **LatentInputProjector**: Linear (latent_dim → hidden_size) for Obj2 conditioning

---

## Stage 0: Data Preparation (DataLoader)

**File**: `tasks/dataset/data_transform.py` -- `process_mdm_sft_latent_example()`

### Input
Multi-turn conversation: `[{role: user, content: ...}, {role: assistant, content: ...}, ...]`

### Processing

1. **Tokenize** via `apply_chat_template_mdm()` → `input_ids` (length `max_seq_len`)
2. **Compute `response_mask`** via `compute_response_mask()`:
   - For each assistant message in the conversation, determine its token span
   - Mark those positions as `True` (response), everything else as `False` (prompt)
   - Supports arbitrary multi-turn: every assistant turn is independently marked

   ```
   response_mask: [F, F, ..., T, T, ..., F, F, ..., T, T, ..., F, F]
                   ╰─ system + user ─╯  ╰─ asst1 ─╯  ╰─ user2 ─╯  ╰─ asst2 ─╯  ╰─ pad ─╯
   ```

3. **Apply noise** (`sft_noise_transition`): randomly mask ~55% of response positions with `<|mask|>` (id=156895)
4. **Compute labels**: `-100` everywhere except at positions that were actually masked

### Output
```python
{
    "input_ids":       (max_seq_len,),  # clean tokens
    "noisy_input_ids": (max_seq_len,),  # response positions ~55% masked
    "labels":          (max_seq_len,),  # -100 at prompt/unmasked, real token at masked positions
    "attention_mask":  (max_seq_len,),  # all ones
    "response_mask":   (max_seq_len,),  # True at all assistant response positions
}
```

**Note**: No latent tokens are inserted at this stage. This is the original coordinate system.

---

## Stage 1: Shared Preprocessing (Training Loop)

**File**: `tasks/train_llada2_bd_latent.py` lines 556-590

### 1a. Insert Latent Tokens

`insert_latent_tokens(input_ids, block_size=32, latent_token_id=156901)`

Takes the raw 2048-token sequence and restructures it into blocks with latent tokens:

```
Original (2048 tokens):
  [t0, t1, ..., t30 | t31, ..., t61 | ... | t1953, ..., t1983]
  (conceptually 64 groups of 31+padding, but actually truncated to 64*31=1984 tokens)

After insertion (2048 tokens = 64 blocks x 32):
  Block 0:  [t0,    t1,    ..., t30,   <BL>]    positions 0-31
  Block 1:  [t31,   t32,   ..., t61,   <BL>]    positions 32-63
  ...
  Block 63: [t1953, t1954, ..., t1983, <BL>]    positions 2016-2047
```

Where `<BL>` = `<|block_latent|>` (id=156901).

**Latent positions**: `[31, 63, 95, ..., 2047]` (last position of each block)

Also returns `clean_blocks: (B, 64, 31)` -- the 31 original tokens per block, used for
computing sentence-transformer embeddings.

### 1b. Remap response_mask

`remap_response_mask(response_mask, block_size=32)`

Maps the per-position response_mask from original coordinates to the new coordinate
system after latent insertion:

- Original position `orig` → new position `(orig // 31) * 32 + (orig % 31)`
- Latent positions (pos 31 of each block) inherit `True` if **any** token in that block is a response token

### 1c. Compute Sentence-Transformer Ground Truth

`compute_latent_ground_truth(clean_blocks, tokenizer, sentence_model, device)`

For each block's 31 tokens:
1. Decode to text via `tokenizer.decode()`
2. Encode with frozen `all-MiniLM-L6-v2` sentence-transformer

```python
st_embeddings: (B, 64, 384)  # one 384-dim semantic vector per block
```

This runs inside `torch.no_grad()` -- the sentence-transformer is completely frozen.

### 1d. Build Labels with Latent

`build_labels_with_latent(labels, block_size=32)`

Inserts `-100` at latent positions so they never contribute to cross-entropy loss:

```
Original labels: [-100, ..., t_masked, -100, ...]     (original coordinates)
After insertion: [-100, ..., t_masked, -100, -100]     (latent pos = -100)
                                              ↑ latent position
```

### 1e. Latent Loss Mask

Blocks that are entirely padding tokens should not contribute to the latent loss:

```python
pad_mask = (clean_blocks == pad_token_id).all(dim=-1)  # (B, 64)
latent_loss_mask = ~pad_mask                            # True = compute loss for this block
```

### 1f. Position IDs

Same as standard block diffusion: `[0, 1, ..., 2047, 0, 1, ..., 2047]`

Each half (S' and S) gets independent position IDs so RoPE is computed separately.

---

## Stage 2: Objective 1 -- Latent Generation

**Goal**: Given only the prompt, predict the semantic embedding for each block.

### 2a. Build Obj1 Input Sequence

`build_noisy_for_obj1(clean_with_latent, response_mask_new, mask_token_id, latent_positions)`

Constructs the S' (noisy) half for Objective 1:

```
S' (obj1_noisy):
  prompt positions:   keep original tokens (clean)
  response positions: ALL replaced with <|mask|>
  latent positions:   also <|mask|>  ← latent is the prediction target

Full input: [S', S] = [obj1_noisy (2048) | clean_with_latent (2048)]
Total length: 4096
```

Concrete example (block_size=4, 4 blocks):

```
S' = [user, tok, clean, <M> | <M>, <M>, <M>, <M> | <M>, <M>, <M>, <M> | <M>, <M>, <M>, <M>]
      ╰── prompt block ───╯   ╰── response (all masked) ──╯  ...     latent pos ↑ also <M>

S  = [user, tok, clean, <BL> | resp1, resp2, resp3, <BL> | resp4, resp5, resp6, <BL> | ...]
      ╰── full clean sequence with latent tokens ──────────────────────────────────────────╯
```

### 2b. Obj1 Attention Mask

`build_obj1_attention_mask(max_seq_len, block_size, response_masks, device, dtype)`

Returns `(B, 1, 4096, 4096)` attention mask. The per-sample construction is necessary
because prompt boundaries vary across samples in a multi-turn setting.

**Quadrant structure**:

```
                 S' (0..2047)              S (2048..4095)
           ┌──────────────────────────┬──────────────────────┐
S'         │     OBJ1_INNER           │    ALL -inf          │
(0..2047)  │                          │    (S invisible)     │
           ├──────────────────────────┼──────────────────────┤
S          │     ALL -inf             │    ALL -inf          │
(2048..    │     (S also invisible)   │                      │
 4095)     └──────────────────────────┴──────────────────────┘
```

**OBJ1_INNER (S' quadrant, 2048x2048)** attention rules:

```
query \ key      prompt pos    latent pos    other response pos
─────────────────────────────────────────────────────────────────
prompt pos        ✅ allow      ❌ block      ❌ block
latent pos        ✅ allow      ✅ allow      ❌ block
other response    ❌ block      ❌ block      ❌ block
```

- **prompt ↔ prompt**: Allowed. Prompt tokens can attend to each other.
- **latent → prompt**: Allowed. Latent needs to read prompt information to predict block semantics.
- **latent ↔ latent**: Allowed (bidirectional). All 64 latent tokens can see each other, enabling coordinated planning across blocks.
- **prompt → latent**: Blocked. Latent tokens are being predicted; prompt should not see them.
- **Other response positions**: Fully invisible (neither as query nor as key). They are masked out entirely in S'.

**Intuition**: The model can only see the prompt, and must "plan" at each latent position what
that block should say. The latent tokens can coordinate with each other across all blocks.

### 2c. Obj1 Forward Pass and Loss

```python
# Forward through the full model (with FSDP)
model(input_ids=[S', S], attention_mask=obj1_mask, position_ids=[0..2047, 0..2047])

# Capture hidden states via registered forward hook
last_hidden = model._captured_last_hidden_state   # (B, 4096, hidden_size)

# Extract latent positions from the S' half only
latent_hidden = last_hidden[:, [31, 63, 95, ..., 2047], :]  # (B, 64, hidden_size)

# Project through bottleneck MLP
predicted = latent_output_head(latent_hidden)  # (B, 64, 384)
# Architecture: hidden_size → Linear → LayerNorm → GELU → Linear → latent_dim

# Cosine embedding loss against sentence-transformer ground truth
latent_loss = CosineEmbeddingLoss(predicted, st_embeddings, loss_mask=latent_loss_mask)
```

Then `backward()` propagates gradients through `latent_output_head` and the main model.

---

## Stage 3: Objective 2 -- Token Generation conditioned on Latent

**Goal**: Given all latent embeddings (as conditioning), denoise and generate response tokens.

### 3a. Build Obj2 Input Sequence

`build_noisy_for_obj2(noisy_input_ids, block_size, latent_token_id)`

Constructs the S' half for Objective 2:

```
S' (noisy_with_latent):
  prompt positions:   keep original tokens (clean)
  response positions: keep noisy state (~55% masked with <|mask|>)
  latent positions:   token ID is <|block_latent|> (placeholder)
                      BUT actual embedding is REPLACED via forward hook

S = clean_with_latent (same as Obj1)

Full input: [S', S] (4096)
```

### 3b. Embedding Injection Mechanism

**File**: `models/llada2_moe/patch_latent_block.py` -- `patch_model_for_latent_injection()`

A `register_forward_pre_hook(with_kwargs=True)` is registered on the inner `LLaDA2MoeModel`.
This hook fires **after** FSDP2 has unsharded the parameters, so `word_embeddings` is accessible.

The hook flow:

```
Normal forward:
  input_ids → word_embeddings(input_ids) → transformer layers

With injection:
  input_ids → word_embeddings(input_ids) → REPLACE latent positions → transformer layers
                                              ↓
                              latent_input_projector(st_embeddings.detach())
                              Linear(384 → hidden_size)
```

The injection replaces latent positions in **both** the S' and S halves:

```python
all_inject_positions = latent_positions + [p + max_seq_len for p in latent_positions]
# = [31, 63, ..., 2047, 2079, 2111, ..., 4095]
```

This means at latent positions, the model does not see the learned `<|block_latent|>` token
embedding, but instead receives the actual semantic information from the sentence-transformer,
projected to the model's hidden dimension.

**FSDP2 hook ordering** (why injection is safe):

```
model.__call__()
  → FSDP outer pre-hook: unshard lm_head
  → model.forward()
    → self.model.__call__()
      → FSDP inner pre-hook: unshard word_embeddings, norm, etc.
      → OUR pre-hook: word_embeddings available ✓ → inject latent embeddings
      → model.model.forward() proceeds with modified inputs_embeds
```

### 3c. Obj2 Attention Mask -- Standard Block Diffusion Mask

This mask is **identical** to the one used in the original `train_llada2_bd.py`. It is
pre-computed once before the training loop as a prototype tensor `(1, 1, 4096, 4096)` and
expanded per batch.

The mask is composed of three logical components:

**1. Block Diagonal (M_BD)**: Tokens within the same block and same section can attend to each other.
```
S' block_i ↔ S' block_i    (within-block self-attention in noisy section)
S  block_i ↔ S  block_i    (within-block self-attention in clean section)
```

**2. Offset Block Causal (M_OBC)**: Noisy blocks can see earlier clean blocks (cross-section).
```
S' block_i → S block_0..i-1    (noisy block sees previous clean blocks for context)
```

**3. Block Causal (M_BC)**: Clean blocks can see earlier clean blocks (within clean section).
```
S  block_i → S block_0..i      (standard causal within clean reference)
```

**Visualization** (4 blocks example):

```
S' queries seeing these keys:

           S'_b0  S'_b1  S'_b2  S'_b3 │ S_b0   S_b1   S_b2   S_b3
S'_b0    [  ██                         │  ██                        ]
S'_b1    [         ██                  │  ██     ██                 ]
S'_b2    [                ██           │  ██     ██     ██          ]
S'_b3    [                       ██    │  ██     ██     ██     ██   ]
          ─────────────────────────────┼────────────────────────────
S_b0     [                             │  ██                        ]
S_b1     [                             │  ██     ██                 ]
S_b2     [                             │  ██     ██     ██          ]
S_b3     [                             │  ██     ██     ██     ██   ]

██ = attention allowed (mask = 0.0)
blank = attention blocked (mask = -inf)
```

**Latent token's role in this mask**: The latent token sits at the last position of each block
(e.g., position 31 in block 0). Since it is within the block diagonal, it naturally:
- Can see and be seen by all other tokens in its block (including the noisy response tokens)
- Via M_OBC, S' block_i's latent can see S blocks 0..i-1 (previous clean blocks including their latents)

This means each block's tokens can condition on:
1. The injected latent embedding at the block's last position (intra-block, via M_BD)
2. Previous blocks' clean tokens and latent embeddings (inter-block, via M_OBC)

### 3d. Obj2 Forward Pass and Loss

```python
logits = model(
    input_ids=[S', S],
    attention_mask=bd_mask,
    position_ids=[0..2047, 0..2047],
).logits

# Extract logits for S' half only
noisy_logits = logits[:, :2048]  # (B, 2048, vocab_size)

# same_token_labels mode (no shift):
token_loss = CrossEntropy(noisy_logits, labels_with_latent)
# labels_with_latent has -100 at:
#   - all prompt positions
#   - all latent positions
#   - response positions that were NOT masked by noise
# Only masked response positions contribute to the loss.
```

Then `backward()` propagates gradients through `latent_input_projector` and the main model.

---

## Stage 4: Optimizer Step

After both Obj1 and Obj2 backward passes have accumulated gradients:

```python
# Gradient clipping (model-specific for FSDP2)
grad_norm = model.clip_grad_norm_(max_grad_norm)

# Single optimizer step updates ALL parameters:
#   - Main LLaDA2 MoE model (FSDP2-sharded)
#   - LatentOutputHead (DDP-wrapped, separate param group)
#   - LatentInputProjector (DDP-wrapped, separate param group)
optimizer.step()
lr_scheduler.step()
optimizer.zero_grad()
```

The combined loss for logging:
```
total_loss = latent_weight * latent_loss + token_weight * token_loss
```

Wandb logs (if enabled) track `latent_loss`, `token_loss`, and combined `loss` separately.

---

## Summary: Obj1 vs Obj2 Comparison

| Aspect | Obj1 (Latent Generation) | Obj2 (Token Gen \| Latent) |
|--------|--------------------------|----------------------------|
| **Goal** | Predict block-level semantic plan from prompt | Denoise tokens given latent conditioning |
| **S' latent positions** | `<\|mask\|>` (prediction target) | Injected ST embedding (given condition) |
| **S' response positions** | ALL `<\|mask\|>` (invisible) | ~55% masked (partially visible) |
| **S section** | Fully blocked (-inf) | Visible via offset block causal |
| **Mask structure** | Custom per-sample (prompt+latent only) | Standard block diffusion (pre-computed) |
| **Loss function** | `CosineEmbeddingLoss` (384-dim) | `CrossEntropyLoss` (vocab logits) |
| **Trainable modules** | model + LatentOutputHead | model + LatentInputProjector |
| **Hidden state extraction** | Via forward hook on inner model | Not needed (uses logits) |
| **Embedding injection** | Not used | Via forward pre-hook (replaces word embeddings) |

---

## Key Files

| File | Purpose |
|------|---------|
| `models/llada2_moe/patch_latent_block.py` | All latent block utilities: modules, mask builders, hooks, loss |
| `tasks/dataset/data_transform.py` | Data preprocessing with multi-turn `response_mask` |
| `tasks/train_llada2_bd_latent.py` | Training script with dual-objective loop |
| `configs/sft/llada2_mini_bd_latent_sft.yaml` | Training configuration |
| `tokenizers/latent/` | Tokenizer with `<\|block_latent\|>` (id=156901) |
| `models/llada2_moe/modeling_llada2_moe.py` | Base model (NOT modified) |

---

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | 32 | Tokens per block including latent (31 content + 1 latent) |
| `max_seq_len` | 2048 | Sequence length after latent insertion (64 blocks x 32) |
| `latent_dim` | 384 | Sentence-transformer embedding dimension |
| `latent_loss_weight` | 1.0 | Weight for Obj1 (latent prediction) loss |
| `token_loss_weight` | 1.0 | Weight for Obj2 (token generation) loss |
| `sentence_model_name` | `all-MiniLM-L6-v2` | Frozen sentence-transformer for GT |
| `noise_range_low/high` | 0.3 / 0.8 | Masking probability range for response tokens |
| `mask_token_id` | 156895 | `<\|mask\|>` token ID |
| `latent_token_id` | 156901 | `<\|block_latent\|>` token ID |
| `same_token_labels` | True | No shift for label alignment (same-position prediction) |
