"""
Latent-Block Diffusion Language Model Training Script

Two-objective training:
  Obj1 (Latent Generation): Predict latent embeddings from prompt via cosine loss.
  Obj2 (Token Generation | Latent): Denoise tokens conditioned on injected latent embeddings.

Based on train_llada2_bd.py with latent token insertion, sentence-transformer
ground truth computation, and dual-forward training loop.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Tuple, Optional

import torch
import torch.distributed as dist
import wandb
from tqdm import trange

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    build_dataloader,
    build_iterative_dataset,
    build_mapping_dataset,
)
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import (
    build_foundation_model,
    build_tokenizer,
    save_model_assets,
    save_model_weights,
)
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    parse_args,
    save_args,
)
from veomni.utils.device import (
    get_device_type,
    get_nccl_backend,
    get_torch_device,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce
from veomni.models.registry import ModelRegistry

ModelRegistry.register_modeling_path("models.llada2_moe")

from dataset.data_transform import process_mdm_sft_latent_example
from dataset import build_local_dataset, build_hf_dataset

# Latent block utilities
from models.llada2_moe.patch_latent_block import (
    LatentOutputHead,
    LatentInputProjector,
    compute_latent_ground_truth,
    insert_latent_tokens,
    remap_response_mask,
    get_latent_positions,
    build_obj1_attention_mask,
    build_noisy_for_obj1,
    build_noisy_for_obj2,
    build_labels_with_latent,
    patch_model_for_hidden_capture,
    patch_model_for_latent_injection,
    compute_latent_loss,
)


def _get_inner_module(module):
    """Get the inner module from a DDP wrapper (or return as-is)."""
    return module.module if hasattr(module, "module") else module


logger = helper.create_logger(__name__)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


@dataclass
class LLaDA2ModelArguments(ModelArguments):
    attn_implementation: Optional[Literal["eager", "sdpa", "flex_attention"]] = field(
        default="sdpa",
        metadata={"help": "Attention implementation to use."},
    )


@dataclass
class LLaDA2DataArguments(DataArguments):
    data_type: Literal["conversation"] = field(
        default="conversation",
        metadata={"help": "Type of the training data."},
    )
    datasets_type: Literal["mapping", "local", "hf", "iterable"] = field(
        default="mapping",
        metadata={"help": "Type of the datasets."},
    )
    train_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The config name for the Huggingface dataset."},
    )
    text_keys: str = field(
        default="messages",
        metadata={"help": "Key to get text from the training data."},
    )
    noise_range_low: float = field(
        default=0.3,
        metadata={"help": "Noise level for random flip input_ids to mask_ids"},
    )
    noise_range_high: float = field(
        default=0.8,
        metadata={"help": "Noise level for random flip input_ids to mask_ids"},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.noise_range_low > self.noise_range_high:
            raise ValueError(
                f"noise_range_low ({self.noise_range_low}) "
                f"cannot be greater than noise_range_high ({self.noise_range_high})."
            )
        if not (0.0 <= self.noise_range_low <= 1.0):
            raise ValueError(f"noise_range_low must be in [0, 1], got {self.noise_range_low}.")
        if not (0.0 <= self.noise_range_high <= 1.0):
            raise ValueError(f"noise_range_high must be in [0, 1], got {self.noise_range_high}.")


@dataclass
class LLaDA2TrainingArguments(TrainingArguments):
    beta1: float = field(default=0.9, metadata={"help": "AdamW optimizer beta1."})
    beta2: float = field(default=0.999, metadata={"help": "AdamW optimizer beta2"})
    block_diffusion_mode: bool = field(
        default=True,
        metadata={"help": "Must be True for latent block diffusion training."},
    )
    block_size: int = field(default=32, metadata={"help": "Block size (including latent token)."})
    same_token_labels: bool = field(
        default=True,
        metadata={"help": "If True, no shift for label alignment (same-position prediction)."},
    )
    # Latent-specific arguments
    latent_loss_weight: float = field(
        default=1.0, metadata={"help": "Weight for latent prediction loss (Obj1)."}
    )
    token_loss_weight: float = field(
        default=1.0, metadata={"help": "Weight for token prediction loss (Obj2)."}
    )
    latent_dim: int = field(
        default=384, metadata={"help": "Dimensionality of sentence-transformer embeddings."}
    )
    sentence_model_name: str = field(
        default="all-MiniLM-L6-v2",
        metadata={"help": "Sentence-transformer model name for latent ground truth."},
    )
    latent_head_lr: float = field(
        default=1.0e-4,
        metadata={"help": "Learning rate for LatentOutputHead and LatentInputProjector (trained from scratch)."},
    )


@dataclass
class Arguments:
    model: "LLaDA2ModelArguments" = field(default_factory=LLaDA2ModelArguments)
    data: "LLaDA2DataArguments" = field(default_factory=LLaDA2DataArguments)
    train: "LLaDA2TrainingArguments" = field(default_factory=LLaDA2TrainingArguments)


# ---------------------------------------------------------------------------
# Block diffusion mask (reused from train_llada2_bd.py for Obj2)
# ---------------------------------------------------------------------------


def block_diffusion_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """Standard block diffusion attention mask for Objective 2."""
    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n

    block_q = torch.where(x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size)
    block_kv = torch.where(x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size)

    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
    offset_block_causal = (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    return block_diagonal | offset_block_causal | block_causal


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    dist.init_process_group(backend=get_nccl_backend())
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(
        dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager
    )

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)
    if not tokenizer.chat_template:
        raise ValueError("No chat template found in the tokenizer.")

    transform = partial(
        process_mdm_sft_latent_example,
        tokenizer=tokenizer,
        max_seq_len=args.data.max_seq_len,
        text_keys=args.data.text_keys,
        noise_range=(args.data.noise_range_low, args.data.noise_range_high),
        mask_token_id=156895,
    )

    if args.data.dataloader_type == "native":
        if args.data.datasets_type == "iterable":
            logger.info_rank0("Start building iterative dataset")
            train_dataset = build_iterative_dataset(
                args.data.train_path, transform=transform, seed=args.train.seed
            )
        elif args.data.datasets_type == "mapping":
            logger.info_rank0("Start building mapping dataset")
            train_dataset = build_mapping_dataset(args.data.train_path, transform=transform)
        elif args.data.datasets_type == "local":
            logger.info_rank0("Start building local dataset")
            train_dataset = build_local_dataset(args.data.train_path, transform=transform)
        elif args.data.datasets_type == "hf":
            logger.info_rank0("Start building Huggingface dataset")
            hf_config_name = args.data.train_config_name
            if hf_config_name == "null":
                hf_config_name = None
            train_dataset = build_hf_dataset(
                args.data.train_path, hf_config_name, transform=transform
            )

        dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
        if args.data.datasets_type in ("mapping", "local", "hf"):
            dataset_length = dataset_length / args.train.data_parallel_size
        args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)

        train_dataloader = build_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train.train_steps,
            rmpad=args.train.rmpad,
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
            dyn_bsz_margin=args.train.dyn_bsz_margin,
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor,
        )
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logger.info_rank0("Prepare model")
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
        force_use_huggingface=args.model.force_use_huggingface,
    )
    model_config = model.config
    hidden_size = model_config.hidden_size
    helper.print_device_mem_info("VRAM usage after building model")

    get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        broadcast_model_weights_from_rank0=args.train.broadcast_model_weights_from_rank0,
    )

    # ------------------------------------------------------------------
    # Latent projection modules (separate from main FSDP model)
    # ------------------------------------------------------------------
    latent_dim = args.train.latent_dim
    device = get_device_type()

    latent_dtype = torch.bfloat16 if args.train.enable_mixed_precision else torch.float32
    latent_output_head = LatentOutputHead(hidden_size, latent_dim).to(
        device=device, dtype=latent_dtype
    )
    latent_input_projector = LatentInputProjector(latent_dim, hidden_size).to(
        device=device, dtype=latent_dtype
    )

    if args.train.world_size > 1:
        latent_output_head = torch.nn.parallel.DistributedDataParallel(
            latent_output_head, device_ids=[args.train.local_rank]
        )
        latent_input_projector = torch.nn.parallel.DistributedDataParallel(
            latent_input_projector, device_ids=[args.train.local_rank]
        )

    # Patch model for hidden state capture and latent embedding injection
    patch_model_for_hidden_capture(model)
    patch_model_for_latent_injection(model)

    # ------------------------------------------------------------------
    # Sentence-transformer (frozen)
    # ------------------------------------------------------------------
    logger.info_rank0(f"Loading sentence-transformer: {args.train.sentence_model_name}")
    from sentence_transformers import SentenceTransformer

    sentence_model = SentenceTransformer(args.train.sentence_model_name, device=device)
    sentence_model.eval()
    for p in sentence_model.parameters():
        p.requires_grad = False

    # ------------------------------------------------------------------
    # Optimizer (includes latent head params)
    # ------------------------------------------------------------------
    latent_params = list(latent_output_head.parameters()) + list(
        latent_input_projector.parameters()
    )
    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        betas=(args.train.beta1, args.train.beta2),
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
    )
    # Add latent module params to the optimizer
    optimizer.add_param_group({"params": latent_params, "lr": args.train.latent_head_lr, "weight_decay": args.train.weight_decay})

    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(
            model, model_config, args.train.data_parallel_mode
        )
        optimizer.register_step_pre_hook(optimizer_pre_hook)

    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train.train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},
            )
        model_assets = [model_config, tokenizer]
        save_model_assets(args.train.model_assets_dir, model_assets)

    if args.train.profile_this_rank:
        profiler = helper.create_profiler(
            start_step=args.train.profile_start_step,
            end_step=args.train.profile_end_step,
            trace_dir=args.train.profile_trace_dir,
            record_shapes=args.train.profile_record_shapes,
            profile_memory=args.train.profile_profile_memory,
            with_stack=args.train.profile_with_stack,
            global_rank=args.train.global_rank,
        )
        profiler.start()

    # ------------------------------------------------------------------
    # Pre-compute block diffusion mask for Obj2
    # ------------------------------------------------------------------
    block_size = args.train.block_size
    max_seq_len = args.data.max_seq_len
    num_blocks = max_seq_len // block_size
    mask_token_id = 156895
    latent_token_id = 156901
    pad_token_id = model_config.pad_token_id

    bd_attn_full_len = max_seq_len * 2
    bd_mask_flag = (
        block_diffusion_mask(
            b=None,
            h=None,
            q_idx=torch.arange(bd_attn_full_len)[:, None],
            kv_idx=torch.arange(bd_attn_full_len)[None, :],
            block_size=block_size,
            n=max_seq_len,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )

    bd_mask_dtype = torch.float32 if args.train.enable_mixed_precision else torch.bfloat16
    bd_mask_prototype = torch.zeros_like(bd_mask_flag, dtype=bd_mask_dtype)
    bd_mask_prototype.masked_fill_(bd_mask_flag.logical_not(), float("-inf"))

    latent_positions = get_latent_positions(num_blocks, block_size)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
    )

    if args.train.load_checkpoint_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}
        Checkpointer.load(args.train.load_checkpoint_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step = global_step % args.train.train_steps
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        # Restore latent projection modules (saved without DDP 'module.' prefix)
        if "latent_output_head" in state["extra_state"]:
            _get_inner_module(latent_output_head).load_state_dict(
                state["extra_state"]["latent_output_head"]
            )
        if "latent_input_projector" in state["extra_state"]:
            _get_inner_module(latent_input_projector).load_state_dict(
                state["extra_state"]["latent_input_projector"]
            )
        if start_step == 0:
            iter(train_dataloader)
        dist.barrier()
        logger.info_rank0(
            f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!"
        )

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload,
        args.train.enable_gradient_checkpointing,
        args.train.activation_gpu_limit,
    )

    # Pre-build position_ids template
    position_ids_template = torch.cat(
        [
            torch.arange(max_seq_len, dtype=torch.long),
            torch.arange(max_seq_len, dtype=torch.long),
        ],
        dim=0,
    )  # (2 * max_seq_len,)

    model.train()
    latent_output_head.train()
    latent_input_projector.train()

    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, "
        f"epochs: {args.train.num_train_epochs}, block_size: {block_size}, "
        f"latent_dim: {latent_dim}, num_blocks: {num_blocks}"
    )

    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)

        for _ in range(start_step, args.train.train_steps):
            global_step += 1

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(
                    f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}"
                )
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0.0
            total_latent_loss = 0.0
            total_token_loss = 0.0
            synchronize()
            start_time = time.time()

            for micro_batch in micro_batches:
                environ_meter.add(micro_batch)
                if args.data.enable_multisource:
                    micro_batch.pop("ds_idx", None)
                    micro_batch.pop("source_name", None)

                # Move tensors to device
                micro_batch = {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }

                input_ids = micro_batch["input_ids"]  # (B, max_seq_len)
                noisy_input_ids = micro_batch["noisy_input_ids"]  # (B, max_seq_len)
                labels = micro_batch.pop("labels")  # (B, max_seq_len)
                response_mask = micro_batch.pop("response_mask")  # (B, max_seq_len)
                batch_size = input_ids.shape[0]

                # ---- Shared prep ----
                clean_with_latent, clean_blocks = insert_latent_tokens(
                    input_ids, block_size, latent_token_id
                )
                response_mask_new = remap_response_mask(response_mask, block_size)
                labels_with_latent = build_labels_with_latent(labels, block_size)

                # Sentence-transformer ground truth (cast to match latent module dtype)
                st_embeddings = compute_latent_ground_truth(
                    clean_blocks, tokenizer, sentence_model, clean_with_latent.device
                ).to(
                    latent_dtype
                )  # (B, num_blocks, latent_dim)

                # Latent loss mask: exclude padding-only blocks
                pad_mask = (clean_blocks == pad_token_id).all(dim=-1)
                latent_loss_mask = ~pad_mask  # (B, num_blocks)

                # Position IDs for [S', S] concatenation
                position_ids = position_ids_template.unsqueeze(0).expand(batch_size, -1).clone()
                position_ids = position_ids.to(device, non_blocking=True)

                # ==============================================================
                # OBJECTIVE 1: Latent Generation
                # Memory optimization: S is fully blocked in Obj1 mask, so we
                # only forward S' (max_seq_len) instead of [S', S] (2*max_seq_len).
                # ==============================================================
                obj1_noisy = build_noisy_for_obj1(
                    clean_with_latent, response_mask_new, mask_token_id, latent_positions,
                    latent_token_id=latent_token_id,
                )
                # Obj1 mask: only need the S' quadrant (max_seq_len x max_seq_len)
                obj1_mask = build_obj1_attention_mask(
                    max_seq_len, block_size, response_mask_new, device, bd_mask_dtype
                )[:, :, :max_seq_len, :max_seq_len]

                obj1_position_ids = (
                    torch.arange(max_seq_len, dtype=torch.long, device=device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )

                # Ensure no injection for Obj1
                model.model._latent_injection = None

                with model_fwd_context:
                    _ = model(
                        input_ids=obj1_noisy,
                        attention_mask=obj1_mask,
                        position_ids=obj1_position_ids,
                        use_cache=False,
                        output_router_logits=False,
                    )
                    last_hidden = model._captured_last_hidden_state  # (B, max_seq_len, hidden_size)
                    latent_hidden = last_hidden[
                        :, latent_positions, :
                    ]  # (B, num_blocks, hidden_size)
                    predicted_latent = latent_output_head(
                        latent_hidden
                    )  # (B, num_blocks, latent_dim)
                    latent_loss = compute_latent_loss(
                        predicted_latent, st_embeddings, latent_loss_mask
                    )

                scaled_latent_loss = (
                    latent_loss * args.train.latent_loss_weight / len(micro_batches)
                )
                with model_bwd_context:
                    scaled_latent_loss.backward()

                total_latent_loss += scaled_latent_loss.item()

                # Free Obj1 intermediates
                del obj1_noisy, obj1_mask, last_hidden, latent_hidden, predicted_latent
                torch.cuda.empty_cache()

                # ==============================================================
                # OBJECTIVE 2: Token Generation | Latent
                # ==============================================================
                noisy_with_latent = build_noisy_for_obj2(
                    noisy_input_ids, block_size, latent_token_id
                )
                full_input_obj2 = torch.cat([noisy_with_latent, clean_with_latent], dim=1)
                obj2_mask = bd_mask_prototype.expand(batch_size, -1, -1, -1).to(
                    device, non_blocking=True
                )

                # Inject latent embeddings at latent positions in both S' and S halves
                latent_input_embeds = latent_input_projector(
                    st_embeddings.detach()
                )  # (B, num_blocks, hidden_size)
                all_inject_positions = latent_positions + [
                    p + max_seq_len for p in latent_positions
                ]
                all_inject_embeds = torch.cat(
                    [latent_input_embeds, latent_input_embeds], dim=1
                )  # (B, 2*num_blocks, hidden_size)
                model.model._latent_injection = (all_inject_positions, all_inject_embeds)

                with model_fwd_context:
                    logits = model(
                        input_ids=full_input_obj2,
                        attention_mask=obj2_mask,
                        position_ids=position_ids,
                        use_cache=False,
                        output_router_logits=False,
                    ).logits
                    noisy_logits = logits[:, :max_seq_len].contiguous()

                    if args.train.same_token_labels:
                        unscaled_loss = torch.nn.functional.cross_entropy(
                            noisy_logits.view(-1, noisy_logits.shape[-1]),
                            labels_with_latent.view(-1),
                            reduction="none",
                        )
                        valid_count = (labels_with_latent != -100).sum()
                        token_loss = unscaled_loss.sum() / valid_count.clamp(min=1)
                    else:
                        shifted_logits = noisy_logits[:, :-1, :].contiguous()
                        shifted_labels = labels_with_latent[:, 1:].contiguous()
                        unscaled_loss = torch.nn.functional.cross_entropy(
                            shifted_logits.view(-1, shifted_logits.shape[-1]),
                            shifted_labels.view(-1),
                            reduction="none",
                        ).view(shifted_logits.shape[0], -1)
                        valid_count = (shifted_labels != -100).sum()
                        token_loss = unscaled_loss.sum() / valid_count.clamp(min=1)

                scaled_token_loss = token_loss * args.train.token_loss_weight / len(micro_batches)
                with model_bwd_context:
                    scaled_token_loss.backward()

                total_token_loss += scaled_token_loss.item()
                total_loss += scaled_latent_loss.item() + scaled_token_loss.item()

                # Free Obj2 intermediates
                del full_input_obj2, obj2_mask, logits, noisy_logits
                del micro_batch

            # ----------------------------------------------------------
            # Optimizer step
            # ----------------------------------------------------------
            if hasattr(model, "clip_grad_norm_"):
                _gn = model.clip_grad_norm_(args.train.max_grad_norm)
                grad_norm = _gn.item() if hasattr(_gn, "item") else float(_gn)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + latent_params, args.train.max_grad_norm
                )

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            total_loss, total_latent_loss, total_token_loss, grad_norm = all_reduce(
                (total_loss, total_latent_loss, total_token_loss, grad_norm),
                group=get_parallel_state().fsdp_group,
            )
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            data_loader_tqdm.update()
            data_loader_tqdm.set_postfix_str(
                f"loss: {total_loss:.2f}, lat: {total_latent_loss:.3f}, "
                f"tok: {total_token_loss:.3f}, gn: {grad_norm:.2f}, lr: {lr:.2e}",
                refresh=True,
            )

            if args.train.global_rank == 0:
                if args.train.use_wandb:
                    train_metrics.update(
                        {
                            "training/loss": total_loss,
                            "training/latent_loss": total_latent_loss,
                            "training/token_loss": total_token_loss,
                            "training/grad_norm": grad_norm,
                            "training/lr": lr,
                        }
                    )
                    wandb.log(train_metrics, step=global_step)

            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
                profiler.step()
                if global_step == args.train.profile_end_step:
                    profiler.stop()

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(
                    args.train.save_checkpoint_path, f"global_step_{global_step}"
                )
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                        "latent_output_head": _get_inner_module(latent_output_head).state_dict(),
                        "latent_input_projector": _get_inner_module(
                            latent_input_projector
                        ).state_dict(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                dist.barrier()
                logger.info_rank0(
                    f"Distributed checkpoint saved at {save_checkpoint_path} successfully!"
                )

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(
                args.train.save_checkpoint_path, f"global_step_{global_step}"
            )
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "latent_output_head": latent_output_head.state_dict(),
                    "latent_input_projector": latent_input_projector.state_dict(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            dist.barrier()
            logger.info_rank0(
                f"Distributed checkpoint saved at {save_checkpoint_path} successfully!"
            )

    synchronize()
    del optimizer, lr_scheduler
    helper.empty_cache()

    if (
        args.train.global_rank == 0
        and args.train.save_hf_weights
        and save_checkpoint_path is not None
    ):
        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=save_checkpoint_path,
            output_dir=args.train.output_dir,
            ckpt_manager=args.train.ckpt_manager,
        )
        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
