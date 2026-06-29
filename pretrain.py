from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2_pytorch import AdamAtan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str
    loss: LossConfig


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_path: str

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Init the reasoning core from a pretrained checkpoint (puzzle_emb is re-learned).
    load_checkpoint: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []

    # Preemption resilience: save a full-state checkpoint (model + optimizers +
    # step) every N optimizer steps so a preempted/requeued job resumes from the
    # latest one instead of restarting. 0 disables (only the per-eval save).
    checkpoint_steps: int = 0


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,

        dataset_path=config.data_path,

        rank=rank,
        num_replicas=world_size,
        
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,

        num_workers=1,
        prefetch_factor=8,

        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore

        batch_size=config.global_batch_size // world_size,

        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model, dynamic=False)  # type: ignore

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    optimizers = [
        CastedSparseEmbeddingSignSGD_Distributed(
            model.model.puzzle_emb.buffers(),  # type: ignore
            
            lr=config.puzzle_emb_lr,  # set initial lr from config; scheduler will still update it
            weight_decay=config.puzzle_emb_weight_decay,

            world_size=world_size
        ),
        AdamAtan2(
            model.parameters(),

            lr=config.lr,  # set initial lr from config; scheduler will still update it
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
    ]
    optimizer_lrs = [
        config.puzzle_emb_lr,
        config.lr
    ]

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, world_size=world_size)

    # Path A (transductive adaptation): initialise the reasoning CORE from a
    # pretrained checkpoint but KEEP the freshly-initialised puzzle_emb table, so
    # we re-learn per-puzzle embeddings for THIS dataset's identifier indexing on
    # the frozen core. Train with lr=0 (core) and puzzle_emb_lr>0 (embeddings).
    if config.load_checkpoint is not None:
        sd = torch.load(config.load_checkpoint, map_location="cuda", weights_only=False)
        model_keys = list(model.state_dict().keys())
        m_has = any(k.startswith("_orig_mod.") for k in model_keys)
        c_has = any(k.startswith("_orig_mod.") for k in sd)
        if m_has and not c_has:
            sd = {f"_orig_mod.{k}": v for k, v in sd.items()}
        elif c_has and not m_has:
            sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        # Drop the checkpoint's puzzle_emb so our fresh, trainable table is kept.
        sd = {k: v for k, v in sd.items() if not k.endswith("puzzle_emb.weights")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        miss_non_pe = [k for k in missing if "puzzle_emb" not in k]
        print(f"[load_checkpoint] init core from {config.load_checkpoint}: "
              f"loaded={len(sd)} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        print(f"[load_checkpoint] non-puzzle_emb missing (should be empty): {miss_non_pe}", flush=True)
        assert not unexpected, f"unexpected checkpoint keys: {unexpected[:8]}"
        assert not miss_non_pe, f"core weights failed to load: {miss_non_pe[:8]}"

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def _latest_ckpt_path(config: PretrainConfig) -> str:
    return os.path.join(config.checkpoint_path, "latest.pt")


def save_full_state(config: PretrainConfig, train_state: TrainState):
    """Atomically save full training state (step + model + optimizers) for resume.

    Written by rank 0 only. Atomic via tmp-file + os.replace so a preemption
    mid-write can never corrupt latest.pt.
    """
    if config.checkpoint_path is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    payload = {
        "step": train_state.step,
        "model": train_state.model.state_dict(),
        "optimizers": [opt.state_dict() for opt in train_state.optimizers],
    }
    path = _latest_ckpt_path(config)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def try_resume(config: PretrainConfig, train_state: TrainState) -> int:
    """Load latest.pt into train_state if present. Returns the resumed step (0 if none)."""
    if config.checkpoint_path is None:
        return 0
    path = _latest_ckpt_path(config)
    if not os.path.exists(path):
        return 0
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    train_state.model.load_state_dict(ckpt["model"])
    try:
        for opt, sd in zip(train_state.optimizers, ckpt.get("optimizers", [])):
            opt.load_state_dict(sd)
    except Exception as e:
        print(f"[resume] WARNING: could not restore optimizer state ({e}); continuing with fresh optimizer state.")
    train_state.step = int(ckpt["step"])
    return train_state.step


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )


def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
            
    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
            
        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def evaluate(config: PretrainConfig, train_state: TrainState, eval_loader: torch.utils.data.DataLoader, eval_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    with torch.inference_mode():
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        
        all_preds = {}

        metric_keys = []
        metric_values = None
        metric_global_batch_size = [0 for _ in range(len(set_ids))]
        
        carry = None
        for set_name, batch, global_batch_size in eval_loader:
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            step_frames = []

            # Forward
            while True:
                req = list(set(config.eval_save_outputs + ["intermediate_preds_step"]))
                carry, _, metrics, preds, all_finish = train_state.model(carry=carry, batch=batch, return_keys=req)
                
                if "intermediate_preds_step" in preds:
                    step_frames.append(preds["intermediate_preds_step"].cpu())


                if all_finish:
                    break

            if step_frames:
                inter = torch.stack(step_frames, dim=0)  # [steps, batch, 81]
                all_preds.setdefault("intermediate_preds", [])
                all_preds["intermediate_preds"].append(inter)

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:

                        all_preds.setdefault(k, [])
                        all_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory
                        
            del carry, preds, batch, all_finish

            # Aggregate
            set_id = set_ids[set_name]
            
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros((len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda")
                
            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            metric_global_batch_size[set_id] += global_batch_size

        if len(all_preds) and config.checkpoint_path is not None:
            for k, vlist in all_preds.items():
                all_preds[k] = torch.cat(vlist, dim=1) if k in ("intermediate_logits","intermediate_preds") else torch.cat(vlist, dim=0)

            os.makedirs(config.checkpoint_path, exist_ok=True)
            torch.save(all_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}"))

        # Logging
        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)
            
            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {set_name: {metric_name: reduced_metrics[set_id, metric_id] for metric_id, metric_name in enumerate(metric_keys)}
                                   for set_id, set_name in enumerate(set_ids)}
                
                # Postprocess
                for set_name, metrics in reduced_metrics.items():
                    count = metrics.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in metrics.items()}

                return reduced_metrics


def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_path).capitalize()} ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)

    # Train state
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)

        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)

    # Resume from latest full-state checkpoint if one exists (all ranks read the
    # shared file). Then fast-forward the deterministic dataloader past the steps
    # already done so data order continues exactly where it stopped.
    resume_step = try_resume(config, train_state)
    if resume_step > 0:
        print(f"[Rank {RANK}] resuming from step {resume_step}/{train_state.total_steps}")
        if RANK == 0:
            progress_bar.update(resume_step)  # type: ignore
    processed = 0  # train batches consumed (for fast-forward alignment)

    # Training Loop
    for _iter_id in range(total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        train_state.model.train()
        block_has_real = False
        for set_name, batch, global_batch_size in train_loader:
            # Fast-forward: skip batches already completed before the resume point.
            if processed < resume_step:
                processed += 1
                continue
            processed += 1
            block_has_real = True

            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore

            # Periodic full-state checkpoint for preemption resilience.
            if RANK == 0 and config.checkpoint_steps and (train_state.step % config.checkpoint_steps == 0):
                save_full_state(config, train_state)

        # If this whole block was skipped during fast-forward, don't eval/save.
        if not block_has_real:
            continue

        ############ Checkpointing (save BEFORE eval so the checkpoint is never gated by
        ############ the cost of evaluate(); eval is read-only and does not change weights).
        if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
            save_train_state(config, train_state)   # model-only step_<N> (for downstream collection)
            save_full_state(config, train_state)    # full state (for resume)

        ############ Evaluation (optionally skipped). The full augmented eval set can be
        ############ ~165k examples × halt_max_steps forward passes = hours; set
        ############ HRM_SKIP_EVAL=1 when an external verifier is used instead
        ############ (e.g. scripts/arc/measure_arc_accuracy.py for the ARC adaptation).
        if os.environ.get("HRM_SKIP_EVAL", "0") != "1":
            train_state.model.eval()
            metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
