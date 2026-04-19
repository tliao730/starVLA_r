# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA's trainer is built directly on native PyTorch + Accelerate + DeepSpeed,
keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if-else chains).

這支檔案是「StarVLA + LoRA」版本的訓練入口:
- 使用 Accelerate 封裝分散式/混合精度/DeepSpeed。
- 依照 YAML + CLI dotlist 組合出 cfg。
- 依 cfg 建立 framework(model) 與 dataloader。
- (可選) 將 LoRA adapters 注入到指定 target_modules。
- 建立 optimizer / scheduler。
- 進入顯式的 train loop, 週期性 eval / log / save。

"""

# Standard Library
import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.config_tracker import (
    AccessTrackedConfig,
    wrap_config,
)
from starVLA.training.trainer_utils.trainer_tools import (
    TrainerUtils,
    build_param_lr_groups,
    normalize_dotlist_args,
)

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def _infer_all_linear_target_module_suffixes(
    model: torch.nn.Module,
) -> list[str]:
    """Infer a PEFT `target_modules` list that approximates "all linear layers".

    This is used as a fallback when PEFT does not support passing
    `target_modules="all-linear"`.
    """
    suffixes: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not name:
            continue
        suffix = name.split(".")[-1]
        # Filter out purely numeric suffixes from Sequential-like modules.
        if suffix.isdigit():
            continue
        suffixes.add(suffix)
    return sorted(suffixes)


def _freeze_modules_lora_safe(
    model: torch.nn.Module, freeze_modules: str | None
) -> list[str]:
    """Freeze module paths while keeping LoRA adapter params trainable.

    Notes:
        We intentionally implement this locally so we don't need to modify
        `trainer_tools.py`. This helper is only used when `use_lora=true`.
    """
    if not freeze_modules or not isinstance(freeze_modules, str):
        return []

    frozen: list[str] = []
    patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]
    for path in patterns:
        module = model
        try:
            for attr in path.split("."):
                module = getattr(module, attr)
        except AttributeError:
            logger.warning(
                f"freeze module path does not exist, cannot freeze: {path}"
            )
            continue

        for name, param in module.named_parameters():
            lowered = name.lower()
            if "lora" in lowered:
                continue
            param.requires_grad = False
        frozen.append(path)

    return frozen


def load_fast_tokenizer():
    """載入 fast tokenizer / processor。

    注意: 此檔案目前未直接使用 tokenizer, 但保留接口便於未來擴充。
    """
    return AutoProcessor.from_pretrained(
        "physical-intelligence/fast", trust_remote_code=True
    )


def setup_directories(cfg) -> Path:
    """建立輸出資料夾與 checkpoint 資料夾。

    分散式訓練時只讓 rank0 建立資料夾, 避免競態。
    """
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """建立 VLA 訓練 dataloader。

    - `build_dataloader` 會依 cfg.datasets.vla_data.* 產生對應資料混合與 sampling。
    - Accelerate 的 `dispatch_batches=False` 通常用於「每個 process 自己拿 batch」而非由主進程分發。

    注意: 此函式呼叫了 `dist.barrier()`, 代表呼叫前需要已 init_process_group。
    """
    logger.info(
        f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`"
    )
    vla_train_dataloader = build_dataloader(
        cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py
    )

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(
    model, cfg
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """建立 optimizer 與 learning-rate scheduler。

    這裡透過 `build_param_lr_groups` 將不同 module 分成不同 lr group (例如 backbone vs head)。
    """
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    if dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(
                f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}"
            )

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(
        self,
        cfg,
        model,
        vla_train_dataloader,
        optimizer,
        lr_scheduler,
        accelerator,
        lora_freeze_modules: str | None = None,
    ):
        # 訓練主狀態: cfg/model/dataloader/optimizer/scheduler/accelerator
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        # When `use_lora=true`, we temporarily clear `cfg.trainer.freeze_modules` before
        # building the optimizer, to prevent LoRA params from being excluded by
        # param-group construction utilities. We keep the original freeze list here.
        self.lora_freeze_modules = lora_freeze_modules

        # completed_steps: 只在 optimizer step (sync_gradients=True) 時增加
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

    def prepare_training(self):
        """訓練前準備: seed、checkpoint、freeze、distributed prepare、wandb。"""
        rank = dist.get_rank()
        seed = (
            self.config.seed + rank
            if hasattr(self.config, "seed")
            else rank + 3047
        )
        set_seed(seed)

        # 1) 決定是否 resume / load pretrained
        self._init_checkpointing()

        # 2) 若 resume 且 completed_steps>0, 先把 scheduler step 到正確位置
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = None
        if (
            self.config
            and hasattr(self.config, "trainer")
            and hasattr(self.config.trainer, "freeze_modules")
        ):
            freeze_modules = self.config.trainer.freeze_modules

        # 3) Freeze
        # - LoRA mode: freeze base modules but keep LoRA params trainable.
        # - Non-LoRA mode: keep original TrainerUtils behavior.
        if self.config.get("use_lora", False):
            frozen_paths = _freeze_modules_lora_safe(
                self.model, self.lora_freeze_modules or freeze_modules
            )
            if dist.get_rank() == 0:
                logger.info(f"🔒 Frozen modules (LoRA-safe): {frozen_paths}")
        else:
            self.model = self.freeze_backbones(
                self.model, freeze_modules=freeze_modules
            )

        if dist.get_rank() == 0:
            self.print_trainable_parameters(self.model)

        # 4) 交給 Accelerate 處理 DDP/DeepSpeed/mixed precision 等包裝
        self.model, self.optimizer, self.vla_train_dataloader = (
            self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
            )
        )

        # 5) 只在 main process 初始化 wandb (避免重複創建 run)
        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        # Global batch size = per_device_batch_size * num_processes * grad_accum
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if dist.get_rank() == 0:
            # dir: 把 wandb 檔案寫到 run output 之下, 便於收集/同步
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(
            self.config.output_dir, "checkpoints"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(
            self.config.trainer, "pretrained_checkpoint", None
        )
        is_resume = getattr(self.config.trainer, "is_resume", False)
        if is_resume:
            # 從 checkpoints/ 找最新 steps_xxx
            resume_from_checkpoint, self.completed_steps = (
                self._get_latest_checkpoint(self.checkpoint_dir)
            )
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint

                # 這裡是「只載入 backbones」的邏輯 (由 TrainerUtils 實作), 通常用於避免重載 optimizer 狀態
                self.model = self.load_pretrained_backbones(
                    self.model, self.resume_from_checkpoint, reload_modules=None
                )
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(
                f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch."
            )
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(
                self.config.trainer, "reload_modules", None
            )

            # 從指定 checkpoint 載入權重 (可指定只載入部分模組)
            self.model = self.load_pretrained_backbones(
                self.model, pretrained_checkpoint, reload_modules=reload_modules
            )
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(
                f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}"
            )
        else:
            logger.info(
                "No pretrained checkpoint provided. Starting training from scratch."
            )
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(
                f"Adjusting LR scheduler for resume from step {self.completed_steps}"
            )
            # 很多 scheduler (例如 linear/cosine) 依 step 更新; resume 後需要對齊
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        # 若你有用 accelerate.save_state / load_state, 這裡會恢復 optimizer/scheduler 等狀態
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Save current training state."""
        if dist.get_rank() == 0:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"steps_{self.completed_steps}"
            )

            # Accelerator 統一處理各種 wrapper 下正確的 state_dict
            state_dict = self.accelerator.get_state_dict(self.model)

            if self.config.get("use_lora", False):
                # LoRA 模式下只保存 adapter 權重 (更小, 更易於合併/部署)
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                state_dict = get_peft_model_state_dict(
                    unwrapped_model, state_dict=state_dict
                )

            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(
                    f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`."
                )

            summary_data = {"steps": self.completed_steps}
            with open(
                os.path.join(self.config.output_dir, "summary.jsonl"), "a"
            ) as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                # AccessTrackedConfig 會記錄「訓練過程中實際被讀取過的 cfg key」
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(
                    output_dir / "config.yaml", use_original_values=False
                )
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if (
            self.completed_steps % self.config.trainer.logging_frequency == 0
            and dist.get_rank() == 0
        ):
            # learning_rate: 取第一組 lr (若有多 group, 可在這裡擴展記錄)
            metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            metrics["epoch"] = round(
                self.completed_steps / len(self.vla_train_dataloader), 2
            )
            wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        # 用 iterator 的好處: 可以自訂 StopIteration 後的 epoch 重置行為
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            # 一個 epoch 跑完: 重置 dataloader 並增加 epoch 計數
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = (
                TrainerUtils._reset_dataloader(
                    self.vla_train_dataloader, self.vla_epoch_count
                )
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()

        # tqdm 只在 local main process 顯示, 避免多進程輸出互相干擾
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                # 只有在 sync_gradients=True (累積步數到達) 時才算完成一個 optimization step
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                # 週期性做一個簡單 action-eval (當前 batch 上的 MSE)
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            if (
                self.completed_steps % self.config.trainer.save_interval == 0
                and self.completed_steps > 0
            ):
                # 週期性保存 checkpoint (可選 pt/safetensors; LoRA 只存 adapters)
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(
        self, step_metrics: dict | None = None
    ) -> dict | None:
        """Run simple action-eval on current batch and attach score to metrics."""
        # 這裡用下一個 batch 做快速 sanity-check: 預測 action 與 GT action 的距離
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        # output_dict = self.model.predict_action(examples=examples, use_ddim=True, num_ddim_steps=20)

        # 若 model 被 accelerate/deepspeed 包裝, 做推理/呼叫自訂方法時常需要 unwrap
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        output_dict = unwrapped_model.predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )
        if dist.get_rank() == 0:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if dist.get_rank() == 0:
            logger.info("***** Training Configuration *****")
            logger.info(
                f"  Total optimization steps = {self.config.trainer.max_train_steps}"
            )
            logger.info(
                f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}"
            )
            logger.info(
                f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}"
            )
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        with self.accelerator.accumulate(self.model):
            # accumulate 會在 gradient_accumulation_steps 之間自動處理 sync/no_sync
            self.optimizer.zero_grad()

            # bfloat16 autocast: 通常在 A100/H100 等上更穩定也更快
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss

            self.accelerator.backward(total_loss)

            if self.config.trainer.gradient_clipping is not None:
                # clipping 透過 accelerator API, 能兼容各種 wrapper
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.trainer.gradient_clipping,
                )

            self.optimizer.step()
            self.lr_scheduler.step()

        return {
            "action_dit_loss": action_loss.item(),
        }

    def _finalize_training(self):
        """Training end processing."""
        if dist.get_rank() == 0:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(
                self.config.output_dir, "final_model"
            )
            os.makedirs(final_checkpoint, exist_ok=True)

            # 最終導出: 同樣遵循 LoRA only / full model 的保存策略
            state_dict = self.accelerator.get_state_dict(self.model)

            if self.config.get("use_lora", False):
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                state_dict = get_peft_model_state_dict(
                    unwrapped_model, state_dict=state_dict
                )

            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(
                    state_dict,
                    os.path.join(final_checkpoint, "model.safetensors"),
                )
            elif save_format == "pt":
                torch.save(
                    state_dict,
                    os.path.join(final_checkpoint, "pytorch_model.pt"),
                )
            else:
                raise ValueError(
                    f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`."
                )
            logger.info(
                f"Training complete. Final model saved at {final_checkpoint}"
            )

        if dist.get_rank() == 0:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    # wrap_config: 讓 cfg 變成可追蹤 access 的 wrapper (用於導出最小必要 config)
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)

    # build_framework: 建立 StarVLA 的模型框架 (含 encoder/decoder/action head 等)
    vla = build_framework(cfg)

    # IMPORTANT: If we're doing LoRA, we must avoid passing `trainer.freeze_modules` into
    # optimizer param-group building, otherwise LoRA params (which live under the frozen
    # parent module) may be excluded. We clear it temporarily and carry the original
    # value into the trainer to apply LoRA-safe freezing later.
    lora_freeze_modules = None
    if (
        cfg.get("use_lora", False)
        and hasattr(cfg, "trainer")
        and hasattr(cfg.trainer, "freeze_modules")
    ):
        lora_freeze_modules = cfg.trainer.freeze_modules
        cfg.trainer.freeze_modules = ""

    if cfg.get("use_lora", False):
        logger.info("Injecting LoRA Adapters into the model")

        # LoRA defaults (no config required).
        # Default target is the whole model (all Linear layers).
        lora_r = 16
        lora_alpha = 32
        lora_dropout = 0.05
        target_modules: str | list[str] = "all-linear"

        # Preferred default: apply LoRA to all linear layers.
        # If the installed PEFT version doesn't support "all-linear", we fall back to
        # an inferred list of Linear module name suffixes.
        try:
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
            )
        except Exception as e:
            if (
                isinstance(target_modules, str)
                and target_modules == "all-linear"
            ):
                inferred = _infer_all_linear_target_module_suffixes(vla)
                if dist.get_rank() == 0:
                    logger.warning(
                        "PEFT does not accept target_modules='all-linear'; "
                        f"falling back to inferred Linear suffix list (n={len(inferred)})."
                    )
                peft_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=inferred,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
            else:
                raise e

        # if hasattr(vla, "enable_input_require_grads"):
        #    vla.enable_input_require_grads()

        # get_peft_model: 以 PEFT 的方式把 LoRA modules 注入到 model
        vla = get_peft_model(vla, peft_config)

        if dist.get_rank() == 0:
            # 只在 rank0 打印可訓練參數量, 避免輸出爆炸
            vla.print_trainable_parameters()

    vla_train_dataloader = prepare_data(
        cfg=cfg, accelerator=accelerator, output_dir=output_dir
    )
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        lora_freeze_modules=lora_freeze_modules,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="starVLA/config/training/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    # 1) 先載入 YAML
    cfg = OmegaConf.load(args.config_yaml)

    # 2) 再把 CLI 以 dotlist 的方式覆蓋 (例如 trainer.max_train_steps=1000)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    if cfg.is_debug and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
