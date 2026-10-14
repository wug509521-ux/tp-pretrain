from __future__ import annotations

"""
最小版 seq2seq Trainer，实现：
- 纯生成损失训练
- 生成损失 + reaction center 辅助分类损失训练

注意：
- 当前 baseline 使用纯文本 seq2seq 模型；
- 辅助头在 encoder pooled representation 上做简单多分类，可替换为更复杂的图模型。
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase, get_linear_schedule_with_warmup

from tp_pretrain.data.collator import Seq2SeqCollator
from tp_pretrain.models import AuxHeadConfig, ReactionCenterAuxHead
from tp_pretrain.training.losses import LossConfig, combine_losses


LOGGER = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    num_epochs: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    use_aux_loss: bool = False
    aux_loss_weight: float = 1.0
    log_steps: int = 10


class Seq2SeqTrainer:
    """
    简化版训练循环，适合作为后续扩展的骨架。
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        trainer_config: Optional[TrainerConfig] = None,
        device: Optional[torch.device] = None,
        aux_head: Optional[ReactionCenterAuxHead] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = trainer_config or TrainerConfig()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 训练过程中的历史记录（用于保存 train_log.csv / metrics.json）
        self.history: list[dict] = []

        # 构建 reaction center 辅助头
        hidden_size = getattr(self.model.config, "d_model", None) or getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Cannot infer encoder hidden size from model.config (expected d_model or hidden_size).")

        if aux_head is None:
            head_cfg = AuxHeadConfig(hidden_size=hidden_size, num_classes=4, loss_weight=self.config.aux_loss_weight)
            aux_head = ReactionCenterAuxHead(head_cfg)
        self.aux_head = aux_head.to(self.device)

        self.loss_config = LossConfig(
            use_aux_loss=self.config.use_aux_loss,
            aux_loss_weight=self.config.aux_loss_weight,
        )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        total_steps = len(self.train_dataloader) * self.config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

    def _compute_aux_loss_and_acc(self, encoder_pooled, batch) -> tuple[Optional[torch.Tensor], Optional[float]]:
        """
        基于 encoder pooled representation 计算 reaction center 辅助损失与准确率。
        """
        if not self.loss_config.use_aux_loss:
            return None, None

        labels = batch.get("som_count_bucket")
        if labels is None:
            return None, None

        aux_out = self.aux_head(encoder_pooled, labels)
        aux_loss: torch.Tensor = aux_out["loss"]

        with torch.no_grad():
            preds = aux_out["logits"].argmax(dim=-1)
            correct = (preds == labels).sum().item()
            total = labels.numel()
            acc = correct / total if total > 0 else 0.0

        return aux_loss, acc

    def train(self) -> None:
        global_step = 0
        self.model.train()

        for epoch in range(self.config.num_epochs):
            LOGGER.info("Starting epoch %d / %d", epoch + 1, self.config.num_epochs)

            epoch_aux_correct = 0
            epoch_aux_total = 0

            for step, batch in enumerate(self.train_dataloader):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                gen_loss: torch.Tensor = outputs.loss

                # encoder pooled representation：简单地对 encoder_last_hidden_state 做 mean-pooling
                encoder_hidden = outputs.encoder_last_hidden_state  # [B, L, H]
                encoder_pooled = encoder_hidden.mean(dim=1)

                aux_loss, aux_acc = self._compute_aux_loss_and_acc(encoder_pooled, batch)
                total_loss = combine_losses(gen_loss, aux_loss, self.loss_config)

                loss = total_loss / self.config.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    if aux_acc is not None:
                        epoch_aux_correct += aux_acc * batch["som_count_bucket"].numel()
                        epoch_aux_total += batch["som_count_bucket"].numel()

                    # 记录历史
                    record = {
                        "global_step": global_step,
                        "loss": float(total_loss.item()),
                        "gen_loss": float(gen_loss.item()),
                        "aux_loss": float(aux_loss.item()) if aux_loss is not None else None,
                        "aux_acc": float(aux_acc) if aux_acc is not None else None,
                    }
                    self.history.append(record)

                    if global_step % self.config.log_steps == 0:
                        aux_val = record["aux_loss"] if record["aux_loss"] is not None else 0.0
                        acc_val = record["aux_acc"] if record["aux_acc"] is not None else 0.0
                        LOGGER.info(
                            "step=%d, loss=%.4f, gen_loss=%.4f, aux_loss=%.4f, aux_acc=%.4f",
                            global_step,
                            record["loss"],
                            record["gen_loss"],
                            aux_val,
                            acc_val,
                        )

            # 这里可以逐步扩展验证逻辑（val_dataloader），并统计 epoch 级别的 aux_acc


__all__ = ["TrainerConfig", "Seq2SeqTrainer"]


