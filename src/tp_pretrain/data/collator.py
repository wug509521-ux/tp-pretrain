from __future__ import annotations

"""
Collator 定义。

负责将 TPSeq2SeqDataset 的样本 batch 化，喂给 Transformers seq2seq 模型。
当前不对 token 与原子做对齐，只是把 SOM 标签作为额外字段传入，
为未来接入图模型 / SOM 头做准备。
"""

from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import PreTrainedTokenizerBase

from tp_pretrain.models.tokenizer import build_input_text
from contextlib import nullcontext


class Seq2SeqCollator:
    """
    简单的 seq2seq collator：

    输入样本格式（来自 TPSeq2SeqDataset）：
    - reaction_id
    - substrate_smiles
    - product_smiles
    - som_atom_indices
    - domain
    - source_dataset
    - metadata

    输出 batch：
    - input_ids, attention_mask, labels（供文本生成使用）
    - reaction_id: List[str]
    - substrate_smiles: List[str]
    - product_smiles: List[str]
    - som_atom_indices: List[List[int]]  # 保留给 SOM 头使用
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_src_length: int = 256,
        max_tgt_length: int = 256,
        use_reaction_text: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.use_reaction_text = use_reaction_text

    def __call__(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        # 构建输入文本
        input_texts: List[str] = []
        for ex in examples:
            substrate = ex["substrate_smiles"]
            domain = ex.get("domain")
            source_dataset = ex.get("source_dataset")
            metadata = ex.get("metadata", {})
            raw_reaction = metadata.get("raw_reaction") if metadata else None

            input_text = build_input_text(
                substrate_smiles=substrate,
                domain=domain,
                source_dataset=source_dataset,
                raw_reaction=raw_reaction,
                use_reaction_text=self.use_reaction_text,
            )
            input_texts.append(input_text)

        products = [ex["product_smiles"] for ex in examples]

        model_inputs = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.max_src_length,
            return_tensors="pt",
        )

        target_ctx = self.tokenizer.as_target_tokenizer() if hasattr(self.tokenizer, "as_target_tokenizer") else nullcontext()
        with target_ctx:
            labels_enc = self.tokenizer(
                products,
                padding=True,
                truncation=True,
                max_length=self.max_tgt_length,
                return_tensors="pt",
            )

        batch: Dict[str, Any] = dict(model_inputs)
        batch["labels"] = labels_enc["input_ids"]

        substrates = [ex["substrate_smiles"] for ex in examples]
        batch["reaction_id"] = [ex["reaction_id"] for ex in examples]
        batch["substrate_smiles"] = substrates
        batch["product_smiles"] = products

        som_indices: List[List[int]] = [list(ex.get("som_atom_indices", [])) for ex in examples]
        batch["som_atom_indices"] = som_indices

        # 从 som_atom_indices 派生粗粒度 reaction-center 标签
        # has_som: 0/1
        # som_count_bucket: 0 (0), 1 (1), 2 (2), 3 (>=3)
        has_som_labels: List[int] = []
        som_count_buckets: List[int] = []
        for atoms in som_indices:
            count = len(atoms)
            has_som_labels.append(1 if count > 0 else 0)
            if count == 0:
                bucket = 0
            elif count == 1:
                bucket = 1
            elif count == 2:
                bucket = 2
            else:
                bucket = 3
            som_count_buckets.append(bucket)

        batch["has_som_label"] = torch.tensor(has_som_labels, dtype=torch.long)
        batch["som_count_bucket"] = torch.tensor(som_count_buckets, dtype=torch.long)

        return batch


__all__ = ["Seq2SeqCollator"]

