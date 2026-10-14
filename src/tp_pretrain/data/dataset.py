from __future__ import annotations

"""
Dataset 定义。

当前提供基于 JSONL / TPReactionRecord 的最小 seq2seq 数据集，
并附带 SOM 标签（som_atoms），用于后续 SOM 辅助监督。
"""

from pathlib import Path
from typing import Dict, List, Sequence

from torch.utils.data import Dataset

from tp_pretrain.data.schema import TPReactionRecord
from tp_pretrain.utils.io import read_jsonl_records


class TPSeq2SeqDataset(Dataset):
    """
    最小的 TP 预测 seq2seq Dataset。

    输出字段：
    - reaction_id: str
    - substrate_smiles: str
    - product_smiles: str
    - som_atom_indices: List[int]  # 底物上的 SOM 原子索引列表
    """

    def __init__(self, jsonl_path: str | Path) -> None:
        super().__init__()
        self.path = Path(jsonl_path)
        self.records: List[TPReactionRecord] = read_jsonl_records(self.path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        return {
            "reaction_id": rec.reaction_id,
            "substrate_smiles": rec.substrate_smiles,
            "product_smiles": rec.product_smiles,
            "som_atom_indices": list(rec.som_atoms or []),
            "domain": rec.domain,
            "source_dataset": rec.source_dataset,
            "metadata": dict(rec.metadata or {}),
        }


__all__ = ["TPSeq2SeqDataset"]



