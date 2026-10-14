from __future__ import annotations

"""
规则库索引。

当前实现为简单的内存列表结构，侧重于接口设计：
- 支持从 CSV / JSONL 加载规则
- 保存每条规则及其示例分子 SMILES（存入 metadata["example_smiles"]）

后续可以替换为更高效的索引（向量检索、分片加载等）。
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .schema import RuleRecord


class RuleLibraryIndex:
    """
    规则库索引的最小实现。

    - rules: List[RuleRecord]
    - example_smiles: List[Optional[str]] 与 rules 对齐，用于相似度检索
    """

    def __init__(self, rules: Sequence[RuleRecord], example_smiles: Sequence[Optional[str]]) -> None:
        assert len(rules) == len(example_smiles)
        self.rules: List[RuleRecord] = list(rules)
        self.example_smiles: List[Optional[str]] = list(example_smiles)

    # ------------------------------------------------------------------
    # 加载函数
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        rule_id_col: str = "rule_id",
        source_col: str = "source",
        radius_col: str = "radius",
        reaction_smarts_col: str = "reaction_smarts",
        example_smiles_col: str = "example_smiles",
        metadata_cols: Optional[Sequence[str]] = None,
    ) -> "RuleLibraryIndex":
        p = Path(path)
        df = pd.read_csv(p)
        rules: List[RuleRecord] = []
        examples: List[Optional[str]] = []
        metadata_cols = list(metadata_cols or [])

        for _, row in df.iterrows():
            meta: Dict[str, Any] = {}
            if metadata_cols:
                for c in metadata_cols:
                    if c in row:
                        meta[c] = row[c]

            example_smi: Optional[str] = None
            if example_smiles_col in row:
                val = row[example_smiles_col]
                example_smi = str(val).strip() if pd.notna(val) else None
                if example_smi:
                    meta.setdefault("example_smiles", example_smi)

            rec = RuleRecord(
                rule_id=str(row.get(rule_id_col, "")),
                source=str(row.get(source_col, "")),
                radius=int(row.get(radius_col, 0)),
                reaction_smarts=str(row.get(reaction_smarts_col, "")),
                metadata=meta,
            )
            rules.append(rec)
            examples.append(example_smi)

        return cls(rules=rules, example_smiles=examples)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        example_smiles_field: str = "example_smiles",
    ) -> "RuleLibraryIndex":
        p = Path(path)
        rules: List[RuleRecord] = []
        examples: List[Optional[str]] = []

        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                meta = dict(obj.get("metadata", {}) or {})

                example_smi_val = obj.get(example_smiles_field)
                example_smi: Optional[str] = None
                if example_smi_val is not None:
                    example_smi = str(example_smi_val).strip()
                    if example_smi:
                        meta.setdefault("example_smiles", example_smi)

                rec = RuleRecord(
                    rule_id=str(obj.get("rule_id", "")),
                    source=str(obj.get("source", "")),
                    radius=int(obj.get("radius", 0)),
                    reaction_smarts=str(obj.get("reaction_smarts", "")),
                    metadata=meta,
                )
                rules.append(rec)
                examples.append(example_smi)

        return cls(rules=rules, example_smiles=examples)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.rules)

    def iter_records(self) -> Iterable[RuleRecord]:
        return iter(self.rules)

    def get_example_smiles(self, idx: int) -> Optional[str]:
        return self.example_smiles[idx]

    def get(self, idx: int) -> RuleRecord:
        return self.rules[idx]


__all__ = ["RuleLibraryIndex"]



