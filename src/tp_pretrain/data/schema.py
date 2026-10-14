from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional


@dataclass
class TPReactionRecord:
    """
    统一表示“底物 -> 单步转化产物”的记录。

    该 schema 设计为后续可扩展更多条件、标签（如 SOM、模板、规则来源等）。
    """

    reaction_id: str
    substrate_smiles: str
    product_smiles: str
    domain: str
    biosystem: Optional[str]
    condition: Dict[str, Any]
    template_id: Optional[str]
    rule_source: Optional[str]
    som_atoms: List[int] = field(default_factory=list)
    depth: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_dataset: Optional[str] = None  # 数据来源标识，如 "EAWAG_BBD" 或 "Sludge"

    # ------------------------------------------------------------------
    # 构造 & 序列化
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TPReactionRecord":
        """
        从原始 dict 构造一个 TPReactionRecord。

        允许缺省部分可选字段，并提供合理的默认值。
        """
        return cls(
            reaction_id=str(data.get("reaction_id", "")),
            substrate_smiles=str(data.get("substrate_smiles", "")).strip(),
            product_smiles=str(data.get("product_smiles", "")).strip(),
            domain=str(data.get("domain", "")).strip(),
            biosystem=data.get("biosystem"),
            condition=dict(data.get("condition", {}) or {}),
            template_id=data.get("template_id"),
            rule_source=data.get("rule_source"),
            som_atoms=list(data.get("som_atoms", []) or []),
            depth=int(data.get("depth", 1)),
            metadata=dict(data.get("metadata", {}) or {}),
            source_dataset=data.get("source_dataset"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        将记录转换为易于 JSON/JSONL 序列化的 dict。
        """
        return {
            "reaction_id": self.reaction_id,
            "substrate_smiles": self.substrate_smiles,
            "product_smiles": self.product_smiles,
            "domain": self.domain,
            "biosystem": self.biosystem,
            "condition": dict(self.condition),
            "template_id": self.template_id,
            "rule_source": self.rule_source,
            "som_atoms": list(self.som_atoms),
            "depth": int(self.depth),
            "metadata": dict(self.metadata),
            "source_dataset": self.source_dataset,
        }

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """
        对记录进行基础校验。

        要求：
        - substrate_smiles 非空
        - product_smiles 非空
        - depth >= 1
        """
        if not self.substrate_smiles:
            raise ValueError("substrate_smiles must be non-empty")

        if not self.product_smiles:
            raise ValueError("product_smiles must be non-empty")

        if self.depth < 1:
            raise ValueError("depth must be >= 1")

        # 可以在此处逐步扩展更复杂的校验逻辑（如 SMILES 合法性、SOM 范围等）


JsonMapping = MutableMapping[str, Any]


def normalize_record_dict(d: Mapping[str, Any]) -> Dict[str, Any]:
    """
    将任意 Mapping 正规化为常规 dict，方便 JSON 序列化或后续处理。
    """
    return dict(d)



