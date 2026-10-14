from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping


@dataclass
class RuleRecord:
    """
    通用的规则 / 模板记录结构。

    仅包含与规则本身相关的核心字段，其它信息（例如示例分子、统计信息等）
    放在 metadata 中，方便兼容不同规则库（RetroRules、MicrobeRX 等）。
    """

    rule_id: str
    source: str
    radius: int
    reaction_smarts: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuleRecord":
        return cls(
            rule_id=str(data.get("rule_id", "")),
            source=str(data.get("source", "")),
            radius=int(data.get("radius", 0)),
            reaction_smarts=str(data.get("reaction_smarts", "")),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "source": self.source,
            "radius": int(self.radius),
            "reaction_smarts": self.reaction_smarts,
            "metadata": dict(self.metadata),
        }


__all__ = ["RuleRecord"]



