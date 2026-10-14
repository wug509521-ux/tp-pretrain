from __future__ import annotations

"""
输入文本构建工具。

支持根据配置构建不同的输入格式，包括可选的 reaction 文本拼接。
"""

from typing import Dict, Optional


def build_input_text(
    substrate_smiles: str,
    domain: Optional[str] = None,
    source_dataset: Optional[str] = None,
    raw_reaction: Optional[str] = None,
    use_reaction_text: bool = False,
) -> str:
    """
    根据配置构建模型输入文本。

    当 use_reaction_text=True 且 raw_reaction 存在时，格式为：
        [DOMAIN=xxx] [SOURCE=xxx] [REACTION_TEXT] xxx [SUBSTRATE] substrate_smiles

    当 use_reaction_text=False 时，格式为：
        substrate_smiles（或根据 domain/source 添加条件 token）

    Args:
        substrate_smiles: 底物 SMILES
        domain: 领域（如 "biotransformation"）
        source_dataset: 数据来源（如 "EAWAG_BBD"）
        raw_reaction: 原始 reaction 文本（来自 metadata["raw_reaction"]）
        use_reaction_text: 是否使用 reaction 文本

    Returns:
        构建好的输入文本字符串
    """
    parts: list[str] = []

    if use_reaction_text and raw_reaction:
        # 构建包含 reaction 文本的格式
        if domain:
            parts.append(f"[DOMAIN={domain}]")
        if source_dataset:
            parts.append(f"[SOURCE={source_dataset}]")
        parts.append("[REACTION_TEXT]")
        parts.append(raw_reaction)
        parts.append("[SUBSTRATE]")
        parts.append(substrate_smiles)
    else:
        # 原有格式：直接使用 substrate_smiles
        # 可选：根据 domain/source 添加条件 token（当前保持简单）
        parts.append(substrate_smiles)

    return " ".join(parts)


__all__ = ["build_input_text"]

