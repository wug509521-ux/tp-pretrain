from __future__ import annotations

from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import DataStructs


def _mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """内部工具：安全地从 SMILES 构建 RDKit Mol。"""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    return mol


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    将 SMILES 规范化为 RDKit 的 canonical SMILES。

    非法 SMILES 返回 None。
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        can = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None
    return can


def is_valid_smiles(smiles: str) -> bool:
    """
    判断 SMILES 是否可以被 RDKit 正常解析。
    """
    return _mol_from_smiles(smiles) is not None


def standardize_smiles(smiles: str) -> Optional[str]:
    """
    对 SMILES 做简单标准化。

    当前实现为：
    - 使用 RDKit 解析为 Mol
    - 再输出 canonical SMILES

    后续可以在此基础上加入更多标准化步骤（电荷中和值、tautomer 归一化等）。
    """
    # 这里暂时与 canonicalize_smiles 行为相同，方便后续扩展
    return canonicalize_smiles(smiles)


def tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """
    计算两条 SMILES 对应分子的 Tanimoto 相似度（基于 Morgan 指纹）。

    - 使用 radius=2、nBits=2048 的 ECFP（Morgan 指纹）
    - 若任一 SMILES 非法，返回 0.0
    """
    mol1 = _mol_from_smiles(smiles1)
    mol2 = _mol_from_smiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0

    try:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0
    return float(sim)


def molecular_formula(smiles: str) -> Optional[str]:
    """
    计算分子式（如 C2H6O）。

    非法 SMILES 返回 None。
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        formula = rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return None
    return formula


def exact_mass(smiles: str) -> Optional[float]:
    """
    计算精确质量（monoisotopic exact mass）。

    非法 SMILES 返回 None。
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        mass = rdMolDescriptors.CalcExactMolWt(mol)
    except Exception:
        return None
    return float(mass)


__all__ = [
    "canonicalize_smiles",
    "is_valid_smiles",
    "standardize_smiles",
    "tanimoto_similarity",
    "molecular_formula",
    "exact_mass",
]



