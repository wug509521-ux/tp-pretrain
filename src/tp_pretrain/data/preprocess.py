from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from tp_pretrain.data.schema import TPReactionRecord


LOGGER = logging.getLogger(__name__)


def _normalize_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[float, float, float]:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Sum of train_ratio, val_ratio and test_ratio must be > 0")
    return train_ratio / total, val_ratio / total, test_ratio / total


def _compute_split_sizes(
    n: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[int, int, int]:
    train_ratio, val_ratio, test_ratio = _normalize_ratios(train_ratio, val_ratio, test_ratio)
    train_n = int(round(n * train_ratio))
    val_n = int(round(n * val_ratio))
    assigned = train_n + val_n
    test_n = max(0, n - assigned)
    return train_n, val_n, test_n


def random_split_records(
    records: Sequence[TPReactionRecord],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Tuple[List[TPReactionRecord], List[TPReactionRecord], List[TPReactionRecord]]:
    """
    按样本级随机切分 TPReactionRecord。
    """
    n = len(records)
    if n == 0:
        return [], [], []

    train_n, val_n, test_n = _compute_split_sizes(n, train_ratio, val_ratio, test_ratio)
    LOGGER.info(
        "Random split target sizes (approx): train=%d, val=%d, test=%d (total=%d)",
        train_n,
        val_n,
        test_n,
        n,
    )

    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)

    train_idx = idxs[:train_n]
    val_idx = idxs[train_n : train_n + val_n]
    test_idx = idxs[train_n + val_n :]

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]

    LOGGER.info(
        "Random split actual sizes: train=%d, val=%d, test=%d (total=%d)",
        len(train_records),
        len(val_records),
        len(test_records),
        len(records),
    )
    return train_records, val_records, test_records


def _substrate_scaffold(smiles: str) -> str:
    """
    基于 substrate_smiles 计算 Bemis-Murcko scaffold。

    若无法计算 scaffold，则返回特殊 token "NO_SCAFFOLD"。
    """
    if not smiles:
        return "NO_SCAFFOLD"
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "NO_SCAFFOLD"
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return "NO_SCAFFOLD"
        scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
        return scaffold_smiles or "NO_SCAFFOLD"
    except Exception:
        return "NO_SCAFFOLD"


def _flatten_groups(groups: List[Tuple[str, List[TPReactionRecord]]]) -> List[TPReactionRecord]:
    records: List[TPReactionRecord] = []
    for _, group in groups:
        records.extend(group)
    return records


def _group_stats(groups: List[Tuple[str, List[TPReactionRecord]]]) -> Tuple[int, int]:
    """
    返回：
    - scaffold 组数
    - 最大 scaffold 组大小
    """
    if not groups:
        return 0, 0
    return len(groups), max(len(g) for _, g in groups)


def _rebalance_empty_split(
    split_to_groups: Dict[str, List[Tuple[str, List[TPReactionRecord]]]],
    empty_split: str,
    donor_priority: Sequence[str],
) -> bool:
    """
    若某个 split 为空，则从 donor 中移动一个最小 scaffold 组过去。
    仅在 donor 至少拥有 2 个 scaffold 组时才移动，避免 donor 自己被掏空。
    """
    for donor in donor_priority:
        donor_groups = split_to_groups[donor]
        if len(donor_groups) <= 1:
            continue
        donor_groups.sort(key=lambda x: (len(x[1]), x[0]))  # 最小 scaffold 组优先挪走
        moved = donor_groups.pop(0)
        split_to_groups[empty_split].append(moved)
        LOGGER.warning(
            "Rebalanced one scaffold group from %s to empty %s: scaffold=%s, group_size=%d",
            donor,
            empty_split,
            moved[0],
            len(moved[1]),
        )
        return True
    return False


def scaffold_split_records(
    records: Sequence[TPReactionRecord],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Tuple[List[TPReactionRecord], List[TPReactionRecord], List[TPReactionRecord]]:
    """
    基于 substrate_smiles 的 Bemis-Murcko scaffold 进行切分。

    改进点：
    1. scaffold 先分组，再按组大小降序分配；
    2. 使用“最小超额 + 接近目标”的分配策略，而不是简单 train -> val -> test 贪心；
    3. 尽量保证 val/test 非空；
    4. 若极端情况下仍出现空 split，会触发一次最小重平衡。
    """
    n = len(records)
    if n == 0:
        return [], [], []

    train_target, val_target, test_target = _compute_split_sizes(n, train_ratio, val_ratio, test_ratio)
    LOGGER.info(
        "Scaffold split target sizes (approx): train=%d, val=%d, test=%d (total=%d)",
        train_target,
        val_target,
        test_target,
        n,
    )

    # 1) 按 scaffold 分组
    scaffold_to_records: Dict[str, List[TPReactionRecord]] = defaultdict(list)
    for rec in records:
        scf = _substrate_scaffold(rec.substrate_smiles)
        scaffold_to_records[scf].append(rec)

    rng = random.Random(seed)

    grouped_items = list(scaffold_to_records.items())
    # 为了可复现，先打散，再按大小降序
    rng.shuffle(grouped_items)
    grouped_items.sort(key=lambda x: len(x[1]), reverse=True)

    n_scaffolds = len(grouped_items)
    max_group_size = max((len(g) for _, g in grouped_items), default=0)

    LOGGER.info(
        "Scaffold grouping stats: total_scaffolds=%d, max_group_size=%d",
        n_scaffolds,
        max_group_size,
    )

    split_to_groups: Dict[str, List[Tuple[str, List[TPReactionRecord]]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_scaffold_counts = {"train": 0, "val": 0, "test": 0}
    targets = {"train": train_target, "val": val_target, "test": test_target}

    remaining_groups = len(grouped_items)

    for scaffold, group in grouped_items:
        gsize = len(group)
        candidate_splits = ["train", "val", "test"]

        # 2) 约束：尽量保证 val/test 至少各有 1 个 scaffold
        # 如果还没分到 val/test，而剩余 scaffold 数目不多，就优先填补缺失 split
        missing_non_train = [s for s in ("val", "test") if split_scaffold_counts[s] == 0]
        valid_candidates: List[str] = []
        for split_name in candidate_splits:
            remaining_after = remaining_groups - 1
            remaining_missing_after = sum(
                1
                for s in missing_non_train
                if not (split_name == s and split_scaffold_counts[s] == 0)
            )
            if remaining_after >= remaining_missing_after:
                valid_candidates.append(split_name)

        if not valid_candidates:
            valid_candidates = candidate_splits

        # 3) 打分：优先不超 target；其次尽量接近 target；再优先给 scaffold 数较少的集合
        best_split = None
        best_score = None

        for split_name in valid_candidates:
            current = split_counts[split_name]
            target = targets[split_name]
            new_count = current + gsize

            overflow = max(0, new_count - target)
            distance = abs(new_count - target)
            underfull_flag = 0 if new_count <= target else 1  # 先偏好不超额
            scaffold_n = split_scaffold_counts[split_name]

            score = (
                underfull_flag,  # 0 better
                overflow,        # smaller better
                distance,        # smaller better
                scaffold_n,      # fewer scaffolds better
                current,         # smaller split better
            )

            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name

        assert best_split is not None

        split_to_groups[best_split].append((scaffold, group))
        split_counts[best_split] += gsize
        split_scaffold_counts[best_split] += 1
        remaining_groups -= 1

    # 4) 若 val/test 为空，做一次最小重平衡
    if n_scaffolds >= 3:
        if split_scaffold_counts["val"] == 0:
            ok = _rebalance_empty_split(split_to_groups, "val", donor_priority=("train", "test"))
            if ok:
                split_scaffold_counts["val"] = len(split_to_groups["val"])
                split_scaffold_counts["train"] = len(split_to_groups["train"])
                split_scaffold_counts["test"] = len(split_to_groups["test"])

        if split_scaffold_counts["test"] == 0:
            ok = _rebalance_empty_split(split_to_groups, "test", donor_priority=("train", "val"))
            if ok:
                split_scaffold_counts["val"] = len(split_to_groups["val"])
                split_scaffold_counts["train"] = len(split_to_groups["train"])
                split_scaffold_counts["test"] = len(split_to_groups["test"])

    train_records = _flatten_groups(split_to_groups["train"])
    val_records = _flatten_groups(split_to_groups["val"])
    test_records = _flatten_groups(split_to_groups["test"])

    LOGGER.info(
        "Scaffold split actual sizes: train=%d, val=%d, test=%d "
        "(total=%d, scaffolds=%d, scaffold_counts: train=%d, val=%d, test=%d, max_group_size=%d)",
        len(train_records),
        len(val_records),
        len(test_records),
        len(records),
        n_scaffolds,
        len(split_to_groups["train"]),
        len(split_to_groups["val"]),
        len(split_to_groups["test"]),
        max_group_size,
    )

    if len(val_records) == 0 or len(test_records) == 0:
        LOGGER.warning(
            "Scaffold split still produced an empty split: train=%d, val=%d, test=%d. "
            "For small datasets, consider using random split or implementing a stricter rebalance mode.",
            len(train_records),
            len(val_records),
            len(test_records),
        )

    return train_records, val_records, test_records


def ensure_unique_reaction_ids(
    train_records: Sequence[TPReactionRecord],
    val_records: Sequence[TPReactionRecord],
    test_records: Sequence[TPReactionRecord],
) -> None:
    """
    确保同一 reaction_id 不会同时出现在多个集合中。
    """
    train_ids = {r.reaction_id for r in train_records}
    val_ids = {r.reaction_id for r in val_records}
    test_ids = {r.reaction_id for r in test_records}

    inter_train_val = train_ids & val_ids
    inter_train_test = train_ids & test_ids
    inter_val_test = val_ids & test_ids

    duplicated = inter_train_val | inter_train_test | inter_val_test
    if duplicated:
        LOGGER.error("Found duplicated reaction_ids across splits: %s", sorted(duplicated))
        raise ValueError("Duplicated reaction_ids found in different splits.")


def split_reaction_records(
    records: Sequence[TPReactionRecord],
    mode: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Tuple[List[TPReactionRecord], List[TPReactionRecord], List[TPReactionRecord]]:
    """
    高层接口：根据 split mode 对记录进行切分。

    mode:
    - "random": 样本级随机切分
    - "scaffold": 基于 Bemis-Murcko scaffold 的切分
    """
    mode = mode.lower()
    if mode not in {"random", "scaffold"}:
        raise ValueError(f"Unsupported split mode: {mode}")

    if mode == "random":
        train_records, val_records, test_records = random_split_records(
            records=records,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
    else:
        train_records, val_records, test_records = scaffold_split_records(
            records=records,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    ensure_unique_reaction_ids(train_records, val_records, test_records)
    return train_records, val_records, test_records


__all__ = [
    "random_split_records",
    "scaffold_split_records",
    "split_reaction_records",
    "ensure_unique_reaction_ids",
]