from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Type, TypeVar

from tp_pretrain.data.schema import TPReactionRecord

T = TypeVar("T", bound=TPReactionRecord)


def _ensure_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def read_jsonl_records(
    path: str | Path,
    record_cls: Type[T] = TPReactionRecord,
) -> List[T]:
    """
    从 JSONLines 文件中读取 TPReactionRecord 列表。

    每一行应为一个 JSON 对象，字段兼容 TPReactionRecord.from_dict。
    """
    p = _ensure_path(path)
    records: List[T] = []

    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {p}") from exc

            rec = record_cls.from_dict(obj)
            records.append(rec)

    return records


def write_jsonl_records(
    path: str | Path,
    records: Sequence[TPReactionRecord],
) -> None:
    """
    将 TPReactionRecord 序列写入 JSONLines 文件。

    每个 record 占一行，使用 UTF-8 编码。
    """
    p = _ensure_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            obj = rec.to_dict()
            json_str = json.dumps(obj, ensure_ascii=False)
            f.write(json_str + "\n")


__all__ = [
    "TPReactionRecord",
    "read_jsonl_records",
    "write_jsonl_records",
]



