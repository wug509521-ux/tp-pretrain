from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from tp_pretrain.data.schema import TPReactionRecord
from tp_pretrain.models.tokenizer import build_input_text
from tp_pretrain.utils.chemistry import canonicalize_smiles, is_valid_smiles


LOGGER = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    model_dir: str
    num_beams: int = 4
    num_return_sequences: int = 4
    max_new_tokens: int = 128


class Seq2SeqTPPredictor:
    """
    最小 seq2seq baseline 预测器。

    默认假设输入格式为 SMILES-to-SMILES，即直接以 substrate_smiles 作为输入序列。
    """

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

        model_dir = Path(config.model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        LOGGER.info("Loading tokenizer and model from %s", model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        if self.config.num_return_sequences > self.config.num_beams:
            LOGGER.warning(
                "num_return_sequences (%d) > num_beams (%d); "
                "setting num_beams = num_return_sequences.",
                self.config.num_return_sequences,
                self.config.num_beams,
            )
            self.config.num_beams = self.config.num_return_sequences

    def _build_input_text(self, record: TPReactionRecord) -> str:
        """
        构建模型输入文本。

        当前简化为直接使用 substrate_smiles，后续可加入 domain / 条件等前缀。
        """
        return record.substrate_smiles

    def generate_for_record(self, record: TPReactionRecord) -> Dict:
        """
        为单条记录生成 top-k 产物候选。

        返回结构：
        {
          "reaction_id": ...,
          "substrate_smiles": ...,
          "gold_product_smiles": ... 或 None,
          "predictions": [
             {"rank": 1, "predicted_smiles": "...", "score": float},
             ...
          ]
        }
        """
        input_text = self._build_input_text(record)

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_beams=self.config.num_beams,
                num_return_sequences=self.config.num_return_sequences,
                max_new_tokens=self.config.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )

        sequences = outputs.sequences  # shape: (num_return_sequences, seq_len)
        seq_scores = getattr(outputs, "sequences_scores", None)

        predictions: List[Dict] = []
        seen_smiles = set()

        for rank, seq in enumerate(sequences, start=1):
            decoded = self.tokenizer.decode(seq, skip_special_tokens=True)
            decoded = decoded.strip()
            if not decoded:
                continue

            # 过滤非法 SMILES，尽量 canonical 化
            if not is_valid_smiles(decoded):
                can = canonicalize_smiles(decoded)
            else:
                can = canonicalize_smiles(decoded)

            if can is None or not is_valid_smiles(can):
                # 非法预测，跳过
                continue

            if can in seen_smiles:
                # 去重：同一 canonical SMILES 只保留一次（rank 更高者）
                continue

            seen_smiles.add(can)

            score_val: Optional[float] = None
            if seq_scores is not None:
                # sequences_scores 通常是 log-prob，转 float 输出
                try:
                    score_val = float(seq_scores[rank - 1].item())
                except Exception:
                    score_val = None

            predictions.append(
                {
                    "rank": rank,
                    "predicted_smiles": can,
                    "score": score_val,
                }
            )

        return {
            "reaction_id": record.reaction_id,
            "substrate_smiles": record.substrate_smiles,
            "gold_product_smiles": record.product_smiles or None,
            "predictions": predictions,
        }

    def generate_for_records(self, records: List[TPReactionRecord]) -> List[Dict]:
        """
        为一批记录生成预测结果，并统计合法预测的比例。
        """
        results: List[Dict] = []
        total_preds = 0
        valid_preds = 0

        for rec in records:
            out = self.generate_for_record(rec)
            results.append(out)
            preds = out.get("predictions", [])
            total_preds += self.config.num_return_sequences
            valid_preds += len(preds)

        valid_rate = (valid_preds / total_preds) if total_preds > 0 else 0.0
        LOGGER.info(
            "Generation finished: total_predictions=%d, valid_predictions=%d, valid_rate=%.3f",
            total_preds,
            valid_preds,
            valid_rate,
        )

        return results


__all__ = ["GenerationConfig", "Seq2SeqTPPredictor"]



