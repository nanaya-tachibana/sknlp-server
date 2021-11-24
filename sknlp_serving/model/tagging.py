from __future__ import annotations
from typing import Sequence, Any
from itertools import accumulate
import math

from scipy.special import expit
import numpy as np

from sknlp_serving.model.base_model import (
    BaseModel,
    TensorMeta,
    TensorProto,
    InferenceResult,
)


class TaggingModel(BaseModel):
    def __init__(
        self,
        task: str,
        classes: Sequence[str],
        token2idx: dict[str, int],
        segmenter: str,
        input_tensor_metas: list[TensorMeta],
        output_tensor_metas: list[TensorMeta],
        output_format: str = "global_pointer",
        add_start_end_tag: bool = True,
        inference_kwargs: dict[str, Any] | None = None,
        custom_kwrgs: dict[str, Any] | None = None,
        max_sequence_length: int | None = None,
        output_logits: bool = True,
        **kwargs
    ) -> None:
        self.output_format = output_format
        self.add_start_end_tag = add_start_end_tag
        self.thresholds = dict()
        if inference_kwargs is not None and "thresholds" in inference_kwargs:
            self.thresholds = inference_kwargs["thresholds"]
        super().__init__(
            task,
            classes,
            token2idx,
            segmenter,
            input_tensor_metas,
            output_tensor_metas,
            inference_kwargs=inference_kwargs,
            custom_kwrgs=custom_kwrgs,
            max_sequence_length=max_sequence_length,
            output_logits=output_logits,
            **kwargs
        )

    def _token_lengths(
        self,
        text: str,
        byte_start_ids: list[int] | None = None,
        byte_end_ids: list[int] | None = None,
    ) -> list[int]:
        if self.tokenizer is not None:
            return [
                len(token) for token in self.tokenizer(text)[: self.max_sequence_length]
            ]
        utf_bytes = text.encode("UTF-8")
        return [
            len(utf_bytes[sid:eid].decode("UTF-8"))
            for sid, eid in zip(byte_start_ids, byte_end_ids)
        ]

    def parse_bio_output(
        self,
        tag_ids: list[int],
        start_mapping: dict[int, int],
        end_mapping: dict[int, int],
    ) -> tuple[list[tuple[int, int, str]], list[float]]:
        if self.add_start_end_tag:
            tag_ids = tag_ids[1:-1]
        num_tag_ids = len(tag_ids)
        current_begin_tag = -1
        begin = 0
        parsed_tags: list[tuple[int, int, str]] = list()
        for i, tag_id in enumerate(tag_ids):
            if (
                i < num_tag_ids - 1
                and tag_id != 0
                and tag_id % 2 == 0
                and tag_id - 1 == current_begin_tag
            ):
                continue

            if i != begin:
                parsed_tags.append(
                    (
                        start_mapping[begin],
                        end_mapping[i - (i < num_tag_ids - 1)],
                        self.classes[(current_begin_tag + 1) // 2],
                    )
                )

            if tag_id % 2 == 1:
                begin = i
                current_begin_tag = tag_id
            else:
                begin = i + 1
                current_begin_tag = -1
        return parsed_tags, []

    def parse_pointer_output(
        self,
        pointer: list[float],
        start_mapping: dict[int, int],
        end_mapping: dict[int, int],
    ) -> tuple[list[tuple[int, int, str]], list[float]]:
        num_classes = len(self.classes)
        square_length = len(pointer) // num_classes
        length = int(math.sqrt(square_length))
        pointer_array: np.ndarray = np.reshape(pointer, (num_classes, length, length))
        parsed_tags: list[tuple[int, int, str]] = []
        scores: list[float] = []
        for i, logits_matrix in enumerate(pointer_array):
            label = self.classes[i]
            threshold = self.thresholds.get(label, 0.5)
            score_matrix = expit(logits_matrix)
            for start, end in zip(*np.where(score_matrix >= threshold)):
                score = score_matrix[start, end]
                start -= self.add_start_end_tag
                end -= self.add_start_end_tag
                parsed_tags.append(
                    (start_mapping[int(start)], end_mapping[int(end)], label)
                )
                scores.append(score)
        return parsed_tags, scores

    def parse_output_tensor(
        self, query: str | list[str], outputs: dict[str, TensorProto]
    ) -> InferenceResult:
        if isinstance(query, list):
            query = query[-1]

        if len(self.output_tensor_metas) == 1:
            token_lengths = self._token_lengths(query)
        elif len(self.output_tensor_metas) == 3:
            start_ids = outputs[self.output_tensor_metas[1].name].int64_val
            end_ids = outputs[self.output_tensor_metas[2].name].int64_val
            token_lengths = self._token_lengths(
                query, byte_start_ids=start_ids, byte_end_ids=end_ids
            )
        else:
            raise ValueError("ERROR")
        cumsum = list(accumulate(token_lengths))
        start_mapping = {
            i: c - l for i, (c, l) in enumerate(zip(cumsum, token_lengths))
        }
        end_mapping = {i: c - 1 for i, c in enumerate(cumsum)}

        if self.output_format == "bio":
            tag_ids = outputs[self.output_tensor_metas[0].name].int_val
            labels, scores = self.parse_bio_output(tag_ids, start_mapping, end_mapping)
        elif self.output_format == "global_pointer":
            pointer = outputs[self.output_tensor_metas[0].name].float_val
            labels, scores = self.parse_pointer_output(
                pointer, start_mapping, end_mapping
            )
        return InferenceResult(self.task, labels, scores)
