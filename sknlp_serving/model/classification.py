from __future__ import annotations
from typing import Sequence, Any

from scipy.special import expit, softmax
import numpy as np

from sknlp_serving.model.base_model import (
    BaseModel,
    TensorMeta,
    TensorProto,
    InferenceResult,
)


class ClassificationModel(BaseModel):
    def __init__(
        self,
        task: str,
        classes: Sequence[str],
        token2idx: dict[str, int],
        segmenter: str,
        input_tensor_metas: list[TensorMeta],
        output_tensor_metas: list[TensorMeta],
        is_multilabel: bool = False,
        inference_kwargs: dict[str, Any] | None = None,
        custom_kwrgs: dict[str, Any] | None = None,
        max_sequence_length: int | None = None,
        output_logits: bool = True,
        **kwargs
    ) -> None:
        self.is_multilabel = is_multilabel
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

    def parse_output_tensor(
        self, query: str | list[str], outputs: dict[str, TensorProto]
    ) -> InferenceResult:
        labels: list[str] = []
        scores: list[float] = []
        output_tensor_meta = self.output_tensor_metas[0]
        logits = outputs[output_tensor_meta.name].float_val
        if self.is_multilabel:
            if self.output_logits:
                probs: np.ndarray = expit(logits)
            else:
                probs: np.ndarray = logits
            for i, prob in enumerate(probs.tolist()):
                if i == 0:
                    continue
                label = self.classes[i]
                threshold = self.thresholds.get(label, 0.5)
                if prob > threshold:
                    labels.append(label)
                    scores.append(prob)
        else:
            if self.output_logits:
                probs = softmax(logits)
            else:
                probs = logits
            argmax = probs.argmax()
            label = self.classes[argmax]
            threshold = self.thresholds.get(label, 0.5)
            if argmax != 0 and probs[argmax] > threshold:
                labels.append(label)
                scores.append(probs[argmax])
        return InferenceResult(self.task, labels, scores)
