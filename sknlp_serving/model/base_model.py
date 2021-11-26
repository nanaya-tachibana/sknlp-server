from __future__ import annotations
from typing import Any, Sequence

import os
import json
import jieba
import tempfile
import dataclasses
from functools import partial

from sknlp_serving.tfserving import TensorProto, TensorShapeProto, DataType
from sknlp_serving.tfserving import TensorMeta, TFServing


@dataclasses.dataclass
class InferenceResult:
    task: str
    result: list[Any] | None = None
    score: list[float] | None = None
    vec: list[float] | None = None

    def to_dict(self):
        score = self.score or list()
        result = self.result or self.vec or list()
        return {"result": result, "score": score}


class BaseModel:
    def __init__(
        self,
        task: str,
        classes: Sequence[str],
        token2idx: dict[str, int],
        segmenter: str,
        input_tensor_metas: list[TensorMeta],
        output_tensor_metas: list[TensorMeta],
        inference_kwargs: dict[str, Any] | None = None,
        custom_kwrgs: dict[str, Any] | None = None,
        max_sequence_length: int | None = None,
        output_logits: bool = True,
        **kwargs,
    ) -> None:
        self.task = task
        self.classes = classes
        self.token2idx = token2idx
        self.max_sequence_length = max_sequence_length
        self.output_logits = output_logits
        self.input_tensor_metas = input_tensor_metas
        self.output_tensor_metas = output_tensor_metas

        if segmenter in ("jieba", "word", "token"):
            with tempfile.NamedTemporaryFile("w") as f:
                for token in token2idx:
                    f.write(f"{token} 1\n")
                f.flush()
                tokenizer = jieba.Tokenizer(dictionary=f.name)
                tokenizer.initialize()
                self.tokenizer = partial(tokenizer.lcut, HMM=False)
        elif segmenter == "char":
            self.tokenizer = list
        else:
            self.tokenizer = None

    def parse_output_tensor(
        self, query: str | list[str], outputs: dict[str, TensorProto]
    ) -> InferenceResult:
        raise NotImplementedError()

    def create_input_tensor(self, query: str | list[str]) -> dict[str, TensorProto]:
        if isinstance(query, str):
            query = [query]
        input_tensor: dict[str, TensorProto] = dict()
        for q, tensor_meta in zip(query, self.input_tensor_metas):
            input_tensor[tensor_meta.name] = self.text2tensor(q, tensor_meta)
        return input_tensor

    def text2ids(self, text: str) -> list[int]:
        tokens = self.tokenizer(text)[: self.max_sequence_length]
        return [self.token2idx.get(token, 1) for token in tokens]

    def text2tensor(self, text: str, tensor_meta: TensorMeta) -> TensorProto:
        dtype = tensor_meta.dtype
        shape = [TensorShapeProto.Dim(size=1)]
        if dtype == DataType.DT_STRING:
            values = [text.encode("UTF-8")]
            if len(tensor_meta.shape) > 1:
                shape.append(TensorShapeProto.Dim(size=1))
        else:
            values = self.text2ids(text)
            values.append(0)  # 额外添加一个0，保证server batching在尾部padding的是0
            shape.append(TensorShapeProto.Dim(size=len(values)))

        kwargs = {
            "dtype": dtype,
            "tensor_shape": TensorShapeProto(dim=shape),
        }
        if dtype == DataType.DT_STRING:
            key = "string_val"
        elif dtype in (
            DataType.DT_INT32,
            DataType.DT_INT16,
            DataType.DT_INT8,
            DataType.DT_UINT8,
        ):
            key = "int_val"
        elif dtype == DataType.DT_INT64:
            key = "int64_val"
        elif dtype == DataType.DT_UINT32:
            key = "uint32_val"
        elif dtype == DataType.DT_UNIT64:
            key = "uint64_val"
        elif dtype == DataType.DT_BOOL:
            key = "bool_val"
        elif dtype == DataType.DT_FLOAT:
            key = "float_val"
        elif dtype == DataType.DT_DOUBLE:
            key = "double_val"
        else:
            raise ValueError(f"不支持的张量类型: {DataType(dtype)}.")
        kwargs[key] = values
        return TensorProto(**kwargs)

    @classmethod
    def load_model(cls, model_name: str, base_path: str, tfs: TFServing) -> "BaseModel":
        model_base_directory = os.path.join(base_path, model_name, "0")
        meta_filename = os.path.join(model_base_directory, "meta.json")
        with open(meta_filename) as f:
            meta = json.loads(f.read())
        vocab_filename = os.path.join(model_base_directory, "vocab.json")
        with open(vocab_filename) as f:
            vocab = json.loads(f.read())
        token2idx = vocab["token2idx"]
        meta.update(token2idx=token2idx)
        inference_kwargs = meta["inference_kwargs"]
        input_names = None
        output_names = None
        if "input_names" in inference_kwargs:
            input_names = inference_kwargs["input_names"]
        if "output_names" in inference_kwargs:
            output_names = inference_kwargs["output_names"]
        input_tensor_metas, output_tensor_metas = tfs.get_model_metadata(
            model_name, input_names=input_names, output_names=output_names
        )
        meta.update(
            input_tensor_metas=input_tensor_metas,
            output_tensor_metas=output_tensor_metas,
        )
        return cls(**meta)
