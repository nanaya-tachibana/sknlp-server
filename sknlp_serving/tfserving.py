from __future__ import annotations
import dataclasses
import enum

import grpc
from tensorflow_serving.apis.model_pb2 import ModelSpec
from tensorflow_serving.apis.model_service_pb2_grpc import ModelServiceStub
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub
from tensorflow_serving.apis.model_management_pb2 import ReloadConfigRequest
from tensorflow_serving.apis.get_model_status_pb2 import GetModelStatusRequest
from tensorflow_serving.apis.predict_pb2 import PredictRequest
from tensorflow_serving.apis.get_model_metadata_pb2 import (
    GetModelMetadataRequest,
    SignatureDefMap,
)
from tensorflow_serving.config.model_server_config_pb2 import (
    ModelServerConfig,
    ModelConfigList,
    ModelConfig,
)
from tensorflow.core.framework.types_pb2 import DataType
from tensorflow.core.protobuf.meta_graph_pb2 import TensorInfo
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto


class ModelStatus(enum.Enum):
    # Default value.
    UNKNOWN = 0
    # The manager is tracking this servable, but has not initiated any action
    # pertaining to it.
    START = 10
    # The manager has decided to load this servable. In particular, checks
    # around resource availability and other aspects have passed, and the
    # manager is about to invoke the loader's Load() method.
    LOADING = 20
    # The manager has successfully loaded this servable and made it available
    # for serving (i.e. GetServableHandle(id) will succeed). To avoid races,
    # this state is not reported until *after* the servable is made
    # available.
    AVAILABLE = 30
    # The manager has decided to make this servable unavailable, and unload # it. To avoid races, this state is reported *before* the servable is
    # made unavailable.
    UNLOADING = 40
    # This servable has reached the end of its journey in the manager. Either
    # it loaded and ultimately unloaded successfully, or it hit an error at
    # some point in its lifecycle.
    END = 50


@dataclasses.dataclass
class TensorMeta:
    name: str
    dtype: DataType
    shape: list[int]


class TFServing:
    def __init__(
        self,
        grpc_channel: grpc.Channel,
        base_path: str,
        timeout: int = 5,
        reloading_timeout: int | None = None,
    ) -> None:
        self.base_path = base_path
        self.timeout = timeout
        self.reloading_timeout = reloading_timeout or self.timeout
        self.model_service_stub = ModelServiceStub(grpc_channel)
        self.prediction_service_stub = PredictionServiceStub(grpc_channel)

    def reload_config(self, model_names: list[str]) -> None:
        model_configs: list[ModelConfig] = []
        for model_name in model_names:
            model_configs.append(
                ModelConfig(
                    name=model_name,
                    base_path="/".join([self.base_path, model_name]),
                    model_platform="tensorflow",
                )
            )
        model_config_list = ModelConfigList(config=model_configs)
        request = ReloadConfigRequest(
            config=ModelServerConfig(model_config_list=model_config_list)
        )
        self.model_service_stub.HandleReloadConfigRequest(
            request, timeout=self.reloading_timeout
        )

    def get_model_metadata(
        self,
        model_name: str,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
    ) -> tuple[list[TensorMeta], list[TensorMeta]]:
        model_spec = ModelSpec(name=model_name, signature_name="serving_default")
        request = GetModelMetadataRequest(
            model_spec=model_spec, metadata_field=["signature_def"]
        )
        response = self.prediction_service_stub.GetModelMetadata(
            request, timeout=self.timeout
        )
        def_map = SignatureDefMap()
        response.metadata["signature_def"].Unpack(def_map)
        metadata = def_map.signature_def["serving_default"]

        if input_names is None:
            input_names = list(metadata.inputs.keys())
        if output_names is None:
            output_names = list(metadata.outputs.keys())

        def read_tensor_meta(
            names: list[str], tensor_info: dict[str, TensorInfo]
        ) -> list[TensorMeta]:
            tensor_metas: list[TensorMeta] = []
            for name in names:
                tensor = tensor_info[name]
                dims = [d.size for d in tensor.tensor_shape.dim]
                tensor_meta = TensorMeta(name, tensor.dtype, dims)
                tensor_metas.append(tensor_meta)
            return tensor_metas

        return (
            read_tensor_meta(input_names, metadata.inputs),
            read_tensor_meta(output_names, metadata.outputs),
        )

    def get_model_status(self, model_name: str) -> ModelStatus:
        model_spec = ModelSpec(name=model_name)
        request = GetModelStatusRequest(model_spec=model_spec)
        response = self.model_service_stub.GetModelStatus(request, timeout=self.timeout)
        for version_status in response.model_version_status:
            if version_status.version == 0:
                return ModelStatus(version_status.state)
        return ModelStatus(0)

    def predict(
        self, model_name: str, input_tensor: dict[str, TensorProto]
    ) -> dict[str, TensorProto]:
        model_spec = ModelSpec(name=model_name)
        request = PredictRequest(model_spec=model_spec)
        for name, tensor in input_tensor.items():
            request.inputs[name].CopyFrom(tensor)
        return dict(
            self.prediction_service_stub.Predict(request, timeout=self.timeout).outputs
        )


if __name__ == "__main__":
    s = TFServing(grpc.insecure_channel("127.0.0.1:8500"))
    s.reload_config(["xxx", "xyz"])
    print(s.get_model_metadata("xxx"))
    tensor = TensorProto(
        dtype=DataType.DT_INT64,
        tensor_shape=TensorShapeProto(
            dim=[TensorShapeProto.Dim(size=1), TensorShapeProto.Dim(size=5)]
        ),
        int64_val=[230, 2130, 4324, 222, 0],
    )
    print(s.predict("xxx", {"token_ids": tensor}))
