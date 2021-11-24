from __future__ import annotations
from typing import List, Union
import os
import json
import asyncio

import grpc
from pydantic import BaseModel
import pydantic
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from sknlp_serving.tfserving import TFServing
from sknlp_serving.model import ClassificationModel, InferenceResult, TaggingModel


TF_SERVING_ADDRESS = os.environ.get("TF_SERVING_ADDRESS", "127.0.0.1:8500")
TF_SERVING_MODEL_BASE_PATH = os.environ.get("TF_SERVING_MODEL_BASE_PATH", "/models")
MODEL_BASE_PATH = os.environ.get("MODEL_BASE_PATH", TF_SERVING_MODEL_BASE_PATH)

lock = asyncio.Lock()
model_handler = dict()
tfserving = TFServing(
    grpc.insecure_channel(TF_SERVING_ADDRESS), TF_SERVING_MODEL_BASE_PATH
)


class InferenceRequest(BaseModel):
    query: Union[str, List[str]]


class LoadingRequest(BaseModel):
    model_name: str


async def load(request):
    try:
        json_data = json.loads(await request.body())
        model_name = LoadingRequest(**json_data).model_name
        model_path = os.path.join(MODEL_BASE_PATH, model_name, "0")
        with open(os.path.join(model_path, "meta.json")) as f:
            task = json.loads(f.read()).get("task", None)
    except FileNotFoundError as exc:
        return JSONResponse({"error": exc.__str__()}, status_code=400)
    except pydantic.ValidationError as exc:
        return JSONResponse({"error": exc.json()}, status_code=400)
    except json.decoder.JSONDecodeError as exc:
        return JSONResponse({"error": exc.msg}, status_code=400)

    if task == "classification":
        model_class = ClassificationModel
    elif task == "tagging":
        model_class = TaggingModel
    else:
        raise ValueError(f"不支持的模型类型: {task}")

    async with lock:
        if model_name in model_handler:
            return PlainTextResponse()
        model_names = list(model_handler.keys())
        model_names.append(model_name)
        try:
            tfserving.reload_config(model_names)
        except grpc.RpcError as exc:
            return JSONResponse({"error": exc.details()}, status_code=500)

        model = model_class.load_model(model_name, MODEL_BASE_PATH, tfserving)

        def inference(query: str | list[str]) -> InferenceResult:
            return model.parse_output_tensor(
                query, tfserving.predict(model_name, model.create_input_tensor(query))
            )

        model_handler[model_name] = inference
    return PlainTextResponse()


async def unload(request):
    model_name = request.path_params["model_name"]
    async with lock:
        if model_name not in model_handler:
            return JSONResponse({"error": "模型未加载"}, status_code=404)

        model_names = [name for name in model_handler.keys() if name != model_name]
        try:
            tfserving.reload_config(model_names)
        except grpc.RpcError as exc:
            return JSONResponse({"error": exc.details()}, status_code=500)
        model_handler.pop(model_name)
    return PlainTextResponse()


async def inference(request):
    try:
        json_data = json.loads(await request.body())
        query = InferenceRequest(**json_data).query
    except pydantic.ValidationError as exc:
        return JSONResponse({"error": exc.json()}, status_code=400)
    except json.decoder.JSONDecodeError as exc:
        return JSONResponse({"error": exc.msg}, status_code=400)
    except grpc.RpcError as exc:
        return JSONResponse({"error": exc.details()}, status_code=500)

    model_name = request.path_params["model_name"]
    if model_name not in model_handler:
        return JSONResponse({"error": "模型未加载"}, status_code=404)

    try:
        res = model_handler[model_name](query)
    except grpc.RpcError as exc:
        return JSONResponse({"error": exc.details()}, status_code=500)
    return JSONResponse(res.to_dict())


async def server_error(request, exc):
    return JSONResponse({"error": exc.__str__()}, status_code=500)


routes = [
    Route("/models", load, methods=["POST"]),
    Route("/models/{model_name:str}", inference, methods=["POST"]),
    Route("/models/{model_name:str}", unload, methods=["DELETE"]),
]
app = Starlette(routes=routes, exception_handlers={500: server_error})
