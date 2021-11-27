运行方式

```shell
docker volume create --driver local --opt type=none --opt device=export模型的目录 --opt o=bind models
docker-compose build
docker-compose up
```

```shell
curl --verbose -d '{"model_name": "${这是一个模型名}"}' -X POST http://127.0.0.1:8888/models  # 加载模型
curl --verbose -d '{"query": "这是一句请求"}' -X POST http://127.0.0.1:8888/models/${这是一个模型名}  # 请求模型
curl --verbose -X DELETE http://127.0.0.1:8888/models/${这是一个模型名}  # 卸载模型
```
