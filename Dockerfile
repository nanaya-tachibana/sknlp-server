FROM python:3.9-slim as build-base
# prevents python creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 
ENV PYTHONUNBUFFERED=1
# pip
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
# poetry
# https://python-poetry.org/docs/configuration/#using-environment-variables
ENV POETRY_VERSION=1.1.6
# make poetry install to this location
ENV POETRY_HOME="/opt/poetry"
# make poetry create the virtual environment in the project's root
# it gets named `.venv`
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
# do not ask any interactive question
ENV POETRY_NO_INTERACTION=1
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN echo deb http://mirrors.aliyun.com/debian/ bullseye main non-free contrib > /etc/apt/sources.list
RUN echo deb-src http://mirrors.aliyun.com/debian/ bullseye main non-free contrib >> /etc/apt/sources.list
RUN echo deb http://mirrors.aliyun.com/debian-security bullseye-security main non-free contrib >> /etc/apt/sources.list
RUN echo deb-src http://mirrors.aliyun.com/debian-security bullseye-security main non-free contrib >> /etc/apt/sources.list
RUN echo deb http://mirrors.aliyun.com/debian/ bullseye-updates main non-free contrib >> /etc/apt/sources.list
RUN echo deb-src http://mirrors.aliyun.com/debian/ bullseye-updates main non-free contrib >> /etc/apt/sources.list
RUN echo deb http://mirrors.aliyun.com/debian/ bullseye-backports main non-free contrib >> /etc/apt/sources.list
RUN echo deb-src http://mirrors.aliyun.com/debian/ bullseye-backports main non-free contrib >> /etc/apt/sources.list
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        # deps for installing poetry
        curl \
        # deps for building python deps
        build-essential \
        # timezone 
        tzdata

# install poetry - respects $POETRY_VERSION & $POETRY_HOME
RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py -o get-poetry.py
RUN python get-poetry.py


FROM build-base as builder
ENV PYSETUP_PATH="/opt/pysetup" 
ENV VENV_PATH="/opt/pysetup/.venv" 
ENV PATH="$VENV_PATH/bin:$PATH"
WORKDIR $PYSETUP_PATH
COPY sknlp_serving sknlp_serving
COPY poetry.lock pyproject.toml ./
RUN poetry install --no-dev


FROM build-base as misc-builder
RUN python -m pip install grpcio-tools==1.42.0
COPY tensorflow-proto tensorflow-proto
RUN python -m grpc_tools.protoc -I tensorflow-proto --python_out=. $(find tensorflow-proto -name '*.proto')
RUN python -m grpc_tools.protoc -I tensorflow-proto --python_out=. --grpc_python_out=. tensorflow-proto/tensorflow_serving/apis/prediction_service.proto
RUN python -m grpc_tools.protoc -I tensorflow-proto --python_out=. --grpc_python_out=. tensorflow-proto/tensorflow_serving/apis/model_service.proto


FROM python:3.9-slim
LABEL developer="nanaya"

ENV PYTHONUNBUFFERED=1
# this is where our requirements + virtual environment will live
ENV PYSETUP_PATH="/opt/pysetup" 
ENV VENV_PATH="/opt/pysetup/.venv" 
ENV PATH="$VENV_PATH/bin:$PATH"
# tensorflow serving model base path
ENV TF_SERVING_MODEL_BASE_PATH="/models"
# model base path
ENV MODEL_BASE_PATH=$TF_SERVING_MODEL_BASE_PATH
# port
ENV HTTP_PORT=8888

# copy in our built poetry + venv
COPY --from=builder $PYSETUP_PATH $PYSETUP_PATH
# change time zone
RUN cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

WORKDIR /app
# copy grpc proto
COPY --from=misc-builder tensorflow tensorflow
COPY --from=misc-builder tensorflow_serving tensorflow_serving
COPY server.py server.py
COPY start_sknlp_server.sh start_sknlp_server.sh
EXPOSE $HTTP_PORT
CMD ["./start_sknlp_server.sh"]