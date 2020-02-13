FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN pip install "poetry>=0.12,<0.13"
RUN poetry config settings.virtualenvs.create false

WORKDIR /workspace/
COPY pyproject.toml .
COPY poetry.lock .
RUN poetry install
