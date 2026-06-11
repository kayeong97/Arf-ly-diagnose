# CPU 전용 FastAPI 이미지
FROM python:3.8-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=-1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_NUM_INTRAOP_THREADS=2 \
    TF_NUM_INTEROP_THREADS=2

RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    echo 'Asia/Seoul' > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget vim \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    pillow \
    numpy \
    tensorflow-cpu==2.10.1

WORKDIR /workspace/afly

CMD ["python", "-m", "uvicorn", "alfy_api:app", "--host", "0.0.0.0", "--port", "8080"]

