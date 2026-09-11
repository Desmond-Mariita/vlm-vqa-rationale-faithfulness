FROM python:3.12-slim
RUN useradd --create-home --uid 1000 researcher
WORKDIR /workspace
COPY requirements.txt /tmp/release-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/release-requirements.txt
USER researcher
ENV PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
