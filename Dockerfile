FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY space-requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r space-requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "demo_search.api:app", "--host", "0.0.0.0", "--port", "7860"]
