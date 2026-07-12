FROM python:3.12-slim

WORKDIR /app

RUN pip install uv --quiet

COPY pyproject.toml .
COPY llm_lab llm_lab
COPY static static
COPY templates templates

RUN uv pip install --system -e . --quiet

EXPOSE 8123

CMD ["uvicorn", "llm_lab.main:app", "--host", "0.0.0.0", "--port", "8123"]
