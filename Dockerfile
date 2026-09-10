FROM python:3.14-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

RUN pip install uv

COPY src ./src

COPY data ./data

RUN uv sync --frozen

ENV PYTHONPATH=/app/src

EXPOSE 8050

CMD ["uv", "run", "uvicorn", "material_mlops.api.main:app", "--host", "0.0.0.0", "--port", "8050"]