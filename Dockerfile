FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
#Lênhh chạy khi chạy docker

#Build docker:
    #docker build -t mlops-demo .  
#Run docker:
    #docker run -p 8000:8000 mlops-demo
#Makefile Docker