FROM python:3.12-bookworm

WORKDIR /app

RUN mkdir -p /app/data
# RUN mkdir -p /app/chroma

COPY ./data/ /app/data/

COPY ./src/ /app

COPY ./pyproject.toml /code/pyproject.toml
RUN pip install /code/.

CMD ["sh", "entrypoint.sh"]