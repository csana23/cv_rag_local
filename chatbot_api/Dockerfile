# chatbot_api/Dockerfile

FROM python:3.12-bookworm

WORKDIR /app
COPY ./src/ /app

COPY ./pyproject.toml /code/pyproject.toml
RUN pip install /code/.

EXPOSE 8005
CMD ["sh", "entrypoint.sh"]