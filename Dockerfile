FROM python:3.13-slim

WORKDIR /app
COPY . /app

RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install

CMD ["poetry", "run", "pytest", "tests/"]
