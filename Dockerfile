# Example Dockerfile
FROM python:3.10-slim

# Install poetry
RUN pip install poetry

# Set work directory
WORKDIR /app

# Copy only dependency files first for caching
COPY pyproject.toml poetry.lock* /app/

# Install dependencies
RUN poetry install --no-root

# Copy the rest of the application
COPY . /app

# Command to run your app (adjust as needed)
CMD ["poetry", "run", "phoskintime"]
