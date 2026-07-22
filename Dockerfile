# Base Image
FROM python:3.13-slim-bookworm

# Set working directory
WORKDIR /pmg-api/

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy applicattion code
COPY ./app ./app

# Copy vectorizer
COPY ./artifacts ./artifacts

# Port
EXPOSE 8000

# Command to run
CMD ["uv", "run", "uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]

