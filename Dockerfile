FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the FAISS index at image build time (data/catalog.json must be present)
RUN python scripts/build_index.py

EXPOSE ${PORT:-8000}
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
