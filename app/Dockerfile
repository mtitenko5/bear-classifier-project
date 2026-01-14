FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bear_app.py .
COPY loaded/embeddings_mini.npz .
COPY loaded/model_bear.tar .
COPY .dockerignore .
COPY bear_dataset_mini/

CMD ["streamlit", "run", "bear_app.py", "--server.port=8501", "--server.address=0.0.0.0"]