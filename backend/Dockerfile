FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY data /data
COPY backend /app
ENV DATA_PATH=/data/german_credit_raw.csv
# Train inside the image so the model matches the installed library versions
RUN python -m app.ml.train
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
