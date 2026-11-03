FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && python -m spacy download en_core_web_trf
COPY --chown=app:app . .
USER app
EXPOSE 8501
# Inside a container the app must listen on 0.0.0.0; `make docker` publishes it on 127.0.0.1 only.
# No keychain in containers: the vault falls back to a 0600 key file inside the mounted data volume.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
