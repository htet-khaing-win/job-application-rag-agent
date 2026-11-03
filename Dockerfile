FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN useradd --create-home --uid 10001 app
WORKDIR /app
# requirements.lock is optional (see scripts/lock.sh). When present, installs are hash-verified.
COPY requirements.txt requirements.loc[k] ./
RUN if [ -f requirements.lock ]; then pip install --require-hashes -r requirements.lock; \
    else pip install -r requirements.txt; fi \
    && python -m spacy download en_core_web_trf
COPY --chown=app:app . .
USER app
EXPOSE 8501
# Inside a container the app must listen on 0.0.0.0; `make docker` publishes it on 127.0.0.1 only.
# No keychain in containers: the vault falls back to a 0600 key file inside the mounted data volume
# (consider enabling the vault passphrase in the Security tab for extra protection).
# NOTE: the spaCy model download above is not hash-verified; vendor the wheel if that matters to you.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
