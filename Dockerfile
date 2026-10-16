FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1
WORKDIR /srv

COPY requirements.txt ./
RUN grep -v '^pytest' requirements.txt > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

COPY app ./app
COPY eval ./eval
COPY start.sh ./start.sh

CMD ["sh", "start.sh"]
