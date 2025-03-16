FROM python:3.12-slim-bullseye AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ python3-dev libopencv-dev git && apt-get clean
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim-bullseye
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY . .
CMD ["python", "main.py"]