FROM python:3.11.8-slim

WORKDIR /senior_project

COPY requirements.txt/

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "predict_server.py"]