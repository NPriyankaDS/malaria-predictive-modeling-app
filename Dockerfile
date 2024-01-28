FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip3 install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "1_🏠_Home.py", "--server.port=8501", "--server.address=0.0.0.0"]