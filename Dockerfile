FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

COPY streamlit_app.py .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0"]