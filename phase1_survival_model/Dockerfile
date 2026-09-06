FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer so Docker can cache this
# step and skip reinstalling packages when only app code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app (dashboard.py, data/, .streamlit/, etc.)
COPY . .

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
