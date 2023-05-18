FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U scikit-learn
COPY . .
CMD ["python", "app.py"]
