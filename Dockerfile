FROM python:3.11-slim

# جلوگیری از cache
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# نصب ابزار لازم
RUN apt-get update && apt-get install -y build-essential

# کپی کردن فایل‌های پروژه
WORKDIR /app
COPY . /app

# نصب وابستگی‌ها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir flwr-xgboost

# فرمان پیش‌فرض
CMD ["python", "run_fl.py"]
