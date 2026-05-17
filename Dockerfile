FROM python:3.13-slim
#Dùng slim chứ bản thường build gần 50p chịu

WORKDIR /app
#Chọn thư mục làm việc bên trong container

COPY pyproject.toml .
#Copy dependencies để khi src đổi không phải cài lại depecdencies từ đầu

RUN pip install .
#tải thư viện cần thiết

COPY . .
# Copy full workspace

EXPOSE 8000 
#Khai báo cổng

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
#Lênhh chạy khi chạy docker

#Build docker:
    #docker build -t mlops-demo .  
#Run docker:
    #docker run -p 8000:8000 mlops-demo
#Makefile Docker