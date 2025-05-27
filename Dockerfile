FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*


RUN python3 -m pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "main.py"]
