FROM python:3.8

RUN apt-get update
RUN apt-get install \
    'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python3", "api.py"]
