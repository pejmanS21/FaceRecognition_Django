FROM python:3

MAINTAINER pejmanS21

# os level reqs
RUN apt-get update -y \
     && apt install libgl1-mesa-glx -y \
     && apt-get install 'ffmpeg' 'libsm6' 'libxext6' -y \
     && apt-get install -y build-essential libzbar-dev

RUN pip install install --upgrade pip

# install large requirements to save time in next builds
RUN pip install --no-cache-dir torch \ 
     && pip install --no-cache-dir numpy \
     && pip install --no-cache-dir opencv-python \
     && pip install --no-cache-dir torchvision \
     && pip install --no-cache-dir matplotlib \
     && pip install --no-cache-dir deep_utils \
     && pip install --no-cache-dir Django \
     && pip install --no-cache-dir facenet-pytorch \
     && pip install --no-cache-dir termcolor


COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 8000