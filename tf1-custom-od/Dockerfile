FROM tensorflow/tensorflow:1.15.5-gpu
MAINTAINER kuczynskilukasz@gmail.com

RUN apt update
RUN apt upgrade -y
RUN apt install -y git wget vim

WORKDIR /root
RUN wget https://github.com/tensorflow/models/archive/r1.13.0.zip
RUN unzip r1.13.0.zip && mv models-r1.13.0 models

RUN apt install -y protobuf-compiler
WORKDIR /root/models/research
RUN protoc object_detection/protos/*.proto --python_out=.
RUN pip install .

WORKDIR /root
ENV PYTHONPATH=$PYTHONPATH:/root/models/research/slim/

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
