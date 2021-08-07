FROM nvidia/cuda:10.2-base
ARG EXPERIMENT_PATH
CMD nvidia-smi

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

COPY requirements.txt .

RUN pip3 install -r requirements.txt --ignore-installed six

COPY ./Models/ /Models/

ENV LC_ALL=C.UTF-8 
ENV LANG=C.UTF-8
ENV EXPERIMENT_PATH=$EXPERIMENT_PATH

RUN mkdir -p $EXPERIMENT_PATH