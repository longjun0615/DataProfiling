# add dockerfile
# docker build . --build-arg http_proxy=http://internet.ford.com:83 --build-arg https_proxy=http://internet.ford.com:83 --tag hpcregistry.hpc.ford.com/skim78/pii-tagging:v1.0

FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
ENV UBUNTU=18.04
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

LABEL UBUNTU_version=$UBUNTU

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
		curl \
        git \
		unzip \
        wget \
        python3 \
        python3-dev \
		python3-setuptools && \
    apt-get clean

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN python3 -m pip install jupyter
RUN pip3 install jupyterlab==0.33.12

COPY . /usr/local/PIITagging

RUN pip3 install -r /usr/local/PIITagging/requirements.txt
RUN pip3 install /usr/local/PIITagging

WORKDIR /usr/local/PIITagging/app/

RUN apt-get update \
    && apt-get install -y \
        nmap \
        vim