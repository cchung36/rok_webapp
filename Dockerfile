FROM "nvidia/cuda:11.6.0-devel-ubuntu20.04"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && yes | apt-get upgrade
RUN apt-get install -y wget software-properties-common python3-pip build-essential npm curl

RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get update && yes | apt-get upgrade && \
    apt-get install -y nodejs

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" && \
    apt-get update && \
    apt-get install libcudnn8=8.4.1.*-1+cuda11.6 && \
    apt-get install libcudnn8-dev=8.4.1.*-1+cuda11.6

RUN wget http://nginx.org/keys/nginx_signing.key && \
    apt-key add nginx_signing.key && \
    echo "deb http://nginx.org/packages/ubuntu focal nginx deb-src" >> /etc/apt/source.list && \
    echo "deb-src http://nginx.org/packages/ubuntu focal nginx" >> /etc/apt/source.list && \
    apt-get update && \
    apt-get install -y nginx

WORKDIR /src

RUN pip3 install --upgrade pip
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

RUN npm install -g npm@latest
RUN npm install -g @vue/cli
RUN npm install --save vue-router
RUN npm install bootstrap-vue --save
RUN npm install axios --save
