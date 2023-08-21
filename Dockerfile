FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
COPY . /src
WORKDIR /src

RUN apt update && apt install -y git
# install the package
RUN pip install -r requirements.txt