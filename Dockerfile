FROM python:3.5

# RUN apk --update-cache \
#     add musl \
#     linux-headers \
#     gcc \
#     g++ \
#     make \
#     gfortran \
#     openblas-dev \
#     python3 \
#     python3-dev

RUN pip3 install --upgrade pip 
WORKDIR /workspace
RUN git clone https://github.com/scikit-multilearn/scikit-multilearn.git
WORKDIR /workspace/scikit-multilearn 
COPY setup.py /workspace/scikit-multilearn/setup.py
RUN pip install -r requirements.txt
RUN pip install tensorflow==1.2.0 keras==2.0.3 numpy==1.16.4
RUN pip install -e .
ADD src/ /workspace/scikit-multilearn 
RUN apt-get update && \
    apt-get install -y openjdk-11-jre-headless && \
    apt-get clean;
RUN python meka.py
ADD example.py /workspace/scikit-multilearn/
RUN python example.py
ENTRYPOINT [ "/bin/bash" ]