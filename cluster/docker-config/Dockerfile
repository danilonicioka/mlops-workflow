FROM ubuntu

RUN apt update && \ 
    apt install -y git \
    curl \
    python3 \
    gcc \
    acl; \
    git clone --branch v2.22.1 https://github.com/kubernetes-sigs/kubespray.git; \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py; \
    python3 get-pip.py --user; \
    /root/.local/bin/pip3 install -r kubespray/requirements-2.12.txt