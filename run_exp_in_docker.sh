#! /bin/bash 

NV_GPU=$1 nvidia-docker run -it -v $PWD:/root/notebooks -p 8002:8888 standy/tensorflow
