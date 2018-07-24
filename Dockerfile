from tensorflow/tensorflow:1.6.0-gpu-py3
RUN apt-get update && apt-get install -y vim protobuf-compiler libgtk2.0 git libboost-all-dev libgoogle-glog-dev wget python3-pip libopencv-dev python3-dev
RUN pip3 install opencv-python simplejson tqdm
RUN wget https://dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.bz2 && tar xf boost_1_67_0.tar.bz2 && cd boost_1_67_0 && ./bootstrap.sh --with-python=python3 && ./b2 && ./b2 install && rm -rf boost_1_67_0 boost_1_67_0.tar.bz2


