FROM bfolkens/docker-opencv:2.4.12-cuda7.0-cudnn4

# Install some dep packages

ENV CAFFE_PACKAGES libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler gfortran libjpeg62 libfreeimage-dev python-dev \
  python-pip python-scipy python-matplotlib python-scikits-learn ipython python-h5py python-leveldb python-networkx python-nose python-pandas \
  python-dateutil python-protobuf python-yaml python-gflags python-skimage python-sympy cython \
  libgoogle-glog-dev libbz2-dev libxml2-dev libxslt-dev libffi-dev libssl-dev libgflags-dev liblmdb-dev libboost1.54-all-dev libatlas-base-dev

RUN apt-get update && \
    apt-get install -y software-properties-common python-software-properties git wget build-essential pkg-config bc unzip cmake && \
    add-apt-repository ppa:boost-latest/ppa && \
    apt-get install -y $CAFFE_PACKAGES && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install -U leveldb  # fix GH Issue #7

# Copy the source files over and build the project

COPY . /usr/local/src/dec
WORKDIR /usr/local/src/dec

RUN cd /usr/local/src/dec/caffe && \
    cp Makefile.config.example Makefile.config && \
    make -j"$(nproc)" all

