FROM bfolkens/docker-caffe:master-cuda7.0-cudnn4

RUN apt-get update && \
    apt-get install -y python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose python-h5py s3cmd bc wget git build-essential unzip

# Install OpenCV

ENV OPENCV_VERSION 2.4.12

RUN cd /usr/local/src && \
    wget -O opencv-$OPENCV_VERSION.zip https://github.com/Itseez/opencv/archive/$OPENCV_VERSION.zip && \
    unzip opencv-$OPENCV_VERSION.zip && \
    cd opencv-$OPENCV_VERSION && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr \
          -D BUILD_EXAMPLES=OFF \
          -D CUDA_GENERATION=Auto \
          -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_VTK=ON -D WITH_OPENGL=OFF -D WITH_QT=OFF .. && \
    make && make install && \
    rm -rf /usr/local/src/opencv*

