# Deep Embedded Clustering

This package implements the algorithm described in paper "Unsupervised Deep Embedding for Clustering Analysis". It depends on opencv, numpy, scipy and Caffe.

This implementation is intended for reproducing the results in the paper. If you only want to try the algorithm and find caffe too difficult to install, there is an easier to use experimental implementation in MXNet: https://github.com/dmlc/mxnet/blob/master/example/dec/dec.py, but note that results can be different from the paper. MXNet is a flexible deep learning library with fewer dependencies. You are welcome to try it. Installation guide can be found here: https://mxnet.readthedocs.org/en/latest/build.html. Once you install MXNet, simple go into directory examples/dec and run `python dec.py`.

## Usage
To run, please first build our custom version of Caffe included in this package following the official guide: http://caffe.berkeleyvision.org/installation.html.

Then download the data set you want to experiment on. We provide scripts for downloading the datasets used in the paper. For example, you can download MNIST by `cd mnist; ./get_data.sh`. Once the download completes, run `cd dec; python make_mnist_data.py` to prepare the data for Caffe.

After the data are ready, run `python dec.py DB` to run an experiment on it with DB. DB can be one of mnist, stl, reutersidf10k, reutersidf. We provide pretrained autoencoder weights with this package. You can use dec/pretrain.py to train your own autoencoder. Please read source for usage info.

### Docker

A Dockerfile has been provided to create a sterile development environment easily.  To build the environment, run `docker build --rm -t dec .` and then `docker run --rm -it dec bash` to shell into the running container.  Alternatively, [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) can be used to enable GPU capability.
