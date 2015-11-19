#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
__global__ void multi_softmax_forward_kernel(const Dtype *x, Dtype *y, const Dtype *label, Dtype *loss, Dtype *acc, int num, int num_class, int num_group) {
	CUDA_KERNEL_LOOP(index, num*num_group) {
		int i = index/num_group;
		int j = index%num_group;
		int N = num_group*num_class;

		Dtype mymax = x[i*N + j];
		int myargmax = 0;
		for (int k = 0; k < num_class; ++k) {
			if (mymax < x[i*N + k*num_group + j]) {
				mymax = x[i*N + k*num_group + j];
				myargmax = k;
			}
		}
		acc[i*num_group + j] = Dtype(myargmax == (int)label[i*num_group + j]);

		Dtype mywegiht = Dtype(0);
		for (int k = 0; k < num_class; ++k) {
			Dtype e = exp(x[i*N + k*num_group + j]-mymax);
			y[i*N + k*num_group +j] = e;
			mywegiht += e;
		}
		
		for (int k = 0; k < num_class; ++k) {
			y[i*N + k*num_group + j] /= mywegiht;
		}

		loss[i*num_group + j] = log(y[i*N + num_group*int(label[i*num_group + j]) + j]);
	}
}

template<typename Dtype>
__global__ void multi_softmax_backward_kernel(const Dtype *y, Dtype *dy, const Dtype *label, Dtype weight, int num, int num_class, int num_group) {
	CUDA_KERNEL_LOOP(index, num*num_group) {
		int i = index/num_group;
		int j = index%num_group;
		int N = num_group*num_class;
		for (int k = 0; k < num_class; ++k) {
			dy[i*N + k*num_group + j] = weight*(y[i*N + k*num_group + j] - Dtype(k == int(label[i*num_group + j])));
		}
	}
}


template <typename Dtype>
void MultiSoftmaxLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
	int num = bottom[0]->num();
	int num_group = bottom[0]->count()/num/num_class_;
	multi_softmax_forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(num*num_group),
      CAFFE_CUDA_NUM_THREADS>>>(bottom[0]->gpu_data(), (*top)[2]->mutable_gpu_data(), bottom[1]->gpu_data(), loss_.mutable_gpu_data(),
      acc_.mutable_gpu_data(), num, num_class_, num_group);
    caffe_gpu_asum(loss_.count(), loss_.gpu_data(), (*top)[0]->mutable_cpu_data());
    (*top)[0]->mutable_cpu_data()[0] /= num*num_group;
    caffe_gpu_asum(acc_.count(), acc_.gpu_data(), (*top)[1]->mutable_cpu_data());

    //std::cout << "acc0:" << (*top)[1]->mutable_cpu_data()[0] << std::endl;
    (*top)[1]->mutable_cpu_data()[0] /= num*num_group; 
    //std::cout << "N:" <<  num*num_group << std::endl;

   	/*std::cout << loss_.stat_data() << std::endl;
    for (int i = 0; i < loss_.num(); i++) {
    	for (int j = 0; j < loss_.channels(); j++) {
    		std::cout << loss_.cpu_data()[i*loss_.channels() + j] << " ";
    	}
    	std::cout << std::endl;
    }
    std::cout << prob_.stat_data() << std::endl;*/
}

template <typename Dtype>
void MultiSoftmaxLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
	int num = (*bottom)[0]->num();
	int num_group = (*bottom)[0]->count()/num/num_class_;
	Dtype weight = top[0]->cpu_diff()[0]/num/num_group;
	if (propagate_down[0]) {
		multi_softmax_backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(num*num_group),
	      CAFFE_CUDA_NUM_THREADS>>>(top[2]->gpu_data(), (*bottom)[0]->mutable_gpu_diff(), (*bottom)[1]->gpu_data(), weight,
	      num, num_class_, num_group);
	}
}

INSTANTIATE_CLASS(MultiSoftmaxLossLayer);


}  // namespace caffe
