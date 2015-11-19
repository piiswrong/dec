#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_entropy_loss(int num, const Dtype *data, const Dtype *label, Dtype *out) {
	CUDA_KERNEL_LOOP(index, num) {
		out[index] = label[index] * log(data[index]);
	}
}

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  kernel_entropy_loss<Dtype><<<CAFFE_GET_BLOCKS(prob_.count()),
      CAFFE_CUDA_NUM_THREADS>>>(prob_.count(), prob_.gpu_data(), bottom[1]->gpu_data(), entropy_.mutable_gpu_data());
  Dtype loss;
  caffe_gpu_asum(entropy_.count(), entropy_.gpu_data(), &loss);
  (*top)[0]->mutable_cpu_data()[0] = loss / entropy_.num();
  if (top->size() == 2) {
    (*top)[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
  	LOG(FATAL) << "SoftmaxCrossEntropyLossLayer cannot propagate to label.";
  }

  if (propagate_down[0]) {
  	caffe_gpu_sub((*bottom)[0]->count(), prob_.gpu_data(), (*bottom)[1]->gpu_data(), (*bottom)[0]->mutable_gpu_diff());
  	const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal((*bottom)[0]->count(), loss_weight / (*bottom)[0]->num(), (*bottom)[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_CLASS(SoftmaxCrossEntropyLossLayer);


}  // namespace caffe
