#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MultiSoftmaxLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top); 
  num_class_ = this->layer_param_.multi_softmax_loss_param().class_per_group();
}

template <typename Dtype>
void MultiSoftmaxLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count()/bottom[0]->num()%num_class_, 0)
  	<< "Input dimension must be multiple of num_class_!";
  CHECK_EQ(bottom[0]->count()/num_class_, bottom[1]->count())
    << "Inconsistent input and label dimensions!";
  acc_.Reshape(bottom[0]->num(), bottom[0]->count()/bottom[0]->num()/num_class_, 1, 1);
  loss_.Reshape(bottom[0]->num(), bottom[0]->count()/bottom[0]->num()/num_class_, 1, 1);

  if (this->layer_param_.loss_weight_size() == 1)
    this->layer_param_.add_loss_weight(Dtype(0));
  (*top)[1]->Reshape(1, 1, 1, 1);
  if (this->layer_param_.loss_weight_size() == 2)
    this->layer_param_.add_loss_weight(Dtype(0));
  (*top)[2]->Reshape(bottom[0]->num(), num_class_, 1, bottom[0]->count()/bottom[0]->num()/num_class_);
}

template <typename Dtype>
void MultiSoftmaxLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  LOG(ERROR) << "Forward_cpu not implemented for MultiSoftmaxLossLayer";
}

template <typename Dtype>
void MultiSoftmaxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  LOG(ERROR) << "Backward_cpu not implemented for MultiSoftmaxLossLayer";
}


#ifdef CPU_ONLY
STUB_GPU(MultiSoftmaxLossLayer);
#endif

INSTANTIATE_CLASS(MultiSoftmaxLossLayer);


}  // namespace caffe

