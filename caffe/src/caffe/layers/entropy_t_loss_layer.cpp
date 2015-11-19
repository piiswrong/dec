#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EntropyTLossLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::LayerSetUp(bottom, top);
  N_ = this->layer_param_.multi_t_loss_param().num_center();
  lambda_ = this->layer_param_.multi_t_loss_param().lambda();
  K_ = bottom[0]->count() / bottom[0]->num();
  LOG(INFO) << N_ << " " << K_ << " " << bottom[0]->count();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(3);
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, K_, N_));
    this->blobs_[1].reset(new Blob<Dtype>(1, 1, K_, N_));
    this->blobs_[2].reset(new Blob<Dtype>(1, 1, 1, N_));

    //shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
    //    this->layer_param_.t_loss_param().weight_filler()));
    //weight_filler->Fill(this->blobs_[0].get());
    Dtype *cpu_weight = this->blobs_[0]->mutable_cpu_data();
    for (int i = 0; i < K_; i++) 
      for (int j = 0; j < N_; j++) 
        cpu_weight[i*N_ + j] = Dtype(i==j);

    Dtype *cpu_inv_sigma2 = this->blobs_[1]->mutable_cpu_data();
    Dtype *cpu_pi = this->blobs_[2]->mutable_cpu_data();
    for (int i = 0; i < N_; i++) {
      for (int j = 0; j < K_; j++)
        cpu_inv_sigma2[j*N_+i] = Dtype(1);
      cpu_pi[i] = Dtype(1.0/N_);
    }
  }  // parameter initialization

  count_.Reshape(N_, 1, 1, 1);
  mean_.Reshape(K_, 1, 1, 1);
  sigma_prod_.Reshape(N_, 1, 1, 1);

  this->param_propagate_down_.resize(this->blobs_.size(), false);
  this->param_propagate_down_[0] = true;
}

template <typename Dtype>
void EntropyTLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  (*top)[0]->Reshape(1, 1, 1, 1);
  if (this->layer_param_.loss_weight_size() == 0)
    this->layer_param_.add_loss_weight(Dtype(1));
  (*top)[1]->Reshape(1, 1, 1, 1);
  if (this->layer_param_.loss_weight_size() == 1)
    this->layer_param_.add_loss_weight(Dtype(0));
  (*top)[2]->Reshape(bottom[0]->num(), 1, 1, 1);
  if (this->layer_param_.loss_weight_size() == 2)
    this->layer_param_.add_loss_weight(Dtype(0));
  (*top)[3]->Reshape(bottom[0]->num(), N_, 1, 1);
  if (this->layer_param_.loss_weight_size() == 3)
    this->layer_param_.add_loss_weight(Dtype(0));

  CHECK_EQ(bottom[0]->count() / bottom[0]->num(), K_) << "Input size "
    "incompatible with clustering loss parameters.";
  CHECK_EQ(bottom[1]->num(), bottom[0]->num());
  CHECK_EQ(bottom[1]->count()/bottom[1]->num(), N_);

  distance_.Reshape(bottom[0]->num(), N_, 1, 1);
  mask_.Reshape(bottom[0]->num(), N_, 1, 1);
  coefm_.Reshape(bottom[0]->num(), 1, 1, 1);
  coefn_.Reshape(N_, 1, 1, 1);
  diff_.Reshape(bottom[0]->num(), K_, 1, 1);
  mu_sigma_.Reshape(1, 1, K_, N_);
}

template <typename Dtype>
void EntropyTLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  LOG(ERROR) << "Forward_cpu not implemented.";
}

template <typename Dtype>
void EntropyTLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  LOG(ERROR) << "Backward_cpu not implemented.";
}

#ifdef CPU_ONLY
STUB_GPU(EntropyTLossLayer);
#endif

INSTANTIATE_CLASS(EntropyTLossLayer);

}  // namespace caffe
