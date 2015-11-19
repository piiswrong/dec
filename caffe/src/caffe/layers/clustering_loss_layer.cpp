#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ClusteringLossLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->num()%TILE_DIM, 0) << "Only support" 
    "batch sizes that are multiples of " << TILE_DIM << ".";
  N_ = this->layer_param_.clustering_loss_param().num_center();
  lambda_ = this->layer_param_.clustering_loss_param().lambda();
  CHECK_EQ(N_%TILE_DIM, 0) << "Only support" 
    "center numbers that are multiples of " << TILE_DIM << ".";
  K_ = bottom[0]->count() / bottom[0]->num();
  CHECK_EQ(K_%TILE_DIM, 0) << "Only support" 
    "input dimensions that are multiples of " << TILE_DIM << ".";
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(2);
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, K_, N_));
    this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, N_));

    coef_margin_.Reshape(1,1,1,N_);

    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.clustering_loss_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());

    FillerParameter filler_param;
    filler_param.set_value(this->layer_param_.clustering_loss_param().margin());
    ConstantFiller<Dtype> margin_filler(filler_param);
    margin_filler.Fill(this->blobs_[1].get());
  }  // parameter initialization

  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void ClusteringLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  (*top)[1]->Reshape(1, 1, 1, 1);
  if (this->layer_param_.loss_weight_size() == 1)
    this->layer_param_.add_loss_weight(Dtype(0));
  (*top)[2]->Reshape(bottom[0]->num(), 1, 1, N_);
  if (this->layer_param_.loss_weight_size() == 2)
    this->layer_param_.add_loss_weight(Dtype(0));
  (*top)[3]->Reshape(bottom[0]->num(), 1, 1, N_);
  if (this->layer_param_.loss_weight_size() == 3)
    this->layer_param_.add_loss_weight(Dtype(0));

  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[0]->count() / bottom[0]->num(), K_) << "Input size "
    "incompatible with clustering loss parameters.";

  distance_.Reshape(bottom[0]->num(), 1, 1, N_);
  mask_.Reshape(bottom[0]->num(), 1, 1, N_);
  coefm_.Reshape(bottom[0]->num(), 1, 1, 1);
  coefn_.Reshape(N_, 1, 1, 1);
  count_.Reshape(1, 1, 1, N_);
  pos_count_.Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void ClusteringLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  LOG(ERROR) << "Forward_cpu not implemented.";
}

template <typename Dtype>
void ClusteringLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  LOG(ERROR) << "Backward_cpu not implemented.";
}

#ifdef CPU_ONLY
STUB_GPU(ClusteringLossLayer);
#endif

INSTANTIATE_CLASS(ClusteringLossLayer);

}  // namespace caffe
