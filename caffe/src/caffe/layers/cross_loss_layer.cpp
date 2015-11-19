#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CrossLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels()*bottom[0]->height()*bottom[0]->width(),
           bottom[1]->channels()*bottom[1]->height()*bottom[1]->width());
  CHECK_EQ(bottom[2]->channels()*bottom[2]->height()*bottom[2]->width(),
           bottom[1]->channels()*bottom[1]->height()*bottom[1]->width());
  diff_fg_bg_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  diff_fg_rnd_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  output_.Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void CrossLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = count/num;
  Dtype margin = this->layer_param_.cross_loss_param().margin();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_fg_bg_.mutable_cpu_data());
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[2]->cpu_data(),
      diff_fg_rnd_.mutable_cpu_data());

  const Dtype *data_fg_bg = diff_fg_bg_.cpu_data();
  const Dtype *data_fg_rnd = diff_fg_rnd_.cpu_data();
  Dtype *output = output_.mutable_cpu_data();

  Dtype loss = Dtype(0);
  for (int i = 0; i < num; i++) {
    Dtype acc = Dtype(0);
    acc += caffe_cpu_dot(dim, data_fg_bg + i*dim, data_fg_bg + i*dim);
    acc -= caffe_cpu_dot(dim, data_fg_rnd + i*dim, data_fg_rnd + i*dim);
    acc = acc / dim + margin;
    output[i] = acc;
    loss += acc > Dtype(0) ? acc : Dtype(0);
  }

  loss = loss / bottom[0]->num() / Dtype(2);
  (*top)[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CrossLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  int count = (*bottom)[0]->count();
  int num = (*bottom)[0]->num();
  int dim = count/num;
  Dtype weight = top[0]->cpu_diff()[0] / count;
  Dtype *output = output_.mutable_cpu_data();
  if ( propagate_down[0] ) {
    Dtype *bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    caffe_sub(
      count,
      diff_fg_bg_.cpu_data(),
      diff_fg_rnd_.cpu_data(),
      bottom_diff
      );
    for (int i = 0; i < num; i++) {
      caffe_scal(dim, (output[i] > 0)*weight, bottom_diff + i*dim);
    }
  }
  if ( propagate_down[1] ) {
    Dtype *bottom_diff = (*bottom)[1]->mutable_cpu_diff();
    const Dtype *data_fg_bg = diff_fg_bg_.cpu_data();
    for (int i = 0; i < num; i++) {
      caffe_cpu_scale(dim, -(output[i] > 0)*weight, data_fg_bg + i*dim, bottom_diff + i*dim);
    }
  }
  if ( propagate_down[2] ) {
    Dtype *bottom_diff = (*bottom)[2]->mutable_cpu_diff();
    const Dtype *data_fg_rnd = diff_fg_rnd_.cpu_data();
    for (int i = 0; i < num; i++) {
      caffe_cpu_scale(dim, (output[i] > 0)*weight, data_fg_rnd + i*dim, bottom_diff + i*dim);
    }
  }

}

#ifdef CPU_ONLY
STUB_GPU(CrossLossLayer);
#endif

INSTANTIATE_CLASS(CrossLossLayer);

}  // namespace caffe
