#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SpringLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels()*bottom[0]->height()*bottom[0]->width(),
           bottom[1]->channels()*bottom[1]->height()*bottom[1]->width());
  CHECK_EQ(bottom[0]->channels()*bottom[0]->height()*bottom[0]->width(),
           bottom[2]->channels()*bottom[2]->height()*bottom[2]->width());
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->num(), bottom[2]->num());
  diff_fg_bg_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  diff_fg_rnd_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void SpringLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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

  const Dtype *cpu_diff_fg_bg = diff_fg_bg_.cpu_data();
  Dtype *cpu_diff_fg_rnd = diff_fg_rnd_.mutable_cpu_data();

  Dtype loss = Dtype(0);
  Dtype pos_loss = Dtype(0);
  Dtype neg_loss = Dtype(0);
  for (int i = 0; i < num; i++) {
    Dtype dist_fg_bg = caffe_cpu_dot(dim, cpu_diff_fg_bg + i*dim, cpu_diff_fg_bg + i*dim) / dim;
    Dtype dist_fg_rnd = caffe_cpu_dot(dim, cpu_diff_fg_rnd + i*dim, cpu_diff_fg_rnd + i*dim) / dim;
    Dtype coef = margin > dist_fg_rnd;
    caffe_scal(dim, coef, cpu_diff_fg_rnd + i*dim);
    loss += dist_fg_bg + (margin - dist_fg_rnd)*coef;
    pos_loss += dist_fg_bg;
    neg_loss += (margin - dist_fg_rnd)*coef;
  }
  
  
  loss = loss / bottom[0]->num() / Dtype(2);
  (*top)[0]->mutable_cpu_data()[0] = loss;
  std::cout << "pos:" << pos_loss << " neg:" << neg_loss << std::endl;
}

template <typename Dtype>
void SpringLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  int count = (*bottom)[0]->count();
  Dtype weight = top[0]->cpu_diff()[0] / count;

  if (propagate_down[0]) {
    caffe_copy(count, diff_fg_bg_.cpu_data(), (*bottom)[0]->mutable_cpu_diff());
    caffe_cpu_axpby(count, -weight, diff_fg_rnd_.cpu_data(), weight, (*bottom)[0]->mutable_cpu_diff());
  }

  if (propagate_down[1]) {
    caffe_cpu_axpby(count, -weight, diff_fg_bg_.cpu_data(), Dtype(0), (*bottom)[1]->mutable_cpu_diff());
  }

  if (propagate_down[2]) {
    caffe_cpu_axpby(count, weight, diff_fg_rnd_.cpu_data(), Dtype(0), (*bottom)[2]->mutable_cpu_diff());
  }
  //LOG(ERROR) << (*bottom)[0]->asum_diff() << " " << (*bottom)[1]->asum_diff() << " " << (*bottom)[2]->asum_diff(); 
}

#ifdef CPU_ONLY
STUB_GPU(SpringLossLayer);
#endif

INSTANTIATE_CLASS(SpringLossLayer);

}  // namespace caffe
