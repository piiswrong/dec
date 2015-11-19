#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ResizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
}

template <typename Dtype>
void ResizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  top_width_ = this->layer_param_.resize_param().top_width();
  top_height_ = this->layer_param_.resize_param().top_height();
  CHECK_EQ(top_height_%bottom[0]->height(), 0) << "incompatible output height=" << top_height_ << " and input height=" << bottom[0]->height();
  CHECK_EQ(top_width_%bottom[0]->width(), 0) << "incompatible output width=" << top_width_ << " and input width=" << bottom[0]->width();
  factor_width_ = top_width_/bottom[0]->width();
  factor_height_ = top_height_/bottom[0]->height();
  (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), top_height_, top_width_);
}

template <typename Dtype>
void ResizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  
  int hend = (*top)[0]->height();
  int wend = (*top)[0]->width();
  int bwend = bottom[0]->width();
  int bhend = bottom[0]->height();
  for (int n = 0; n < (*top)[0]->num()*(*top)[0]->channels(); n++) {
    for (int h = 0; h < hend; h++) {
      for (int w = 0; w < wend; w++) {
        top_data[w+h*wend] = bottom_data[w/factor_width_+h/factor_height_*bwend];
      }
    }
    top_data += wend*hend;
    bottom_data += bwend*bhend;
  }
}

template <typename Dtype>
void ResizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();

  caffe_set((*bottom)[0]->count(), Dtype(0), bottom_diff);

  int hend = top[0]->height();
  int wend = top[0]->width();
  int bwend = (*bottom)[0]->width();
  int bhend = (*bottom)[0]->height();
  for (int n = 0; n < top[0]->num()*top[0]->channels(); n++) {
    for (int h = 0; h < hend; h++) {
      for (int w = 0; w < wend; w++) {
        bottom_diff[w/factor_width_+h/factor_height_*bwend] += top_diff[w+h*wend];
      }
    }
    top_diff += wend*hend;
    bottom_diff += bwend*bhend;
  }
}


#ifdef CPU_ONLY
STUB_GPU(ResizeLayer);
#endif

INSTANTIATE_CLASS(ResizeLayer);


}  // namespace caffe
