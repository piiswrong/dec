#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
      (*top)[0]->mutable_gpu_data());
  if (this->output_labels_) {
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        (*top)[1]->mutable_gpu_data());
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
}

template <typename Dtype>
void BaseShufflingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  int batch_size = this->layer_param_.shuffling_data_param().batch_size();
  int offset = 0;
  int N = prefetch_data_[current_buffer_].num();
  int dim = prefetch_data_[current_buffer_].count() / N;
  while (batch_size - offset) {
    if (current_row_ == N) {
      JoinPrefetchThread();
      current_row_ = 0;
      current_buffer_ = 1 - current_buffer_;
      CreatePrefetchThread();
    }
    int avail = std::min(batch_size-offset, N - current_row_);
    caffe_copy(avail * dim, prefetch_data_[current_buffer_].cpu_data() + 
               prefetch_data_[current_buffer_].offset(current_row_),
               (*top)[0]->mutable_gpu_data() + (*top)[0]->offset(offset));
    if (this->output_labels_) {
      dim = prefetch_label_[current_buffer_].count() / N;
      caffe_copy(avail * dim, prefetch_label_[current_buffer_].cpu_data() + 
                  prefetch_label_[current_buffer_].offset(current_row_),
                 (*top)[1]->mutable_gpu_data() + (*top)[1]->offset(offset));
    }
    current_row_ += avail;
    offset += avail;
  }
}

INSTANTIATE_CLASS(BasePrefetchingDataLayer);
INSTANTIATE_CLASS(BaseShufflingDataLayer);

}  // namespace caffe
