#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <boost/iostreams/device/mapped_file.hpp>
#include <opencv2/opencv.hpp>

namespace caffe {

template <typename Dtype>
JPEGDataLayer<Dtype>::~JPEGDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the database resources
  source->close();
  delete source;
  delete cv_img;
}

template <typename Dtype>
void JPEGDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Initialize mmap
  source = new boost::iostreams::mapped_file_source(boost::iostreams::mapped_file_params(this->layer_param_.jpeg_data_param().source()));
  CHECK(source->is_open()) << "cannot open source " << this->layer_param_.jpeg_data_param().source();

  FILE *fin = fopen(this->layer_param_.jpeg_data_param().info().c_str(), "r");
  CHECK(fin) << "cannot open info file " << this->layer_param_.jpeg_data_param().info();
  long long int offset, size, label;
  while (EOF != fscanf(fin, "%*s\t%lld\t%lld\t%lld\n", &offset, &size, &label)) {
    CHECK(offset >= 0 && size >= 0 && offset + size <= source->size()) << "chunk (" << offset << "," << size 
      << ") lies out side file " << this->layer_param_.jpeg_data_param().source() << "!";
    offset_.push_back(offset);
    size_.push_back(size);
    label_.push_back(label);
  }
  fclose(fin);

  cursor_ = 0;

  cv_img = new cv::Mat;

  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  CHECK(DecodeImageToDatum(source->data()+offset_[0], size_[0], *cv_img, 
    0, 0, 0, this->layer_param_.jpeg_data_param().is_color(), &datum)) 
    << "cannot read from source!";
  // image
  int crop_size = this->layer_param_.transform_param().crop_size();
  bool convert_to_grey = this->layer_param_.transform_param().convert_to_grey();
  if (crop_size > 0) {
    if (convert_to_grey) {
      (*top)[0]->Reshape(this->layer_param_.jpeg_data_param().batch_size(),
                         1, crop_size, crop_size);
      this->prefetch_data_.Reshape(this->layer_param_.jpeg_data_param().batch_size(),
          1, crop_size, crop_size);
    }else {
      (*top)[0]->Reshape(this->layer_param_.jpeg_data_param().batch_size(),
                         datum.channels(), crop_size, crop_size);
      this->prefetch_data_.Reshape(this->layer_param_.jpeg_data_param().batch_size(),
          datum.channels(), crop_size, crop_size);
    }  
  } else {
    (*top)[0]->Reshape(
        this->layer_param_.jpeg_data_param().batch_size(), datum.channels(),
        datum.height(), datum.width());
    this->prefetch_data_.Reshape(this->layer_param_.jpeg_data_param().batch_size(),
        datum.channels(), datum.height(), datum.width());
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  if (this->output_labels_) {
    if (this->layer_param_.jpeg_data_param().compute_label()) {
      CHECK(datum.channels() == 3);
      CHECK(datum.data().size());
      int start = this->layer_param_.jpeg_data_param().label_start();
      int stride = this->layer_param_.jpeg_data_param().label_stride();
      int n = ((*top)[0]->width() - 1 - start)/stride + 1;
      (*top)[1]->Reshape(this->layer_param_.jpeg_data_param().batch_size(), 1, n, n);
      this->prefetch_label_.Reshape(this->layer_param_.jpeg_data_param().batch_size(),
          1, n, n);
    } else {
      (*top)[1]->Reshape(this->layer_param_.jpeg_data_param().batch_size(), 1, 1, 1);
      this->prefetch_label_.Reshape(this->layer_param_.jpeg_data_param().batch_size(),
          1, 1, 1);
    }
    
  }
  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void JPEGDataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.jpeg_data_param().batch_size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    CHECK(DecodeImageToDatum(source->data()+offset_[cursor_], size_[cursor_], *cv_img, 
      label_[cursor_], 0, 0, this->layer_param_.jpeg_data_param().is_color(), &datum)) 
      << "cannot read from source at " << offset_[cursor_] << ":" << size_[cursor_];
    cursor_ = (cursor_+1)%offset_.size();

    // Apply data transformations (mirror, scale, crop...)
    int h_off = 0, w_off = 0;
    bool mirrored = false;
    this->data_transformer_.Transform(item_id, datum, this->mean_, top_data, &h_off, &w_off, &mirrored);

    if (this->output_labels_) {
      if (this->layer_param_.jpeg_data_param().compute_label()) {
        CHECK(!mirrored) << "TODO: repair mirror";
        int start = this->layer_param_.jpeg_data_param().label_start();
        int stride = this->layer_param_.jpeg_data_param().label_stride();
        int wend = this->prefetch_data_.width();
        int hend = this->prefetch_data_.height();
        int n = (wend - 1 - start)/stride + 1;
        const string& data = datum.data();
        for (int h = start; h < hend; h += stride) {
          for (int w = start; w < wend; w += stride) {
            int label = 0;
            if (h + h_off >= 0 && h + h_off < datum.height() && 
                w + w_off >= 0 && w + w_off < datum.width()) {
              for (int c = 0; c < datum.channels(); c++) {
                int data_index = (c * datum.height() + h + h_off) * datum.width() + w + w_off;
                label += (1<<(2*(datum.channels()-c-1)))*(static_cast<Dtype>(static_cast<uint8_t>(data[data_index])/64));
                //LOG(INFO) << label;
              }
            }
            if (mirrored) {
              top_label[(item_id*n + (h-start)/stride)*n + (w-start)/stride] = label;
            } else {
              top_label[(item_id*n + (h-start)/stride)*n + n - 1 - (w-start)/stride] = label;
            }
          }
        }
      } else {
        top_label[item_id] = datum.label();
      }
    }
  }
}

INSTANTIATE_CLASS(JPEGDataLayer);

}  // namespace caffe
