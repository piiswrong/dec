#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SoftmaxCrossEntropyLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SoftmaxCrossEntropyLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 50, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 50, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    caffe_rng_uniform(blob_bottom_label_->count(), Dtype(0), Dtype(1), blob_bottom_label_->mutable_cpu_data());
    int channels = blob_bottom_label_->count()/blob_bottom_label_->num();
    for (int i = 0; i < blob_bottom_label_->num(); ++i) {
      Dtype norm = 0;
      for (int j = 0; j < channels; j++) {
        //blob_bottom_label_->mutable_cpu_data()[i*channels + j] = Dtype(j);
        norm += blob_bottom_label_->mutable_cpu_data()[i*channels + j];
      }
      for (int j = 0; j < channels; j++) {
        blob_bottom_label_->mutable_cpu_data()[i*channels + j] /= norm;
      }
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~SoftmaxCrossEntropyLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxCrossEntropyLossLayerTest, TestDtypesAndDevices);


TYPED_TEST(SoftmaxCrossEntropyLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  SoftmaxCrossEntropyLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_), 0);
}

}  // namespace caffe
