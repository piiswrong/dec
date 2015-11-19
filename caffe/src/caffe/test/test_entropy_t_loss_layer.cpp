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

const int M_ = 10;
const int K_ = 10;
const int N_ = 4;

template <typename TypeParam>
class EntropyTLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  EntropyTLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(M_, 1, 1, K_)),
        blob_bottom_label_(new Blob<Dtype>(M_, 1, 1, N_)),
        blob_top_loss_(new Blob<Dtype>()),
        blob_top_std_(new Blob<Dtype>()),
        blob_top_ind_(new Blob<Dtype>()), 
        blob_top_dist_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(0);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);

    Dtype *cpu_label = blob_bottom_label_->mutable_cpu_data();
    caffe_rng_uniform(blob_bottom_label_->count(), Dtype(0), Dtype(1), cpu_label);
    for (int i = 0; i < M_; i++) {
      Dtype norm = Dtype(0);
      for (int j = 0; j < N_; j++) {
        norm += cpu_label[i*N_+j];
      }
      for (int j = 0; j < N_; j++) {
        cpu_label[i*N_+j] /= norm;
      }
    }

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
    blob_top_vec_.push_back(blob_top_std_);
    blob_top_vec_.push_back(blob_top_ind_);
    blob_top_vec_.push_back(blob_top_dist_);
  }
  virtual ~EntropyTLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
    delete blob_top_ind_;
    delete blob_top_std_;
    delete blob_top_dist_;
  }

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    layer_param.mutable_multi_t_loss_param()->set_num_center(N_);
    EntropyTLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
    layer_weight_1.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);

    FillerParameter filler_param;
    GaussianFiller<Dtype> filler2(filler_param);
    filler2.Fill(layer_weight_1.blobs()[0].get());
    caffe_rng_uniform(layer_weight_1.blobs()[1]->count(), Dtype(0.9), Dtype(1.1), layer_weight_1.blobs()[1]->mutable_cpu_data());
    layer_weight_1.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);

    const Dtype loss_weight_1 =
        layer_weight_1.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Dtype kLossWeight = 3.7;
    LayerParameter layer_param2;
    layer_param2.mutable_multi_t_loss_param()->set_num_center(N_);
    layer_param2.add_loss_weight(kLossWeight);
    EntropyTLossLayer<Dtype> layer_weight_2(layer_param2);
    layer_weight_2.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
    layer_weight_2.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);

    caffe_copy(layer_weight_2.blobs()[0]->count(), layer_weight_1.blobs()[0]->cpu_data(), 
      layer_weight_2.blobs()[0]->mutable_cpu_data());
    caffe_copy(layer_weight_2.blobs()[1]->count(), layer_weight_1.blobs()[1]->cpu_data(), 
      layer_weight_2.blobs()[1]->mutable_cpu_data());
    layer_weight_2.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);

    const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-3;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);
    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);

    int m = M_, n = layer_param.multi_t_loss_param().num_center(), p = K_;
    Blob<Dtype> *distance = layer_weight_1.distance();
    const Dtype *cpu_data = blob_bottom_data_->cpu_data();
    const Dtype *cpu_dist = distance->cpu_data();
    const Dtype *cpu_center = layer_weight_1.blobs()[0]->cpu_data();
    const Dtype *cpu_sigma = layer_weight_1.blobs()[1]->cpu_data();
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        Dtype acc = Dtype(0);
        for (int k = 0; k < p; ++k) {
          acc += (cpu_data[i*p + k] - cpu_center[k*n + j])*(cpu_data[i*p + k] - cpu_center[k*n + j])*cpu_sigma[k*n+j];
        }
        EXPECT_NEAR(acc, cpu_dist[i*n + j], kErrorMargin) << i << " " << j;
      }
    }
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  Blob<Dtype>* const blob_top_std_;
  Blob<Dtype>* const blob_top_ind_;
  Blob<Dtype>* const blob_top_dist_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(EntropyTLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(EntropyTLossLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(EntropyTLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 1.0;
  layer_param.add_loss_weight(kLossWeight);
  layer_param.mutable_multi_t_loss_param()->set_num_center(N_);

  
  EntropyTLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);

  FillerParameter filler_param;
  filler_param.set_value(0);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_data_); 
  filler.Fill(layer.blobs()[0].get());
  //caffe_rng_uniform(layer.blobs()[1]->count(), Dtype(0.9), Dtype(1.1), layer.blobs()[1]->mutable_cpu_data());
  //layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);

  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientSingle(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_), 0, 0, 0);
}

}  // namespace caffe
