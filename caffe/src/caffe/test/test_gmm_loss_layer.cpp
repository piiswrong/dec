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

const int dim = 100;

template <typename TypeParam>
class GMMLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  GMMLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(dim, 1, 1, dim)),
        blob_bottom_label_(new Blob<Dtype>(dim, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()),
        blob_top_std_(new Blob<Dtype>()),
        blob_top_ind_(new Blob<Dtype>()), 
        blob_top_dist_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);

    for (int i = 0; i < dim; i++) 
      blob_bottom_label_->mutable_cpu_data()[i] = i;

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
    blob_top_vec_.push_back(blob_top_std_);
    blob_top_vec_.push_back(blob_top_ind_);
    blob_top_vec_.push_back(blob_top_dist_);
  }
  virtual ~GMMLossLayerTest() {
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
    layer_param.mutable_gmm_loss_param()->set_beta(10);
    layer_param.mutable_gmm_loss_param()->set_num_center(dim);
    GMMLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
    layer_weight_1.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);

    FillerParameter filler_param;
    GaussianFiller<Dtype> filler2(filler_param);
    filler2.Fill(layer_weight_1.blobs()[0].get());

    const Dtype loss_weight_1 =
        layer_weight_1.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Dtype kLossWeight = 3.7;
    LayerParameter layer_param2;
    layer_param2.mutable_gmm_loss_param()->set_beta(10);
    layer_param2.mutable_gmm_loss_param()->set_num_center(dim);
    layer_param2.add_loss_weight(kLossWeight);
    GMMLossLayer<Dtype> layer_weight_2(layer_param2);
    layer_weight_2.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
    layer_weight_2.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);

    caffe_copy(layer_weight_2.blobs()[0]->count(), layer_weight_1.blobs()[0]->cpu_data(), 
      layer_weight_2.blobs()[0]->mutable_cpu_data());

    const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-3;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);
    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);

    int m = dim, n = layer_param.gmm_loss_param().num_center(), p = dim;
    Blob<Dtype> *distance = layer_weight_1.distance();
    const Dtype *cpu_data = blob_bottom_data_->cpu_data();
    const Dtype *cpu_dist = distance->cpu_data();
    const Dtype *cpu_center = layer_weight_1.blobs()[0]->cpu_data();
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        Dtype acc = Dtype(0);
        for (int k = 0; k < p; ++k) {
          acc += (cpu_data[i*p + k] - cpu_center[k*n + j])*(cpu_data[i*p + k] - cpu_center[k*n + j]);
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

TYPED_TEST_CASE(GMMLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(GMMLossLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(GMMLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 1.0;
  layer_param.add_loss_weight(kLossWeight);
  layer_param.mutable_gmm_loss_param()->set_beta(10);
  layer_param.mutable_gmm_loss_param()->set_num_center(dim);
  layer_param.mutable_gmm_loss_param()->set_lambda(1);

  
  GMMLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);

  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_data_);
  filler.Fill(layer.blobs()[0].get());

  GradientChecker<Dtype> checker(1e-1, 1e-3, 1701);
  checker.CheckGradientSingle(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_), 0, 0, 0);
}

}  // namespace caffe
