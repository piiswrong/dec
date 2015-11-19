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

const int dim = 32;

template <typename TypeParam>
class ClusteringLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ClusteringLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(dim, 1, 1, dim)),
        blob_bottom_label_(new Blob<Dtype>(dim, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()),
        blob_top_std_(new Blob<Dtype>()),
        blob_top_ind_(new Blob<Dtype>()), 
        blob_top_dist_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(0.0005);
    ConstantFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);

    Dtype *data = this->blob_bottom_data_->mutable_cpu_data();
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        data[i*dim + j] *= j;
      }
      data[i*dim + i] += Dtype(1.0);
    }

    Dtype *cpu_data = this->blob_bottom_label_->mutable_cpu_data();
    for (int i = 0; i < dim; i++) {
      if (i < dim/2)
        cpu_data[i] = Dtype(1);
      else
        cpu_data[i] = Dtype(0);

    }
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
    blob_top_vec_.push_back(blob_top_std_);
    blob_top_vec_.push_back(blob_top_ind_);
    blob_top_vec_.push_back(blob_top_dist_);
  }
  virtual ~ClusteringLossLayerTest() {
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
    layer_param.mutable_clustering_loss_param()->set_margin(10);
    layer_param.mutable_clustering_loss_param()->set_num_center(dim);
    ClusteringLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
    layer_weight_1.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);

    FillerParameter filler_param;
    filler_param.set_value(-0.0005);
    ConstantFiller<Dtype> filler2(filler_param);
    filler2.Fill(layer_weight_1.blobs()[0].get());

    Dtype *center = layer_weight_1.blobs()[0]->mutable_cpu_data();
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        center[j*dim + i] *= j;
      }
      center[i*dim + i] += Dtype(1.0);
    }

    const Dtype loss_weight_1 =
        layer_weight_1.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);


    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Dtype kLossWeight = 3.7;
    LayerParameter layer_param2;
    layer_param2.mutable_clustering_loss_param()->set_margin(10);
    layer_param2.mutable_clustering_loss_param()->set_num_center(dim);
    layer_param2.add_loss_weight(kLossWeight);
    ClusteringLossLayer<Dtype> layer_weight_2(layer_param2);
    layer_weight_2.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
    layer_weight_2.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);

    filler2.Fill(layer_weight_2.blobs()[0].get());

    center = layer_weight_2.blobs()[0]->mutable_cpu_data();
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        center[j*dim + i] *= j;
      }
      center[i*dim + i] += Dtype(1.0);
    }

    const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-5;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);
    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);

    int m = dim, n = layer_param.clustering_loss_param().num_center(), p = dim;
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
        EXPECT_NEAR(std::sqrt(acc/p), cpu_dist[i*n + j], kErrorMargin);
      }
    }
    /*
    const Dtype *cpu_ind = blob_top_ind_->cpu_data();
    for (int i = 0; i < m; i++) 
      EXPECT_EQ(cpu_ind[i], i);
    EXPECT_NEAR(blob_top_std_->cpu_data()[0], Dtype(0), kErrorMargin);*/
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

TYPED_TEST_CASE(ClusteringLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(ClusteringLossLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(ClusteringLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 1.0;
  layer_param.add_loss_weight(kLossWeight);
  layer_param.mutable_clustering_loss_param()->set_margin(10);
  layer_param.mutable_clustering_loss_param()->set_num_center(dim);
  layer_param.mutable_clustering_loss_param()->set_lambda(0.5);

  
  ClusteringLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);

  FillerParameter filler_param;
  filler_param.set_value(0.0005);
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_data_);
  filler_param.set_value(-0.0005);
  ConstantFiller<Dtype> filler2(filler_param);
  filler2.Fill(layer.blobs()[0].get());

  Dtype *center = layer.blobs()[0]->mutable_cpu_data();
  Dtype *data = this->blob_bottom_data_->mutable_cpu_data();
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      data[i*dim + j] *= j;
      center[j*dim + i] *= j;
    }
    center[i*dim + i] += Dtype(1.0);
    data[i*dim + i] += Dtype(1.0);
  }

  GradientChecker<Dtype> checker(1e-1, 1e-10, 1701);
  checker.CheckGradientSingle(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_), 0, 0, 0);
}

}  // namespace caffe
