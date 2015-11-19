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
class ResizeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ResizeLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 6, 5);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ResizeLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestForward() {
    int factor = 8;
    LayerParameter layer_param;
    ResizeParameter *resize_param = layer_param.mutable_resize_param();
    resize_param->set_top_width(blob_bottom_->width()*factor);
    resize_param->set_top_height(blob_bottom_->height()*factor);
    ResizeLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);

    EXPECT_EQ(blob_top_->num(), blob_bottom_->num());
    EXPECT_EQ(blob_top_->channels(), blob_bottom_->channels());
    EXPECT_EQ(blob_top_->width(), blob_bottom_->width()*factor);
    EXPECT_EQ(blob_top_->height(), blob_bottom_->height()*factor);

    layer.Forward(blob_bottom_vec_, &blob_top_vec_);

    const Dtype *cpu_bottom = blob_bottom_->cpu_data();
    const Dtype *cpu_top = blob_top_->cpu_data();
    for (int i = 0; i < blob_top_->num(); i++) {
      for (int c = 0; c < blob_top_->channels(); c++) {
        for (int h = 0; h < blob_top_->height(); h++) {
          for (int w = 0; w < blob_top_->width(); w++) {
            EXPECT_EQ(cpu_top[((i*blob_top_->channels() + c)*blob_top_->height() + h)*blob_top_->width() + w], 
              cpu_bottom[((i*blob_bottom_->channels() + c)*blob_bottom_->height() + h/factor)*blob_bottom_->width() + w/factor])
              << "n=" << i << " c=" << c << " h=" << h << " w=" << w;
          }
        }
      }
    }
  }

  void TestBackward() {
    int factor = 8;
    Dtype epsilon = 1e-8;
    LayerParameter layer_param;
    ResizeParameter *resize_param = layer_param.mutable_resize_param();
    resize_param->set_top_width(blob_bottom_->width()*factor);
    resize_param->set_top_height(blob_bottom_->height()*factor);
    ResizeLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);

    EXPECT_EQ(blob_top_->num(), blob_bottom_->num());
    EXPECT_EQ(blob_top_->channels(), blob_bottom_->channels());
    EXPECT_EQ(blob_top_->width(), blob_bottom_->width()*factor);
    EXPECT_EQ(blob_top_->height(), blob_bottom_->height()*factor);

    layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    caffe_copy(blob_top_->count(), blob_top_->cpu_data(), blob_top_->mutable_cpu_diff());
    layer.Backward(blob_top_vec_, std::vector<bool>(1, true), &blob_bottom_vec_);

    const Dtype *cpu_bottom = blob_bottom_->cpu_diff();
    const Dtype *cpu_top = blob_top_->cpu_diff();
    for (int i = 0; i < blob_bottom_->num(); i++) {
      for (int c = 0; c < blob_bottom_->channels(); c++) {
        for (int h = 0; h < blob_bottom_->height(); h++) {
          for (int w = 0; w < blob_bottom_->width(); w++) {
            Dtype sum = 0;
            for (int hi = 0; hi < factor; hi++) {
              for (int wi = 0; wi < factor; wi++) {
                sum += cpu_top[((i*blob_top_->channels() + c)*blob_top_->height() + h*factor+hi)*blob_top_->width() + w*factor+wi];
              }
            }
            EXPECT_NEAR(cpu_bottom[((i*blob_bottom_->channels() + c)*blob_bottom_->height() + h)*blob_bottom_->width() + w], 
              sum, epsilon) << "n=" << i << " c=" << c << " h=" << h << " w=" << w;
          }
        }
      }
    }
  }
};

TYPED_TEST_CASE(ResizeLayerTest, TestDtypesAndDevices);

TYPED_TEST(ResizeLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(ResizeLayerTest, TestBackward) {
  this->TestBackward();
}

}  // namespace caffe
