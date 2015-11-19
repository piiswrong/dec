#include <string>

#include "caffe/data_transformer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const int batch_item_id,
                                       const Datum& datum,
                                       const Dtype* mean,
                                       Dtype* transformed_data,
                                       int *h_off,
                                       int *w_off,
                                       bool *mirrored) {
  const string& data = datum.data();
  const int channels = datum.channels();
  const int height = datum.height();
  const int width = datum.width();
  const int size = datum.channels() * datum.height() * datum.width();

  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();
  const int soffset = param_.soffset();
  const int eoffset = param_.eoffset();

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
               << "set at the same time.";
  }

  if (crop_size) {
    CHECK(data.size()) << "Image cropping only support uint8 data";
    // We only do random crop when we do training.
    if (h_off == NULL || w_off == NULL || *h_off == 0 || *w_off == 0) {
      if (phase_ == Caffe::TRAIN) {
        if (soffset != 0 || eoffset != 0) {
          *h_off = Rand() % (eoffset - soffset) + soffset;
          *w_off = Rand() % (eoffset - soffset) + soffset;
        } else {
          *h_off = Rand() % (height - crop_size);
          *w_off = Rand() % (width - crop_size);
        }
      } else {
        *h_off = (height - crop_size) / 2;
        *w_off = (width - crop_size) / 2;
      }
    }
    if (mirror && Rand() % 2) {
      if (mirrored) *mirrored = true;
      // Copy mirrored version
      if (param_.convert_to_grey()) {
        CHECK(channels == 3) << "Convert to grey scale is only supported when input has 3 channels.";
        const Dtype weight[] = {0.114, 0.587, 0.299};
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int top_index = (batch_item_id * crop_size + h)
                * crop_size + (crop_size - 1 - w);
            if (h + *h_off >= 0 && h + *h_off < height && 
                w + *w_off >= 0 && w + *w_off < width) {
              Dtype datum_element = Dtype(0);
              for (int c = 0; c < channels; ++c) {
                int data_index = (c * height + h + *h_off) * width + w + *w_off;
                datum_element += weight[c] *
                  (static_cast<Dtype>(static_cast<uint8_t>(data[data_index])) - mean[data_index]);
              }
              transformed_data[top_index] = datum_element * scale;
            }else {
              transformed_data[top_index] = Dtype(0);
            }
          }
        }
      } else {
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < crop_size; ++h) {
            for (int w = 0; w < crop_size; ++w) {
              int data_index = (c * height + h + *h_off) * width + w + *w_off;
              int top_index = ((batch_item_id * channels + c) * crop_size + h)
                  * crop_size + (crop_size - 1 - w);
              if (h + *h_off >= 0 && h + *h_off < height && 
                  w + *w_off >= 0 && w + *w_off < width) {
                Dtype datum_element =
                    static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
                transformed_data[top_index] =
                    (datum_element - mean[data_index]) * scale;
              }else {
                transformed_data[top_index] = Dtype(0);
              }
            }
          }
        }
      }
    } else {
      if (mirrored) *mirrored = false;
      // Normal copy
      if (param_.convert_to_grey()) {
        CHECK(channels == 3) << "Convert to grey scale is only supported when input has 3 channels.";
        const Dtype weight[] = {0.114, 0.587, 0.299};
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int top_index = (batch_item_id * crop_size + h)
                  * crop_size + w;
            if (h + *h_off >= 0 && h + *h_off < height && 
                w + *w_off >= 0 && w + *w_off < width) {
              Dtype datum_element = Dtype(0);
              for (int c = 0; c < channels; ++c) {
                int data_index = (c * height + h + *h_off) * width + w + *w_off;
                datum_element += weight[c] *
                  static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
              }
              int data_index = (0 * height + h + *h_off) * width + w + *w_off;
              transformed_data[top_index] =
                  (datum_element - mean[data_index]) * scale;
            }else {
              transformed_data[top_index] = Dtype(0);
            }
          }
        }
      } else {
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < crop_size; ++h) {
            for (int w = 0; w < crop_size; ++w) {
              int top_index = ((batch_item_id * channels + c) * crop_size + h)
                  * crop_size + w;
              int data_index = (c * height + h + *h_off) * width + w + *w_off;
              if (h + *h_off >= 0 && h + *h_off < height && 
                  w + *w_off >= 0 && w + *w_off < width) {
                Dtype datum_element =
                    static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
                transformed_data[top_index] =
                    (datum_element - mean[data_index]) * scale;
              }else {
                transformed_data[top_index] = Dtype(0);
              }
            }
          }
        }
      }
    }
  } else {
    CHECK(!param_.convert_to_grey());
    // we will prefer to use data() first, and then try float_data()
    if (data.size()) {
      for (int j = 0; j < size; ++j) {
        Dtype datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[j]));
        transformed_data[j + batch_item_id * size] =
            (datum_element - mean[j]) * scale;
      }
    } else {
      for (int j = 0; j < size; ++j) {
        transformed_data[j + batch_item_id * size] =
            (datum.float_data(j) - mean[j]) * scale;
      }
    }
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = (phase_ == Caffe::TRAIN) &&
      (param_.mirror() || param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
unsigned int DataTransformer<Dtype>::Rand() {
  CHECK(rng_);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return (*rng)();
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
