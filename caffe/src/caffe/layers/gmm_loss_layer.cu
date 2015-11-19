#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype, int TILE_DIM>
__global__ void gmm_distance_kernel(const Dtype *x, const Dtype *y, Dtype *z, int m, int n, int p) {
    __shared__ Dtype sx[TILE_DIM][TILE_DIM+1];
    __shared__ Dtype sy[TILE_DIM][TILE_DIM+1];

    const int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    const int i = bx*TILE_DIM + tx;
    const int j = by*TILE_DIM + ty;
    Dtype c = 0.0;
    for (int k = 0; k < p; k += TILE_DIM) {
        if (k + tx < p && j < m) sx[ty][tx] = x[j*p + k + tx];
        else                     sx[ty][tx] = Dtype(0);

        if (k + ty < p && i < n) sy[ty][tx] = y[(k+ty)*n + i];
        else                     sy[ty][tx] = Dtype(0);

        __syncthreads();

        for (int kk = 0; kk < TILE_DIM; kk++) {
            c += (sx[ty][kk] - sy[kk][tx])*(sx[ty][kk] - sy[kk][tx]);
        }
        __syncthreads();
    }
    if (j < m && i < n) z[i+j*n] = c/p;
}

template <typename Dtype>
void GMMLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  dim3 grid((N_-1)/TILE_DIM+1, (bottom[0]->num()-1)/TILE_DIM+1, 1);
  dim3 block(TILE_DIM, TILE_DIM, 1);

  const Dtype *gpu_input = bottom[0]->gpu_data();
  const Dtype *gpu_weight = this->blobs_[0]->gpu_data();
  Dtype *gpu_dist = distance_.mutable_gpu_data();
  int M = bottom[0]->num();

  gmm_distance_kernel<Dtype, GMMLossLayer<Dtype>::TILE_DIM><<<grid, block>>>(gpu_input, gpu_weight, gpu_dist, M, N_, K_);
  CUDA_POST_KERNEL_CHECK;

  std::cout << "dist: " << distance_.stat_data() << std::endl;

	const Dtype *cpu_distance = distance_.cpu_data();
  const Dtype *cpu_label = bottom[1]->cpu_data();
  const Dtype *cpu_sigma2 = this->blobs_[1]->cpu_data();
  const Dtype *cpu_pi = this->blobs_[2]->cpu_data();
	Dtype *cpu_mask = mask_.mutable_cpu_data();
  Dtype *cpu_coefm = coefm_.mutable_cpu_data();
  Dtype *cpu_coefn = coefn_.mutable_cpu_data();
	Dtype *cpu_min_distance = (*top)[3]->mutable_cpu_data();
  Dtype *cpu_ind = (*top)[2]->mutable_cpu_data();
  Dtype *cpu_count = count_.mutable_cpu_data();

  Dtype beta = this->layer_param_.gmm_loss_param().beta();
  Dtype bandwidth = this->layer_param_.gmm_loss_param().bandwidth();

	Dtype loss = Dtype(0);
  Dtype acc = Dtype(0);
  
  caffe_memset(bottom[0]->num()*N_*sizeof(Dtype), Dtype(0), cpu_mask);
  caffe_memset(bottom[0]->num()*sizeof(Dtype), Dtype(0), cpu_coefm);
  caffe_memset(N_*sizeof(Dtype), Dtype(0), cpu_coefn);
  caffe_memset(N_*sizeof(Dtype), Dtype(0), cpu_count);
  for (int i = 0; i < bottom[0]->num(); i++) {
    Dtype norm = Dtype(0);
    for (int j = 0; j < N_; j++) {
      Dtype sigma2 = cpu_sigma2[j];
      Dtype pi = cpu_pi[j];
      Dtype weight = pi*std::exp(-cpu_distance[i*N_+j]/Dtype(2)/sigma2);// - K_/Dtype(2)*std::log(sigma2));
      cpu_mask[i*N_+j] = weight/sigma2;
      norm += weight;
    }
    for (int j = 0; j < N_; j++) {
      //cpu_mask[i*N_+j] /= norm;
      cpu_coefm[i] += cpu_mask[i*N_+j];
      cpu_coefn[j] += cpu_mask[i*N_+j];
    }
    loss -= norm;
  }


  Dtype Sb = Dtype(0);
  Dtype *cpu_mean = mean_.mutable_cpu_data();
  const Dtype *cpu_x = bottom[0]->cpu_data();
  Dtype *cpu_diff = diff_.mutable_cpu_data();
  caffe_memset(mean_.count()*sizeof(Dtype), Dtype(0), cpu_mean);

  for (int i = 0; i < bottom[0]->num(); i++) 
    caffe_add(K_, cpu_x+i*K_, cpu_mean, cpu_mean);

  caffe_cpu_scale(K_, Dtype(1)/bottom[0]->num(), cpu_mean, cpu_mean);

  for (int i = 0; i < bottom[0]->num(); i++) {
    caffe_sub(K_, cpu_x+i*K_, cpu_mean, cpu_diff+i*K_);
    for (int j = 0; j < K_; j++) 
      Sb += cpu_diff[i*K_ + j]*cpu_diff[i*K_ + j];
  }
  Sb /= Dtype(2)*bottom[0]->num();

  std:: cout << "pos: " << loss/bottom[0]->num()
             << "\tneg: " << lambda_ * std::max(beta - Sb, Dtype(0)) 
             << std::endl;
  loss = loss/bottom[0]->num() + lambda_ * std::max(beta - Sb, Dtype(0));
  sign_ = beta - Sb > Dtype(0);

  (*top)[0]->mutable_cpu_data()[0] = loss;
  (*top)[1]->mutable_cpu_data()[0] = acc/bottom[0]->num();

  //std::cout << "loss: " << loss << std::endl;
  //std::cout << "acc: " << acc/bottom[0]->num() << std::endl;
  //std::cout << "count: " << count_.stat_data() << std::endl;
}

template <typename Dtype>
void GMMLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  Dtype weight = top[0]->cpu_diff()[0]/K_/(*bottom)[0]->num();		
  if (propagate_down[0]) {
    caffe_gpu_dgmm(CblasLeft, (*bottom)[0]->num(), K_, (*bottom)[0]->gpu_data(),
      coefm_.gpu_data(), (*bottom)[0]->mutable_gpu_diff());
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, mask_.num(), K_, N_, -weight, 
      mask_.gpu_data(), this->blobs_[0]->gpu_data(), weight, (*bottom)[0]->mutable_gpu_diff());
    //LOG(INFO) << "bottom diff:" << (*bottom)[0]->stat_diff();
    if (sign_) {
      caffe_gpu_axpy((*bottom)[0]->count(), -lambda_*weight, diff_.gpu_data(), (*bottom)[0]->mutable_gpu_diff());
    }
  }
  
  if (this->param_propagate_down_[0]) {
    caffe_gpu_dgmm(CblasRight, K_, N_, this->blobs_[0]->gpu_data(), coefn_.gpu_data(), this->blobs_[0]->mutable_gpu_diff());
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, K_, N_, mask_.num(), -weight, 
      (*bottom)[0]->gpu_data(), mask_.gpu_data(), weight, this->blobs_[0]->mutable_gpu_diff());
    //std::cout << "diff: " <<  this->blobs_[0]->stat_diff() << std::endl;
    //LOG(INFO) << "center diff:" << this->blobs_[0]->stat_diff();
  }
}

INSTANTIATE_CLASS(GMMLossLayer);

}  // namespace caffe
