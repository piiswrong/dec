#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype, int TILE_DIM>
__global__ void entropy_t_distance_kernel(const Dtype *x, const Dtype *y, const Dtype *sigma, Dtype *z, int m, int n, int p) {
    __shared__ Dtype sx[TILE_DIM][TILE_DIM+1];
    __shared__ Dtype sy[TILE_DIM][TILE_DIM+1];
    __shared__ Dtype ssigma[TILE_DIM][TILE_DIM+1];

    const int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    const int i = bx*TILE_DIM + tx;
    const int j = by*TILE_DIM + ty;
    Dtype c = 0.0;
    for (int k = 0; k < p; k += TILE_DIM) {
        if (k + tx < p && j < m) sx[ty][tx] = x[j*p + k + tx];
        else                     sx[ty][tx] = Dtype(0);

        if (k + ty < p && i < n) {
          sy[ty][tx] = y[(k+ty)*n + i];
          ssigma[ty][tx] = sigma[(k+ty)*n + i];
        }else {
          sy[ty][tx] = Dtype(0);
          ssigma[ty][tx] = Dtype(1);
        }

        __syncthreads();

        for (int kk = 0; kk < TILE_DIM; kk++) {
            c += (sx[ty][kk] - sy[kk][tx])*(sx[ty][kk] - sy[kk][tx])*ssigma[kk][tx];
        }
        __syncthreads();
    }
    if (j < m && i < n) z[i+j*n] = c;
}

template <typename Dtype>
void EntropyTLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  dim3 grid((N_-1)/TILE_DIM+1, (bottom[0]->num()-1)/TILE_DIM+1, 1);
  dim3 block(TILE_DIM, TILE_DIM, 1);

  const Dtype *gpu_input = bottom[0]->gpu_data();
  const Dtype *gpu_weight = this->blobs_[0]->gpu_data();
  const Dtype *gpu_inv_sigma2 = this->blobs_[1]->gpu_data();
  Dtype *gpu_dist = distance_.mutable_gpu_data();
  int M = bottom[0]->num();

  entropy_t_distance_kernel<Dtype, EntropyTLossLayer<Dtype>::TILE_DIM><<<grid, block>>>(gpu_input, gpu_weight, gpu_inv_sigma2, gpu_dist, M, N_, K_);
  CUDA_POST_KERNEL_CHECK;

  //std::cout << "dist: " << distance_.stat_data() << std::endl;

	const Dtype *cpu_distance = distance_.cpu_data();
  const Dtype *cpu_label = bottom[1]->cpu_data();
  const Dtype *cpu_inv_sigma2 = this->blobs_[1]->cpu_data();
  Dtype *cpu_inv_sigma_prod = sigma_prod_.mutable_cpu_data();
  const Dtype *cpu_pi = this->blobs_[2]->cpu_data();
	Dtype *cpu_mask = mask_.mutable_cpu_data();
  Dtype *cpu_coefm = coefm_.mutable_cpu_data();
  Dtype *cpu_coefn = coefn_.mutable_cpu_data();
	Dtype *cpu_proba = (*top)[3]->mutable_cpu_data();
  Dtype *cpu_ind = (*top)[2]->mutable_cpu_data();
  Dtype *cpu_count = count_.mutable_cpu_data();

  Dtype beta = this->layer_param_.multi_t_loss_param().beta();
  Dtype bandwidth = this->layer_param_.multi_t_loss_param().bandwidth();
  Dtype alpha = this->layer_param_.multi_t_loss_param().alpha();
  Dtype alpha_exp =  Dtype((alpha+1.0)/2.0);
  Dtype alpha_weight = Dtype((alpha+1.0)/alpha);

	Dtype loss = Dtype(0);
  //Dtype acc = Dtype(0);


  for (int i = 0; i < N_; i++) {
    Dtype prod = Dtype(1);
    for (int j = 0; j < K_; j++) prod *= cpu_inv_sigma2[j*N_+i];
    cpu_inv_sigma_prod[i] = std::sqrt(prod);
  }
  
  caffe_memset(bottom[0]->num()*N_*sizeof(Dtype), Dtype(0), cpu_mask);
  caffe_memset(bottom[0]->num()*sizeof(Dtype), Dtype(0), cpu_coefm);
  caffe_memset(N_*sizeof(Dtype), Dtype(0), cpu_coefn);
  caffe_memset(N_*sizeof(Dtype), Dtype(0), cpu_count);
  for (int i = 0; i < bottom[0]->num(); i++) {
    Dtype norm = Dtype(0);
    Dtype l_i = Dtype(0);
    for (int j = 0; j < N_; j++) {
      cpu_mask[i*N_+j] = Dtype(1)/(Dtype(1)+cpu_distance[i*N_+j]/alpha);
      cpu_proba[i*N_ + j] = cpu_inv_sigma_prod[j]*std::pow(cpu_mask[i*N_+j], alpha_exp); 
      norm += cpu_proba[i*N_ + j];
    }
    for (int j = 0; j < N_; j++) {
      cpu_proba[i*N_ + j] /= norm;
      l_i += - cpu_proba[i*N_ + j]*std::log(cpu_proba[i*N_ + j]);
    }
    for (int j = 0; j < N_; j++) {
      cpu_mask[i*N_+j] = alpha_weight*(std::log(cpu_proba[i*N_ + j]) + l_i)*cpu_proba[i*N_ + j]*cpu_mask[i*N_+j];
      cpu_coefm[i] += cpu_mask[i*N_+j];
      cpu_coefn[j] += cpu_mask[i*N_+j];
    }
    loss += l_i;
  }

  (*top)[0]->mutable_cpu_data()[0] = loss/bottom[0]->num();

  //std::cout << "mask: " << mask_.stat_data() << std::endl;
  //std::cout << "coefm: " << coefm_.stat_data() << std::endl;
  //std::cout << "proba: " << (*top)[3]->stat_data() << std::endl;
  //(*top)[3]->sample_data(10,10);
  //sigma_prod_.sample_data(10,10);
  //std::cout << "acc: " << acc/bottom[0]->num() << std::endl;
  //std::cout << "count: " << count_.stat_data() << std::endl;
}

template <typename Dtype>
void EntropyTLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  Dtype weight = top[0]->cpu_diff()[0]/(*bottom)[0]->num();		
  if (propagate_down[0]) {
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, mask_.num(), K_, N_, weight,
      mask_.gpu_data(), this->blobs_[1]->gpu_data(), Dtype(0), (*bottom)[0]->mutable_gpu_diff());
    caffe_gpu_mul(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(), this->blobs_[1]->gpu_data(), mu_sigma_.mutable_gpu_data());
    caffe_gpu_mul((*bottom)[0]->count(), (*bottom)[0]->gpu_diff(), (*bottom)[0]->gpu_data(), (*bottom)[0]->mutable_gpu_diff());
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, mask_.num(), K_, N_, -weight,
      mask_.gpu_data(), mu_sigma_.gpu_data(), Dtype(1), (*bottom)[0]->mutable_gpu_diff());
    //caffe_gpu_axpy((*bottom)[0]->count(), -weight, (*bottom)[0]->gpu_data(), (*bottom)[0]->mutable_gpu_diff());
    //LOG(INFO) << "bottom diff:" << (*bottom)[0]->stat_diff();
  }
  
  if (this->param_propagate_down_[0]) {
    caffe_gpu_dgmm(CblasRight, K_, N_, this->blobs_[0]->gpu_data(), coefn_.gpu_data(), this->blobs_[0]->mutable_gpu_diff());
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, K_, N_, mask_.num(), -weight,
      (*bottom)[0]->gpu_data(), mask_.gpu_data(), weight, this->blobs_[0]->mutable_gpu_diff());
    caffe_gpu_mul(this->blobs_[0]->count(), this->blobs_[0]->gpu_diff(), this->blobs_[1]->gpu_data(), this->blobs_[0]->mutable_gpu_diff());
    //std::cout << "diff: " <<  this->blobs_[0]->stat_diff() << std::endl;
    //LOG(INFO) << "center diff:" << this->blobs_[0]->stat_diff();
  }
}

INSTANTIATE_CLASS(EntropyTLossLayer);

}  // namespace caffe
