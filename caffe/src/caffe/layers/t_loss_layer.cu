#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype, int TILE_DIM>
__global__ void t_distance_kernel(const Dtype *x, const Dtype *y, Dtype *z, int m, int n, int p) {
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
    if (j < m && i < n) z[i+j*n] = c;
}

template <typename Dtype>
void TLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  dim3 grid((N_-1)/TILE_DIM+1, (bottom[0]->num()-1)/TILE_DIM+1, 1);
  dim3 block(TILE_DIM, TILE_DIM, 1);

  const Dtype *gpu_input = bottom[0]->gpu_data();
  const Dtype *gpu_weight = this->blobs_[0]->gpu_data();
  Dtype *gpu_dist = distance_.mutable_gpu_data();
  int M = bottom[0]->num();

  t_distance_kernel<Dtype, TLossLayer<Dtype>::TILE_DIM><<<grid, block>>>(gpu_input, gpu_weight, gpu_dist, M, N_, K_);
  CUDA_POST_KERNEL_CHECK;

  std::cout << "dist: " << distance_.stat_data() << std::endl;

	const Dtype *cpu_distance = distance_.cpu_data();
  const Dtype *cpu_label = bottom[1]->cpu_data();
  const Dtype *cpu_sigma2 = this->blobs_[1]->cpu_data();
  const Dtype *cpu_pi = this->blobs_[2]->cpu_data();
	Dtype *cpu_mask = mask_.mutable_cpu_data();
  Dtype *cpu_coefm = coefm_.mutable_cpu_data();
  Dtype *cpu_coefn = coefn_.mutable_cpu_data();
	Dtype *cpu_proba = (*top)[3]->mutable_cpu_data();
  Dtype *cpu_ind = (*top)[2]->mutable_cpu_data();
  Dtype *cpu_count = count_.mutable_cpu_data();

  Dtype beta = this->layer_param_.t_loss_param().beta();
  Dtype bandwidth = this->layer_param_.t_loss_param().bandwidth();
  Dtype alpha = this->layer_param_.t_loss_param().alpha();
  Dtype alpha_exp =  Dtype((alpha+1.0)/2.0);
  Dtype alpha_weight = Dtype((alpha+1.0)/alpha);

	Dtype loss = Dtype(0);
  //Dtype acc = Dtype(0);
  
  caffe_memset(bottom[0]->num()*N_*sizeof(Dtype), Dtype(0), cpu_mask);
  caffe_memset(bottom[0]->num()*sizeof(Dtype), Dtype(0), cpu_coefm);
  caffe_memset(N_*sizeof(Dtype), Dtype(0), cpu_coefn);
  caffe_memset(N_*sizeof(Dtype), Dtype(0), cpu_count);
  for (int i = 0; i < bottom[0]->num(); i++) {
    Dtype norm = Dtype(0);
    Dtype sqr_norm = Dtype(0);
    int ind = 0;
    Dtype mymin = Dtype(0);
    for (int j = 0; j < N_; j++) {
      if (cpu_distance[i*N_+j] < mymin) {
        mymin = cpu_distance[i*N_+j];
        ind = j;
      }
      Dtype weight = Dtype(1)/(Dtype(1)+cpu_distance[i*N_+j]/alpha);
      norm += std::pow(weight, alpha_exp);
      sqr_norm += weight*weight;
      cpu_mask[i*N_+j] = weight;
    }
    sqr_norm /= norm*norm;
    //bool selected = true;// cpu_mask[i*N_+ind]/norm > cpu_sigma2[ind];
    //Dtype pmax = cpu_mask[i*N_+ind]/norm;
    for (int j = 0; j < N_; j++) {
      Dtype qij = std::pow(cpu_mask[i*N_+j], alpha_exp)/norm;
      Dtype pij = cpu_label[i*N_+j];//qij*qij/sqr_norm;
      cpu_proba[i*N_ + j] = qij;
      /*if (selected) {
        if (j == ind) {
          pij = Dtype(1);
          loss += pij * std::log(pij/qij);
        }else {
          pij = Dtype(0);
        }
        pij = ;
        
      }*/
      cpu_mask[i*N_+j] = alpha_weight*cpu_mask[i*N_+j]*(pij - qij);
      cpu_coefm[i] += cpu_mask[i*N_+j];
      cpu_coefn[j] += cpu_mask[i*N_+j];
      loss += pij * std::log(pij/qij);
    }
  }

  (*top)[0]->mutable_cpu_data()[0] = loss/bottom[0]->num();

  std::cout << "mask: " << mask_.stat_data() << std::endl;
  //std::cout << "acc: " << acc/bottom[0]->num() << std::endl;
  //std::cout << "count: " << count_.stat_data() << std::endl;
}

template <typename Dtype>
void TLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  Dtype weight = top[0]->cpu_diff()[0]/(*bottom)[0]->num();		
  if (propagate_down[0]) {
    caffe_gpu_dgmm(CblasLeft, (*bottom)[0]->num(), K_, (*bottom)[0]->gpu_data(),
      coefm_.gpu_data(), (*bottom)[0]->mutable_gpu_diff());
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, mask_.num(), K_, N_, -weight, 
      mask_.gpu_data(), this->blobs_[0]->gpu_data(), weight, (*bottom)[0]->mutable_gpu_diff());
    //LOG(INFO) << "bottom diff:" << (*bottom)[0]->stat_diff();
  }
  
  if (this->param_propagate_down_[0]) {
    caffe_gpu_dgmm(CblasRight, K_, N_, this->blobs_[0]->gpu_data(), coefn_.gpu_data(), this->blobs_[0]->mutable_gpu_diff());
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, K_, N_, mask_.num(), -weight, 
      (*bottom)[0]->gpu_data(), mask_.gpu_data(), weight, this->blobs_[0]->mutable_gpu_diff());
    //std::cout << "diff: " <<  this->blobs_[0]->stat_diff() << std::endl;
    //LOG(INFO) << "center diff:" << this->blobs_[0]->stat_diff();
  }
}

INSTANTIATE_CLASS(TLossLayer);

}  // namespace caffe
