#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype, int TILE_DIM>
__global__ void kmeans_distance_kernel(const Dtype *x, const Dtype *y, Dtype *z, int m, int n, int p) {
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
void KmeansLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  dim3 grid((N_-1)/TILE_DIM+1, (bottom[0]->num()-1)/TILE_DIM+1, 1);
  dim3 block(TILE_DIM, TILE_DIM, 1);

  const Dtype *gpu_input = bottom[0]->gpu_data();
  const Dtype *gpu_weight = this->blobs_[0]->gpu_data();
  Dtype *gpu_dist = distance_.mutable_gpu_data();
  int M = bottom[0]->num();

  kmeans_distance_kernel<Dtype, KmeansLossLayer<Dtype>::TILE_DIM><<<grid, block>>>(gpu_input, gpu_weight, gpu_dist, M, N_, K_);
  CUDA_POST_KERNEL_CHECK;

  //std::cout << "dist: " << distance_.stat_data() << std::endl;

	const Dtype *cpu_distance = distance_.cpu_data();
  const Dtype *cpu_label = bottom[1]->cpu_data();
	Dtype *cpu_mask = mask_.mutable_cpu_data();
	Dtype *cpu_min_distance = (*top)[3]->mutable_cpu_data();
  Dtype *cpu_ind = (*top)[2]->mutable_cpu_data();
  Dtype *cpu_count = count_.mutable_cpu_data();

  Dtype beta = this->layer_param_.kmeans_loss_param().beta();

	Dtype loss = Dtype(0);
  Dtype acc = Dtype(0);
  
  caffe_memset(bottom[0]->num()*N_*sizeof(Dtype), Dtype(0), cpu_mask);
  caffe_memset(N_*sizeof(Dtype), Dtype(0), cpu_count);
  for (int i = 0; i < bottom[0]->num(); i++) {
    Dtype min_dist = cpu_distance[i*N_];
    int min_j = 0;
    for (int j = 0; j < N_; j++) {
      if (cpu_distance[i*N_ + j] < min_dist) {
        min_dist = cpu_distance[i*N_ + j];
        min_j = j;
      }
    }
    acc += min_j == (int)cpu_label[i];
    min_j = (int)cpu_label[i];
    min_dist = cpu_distance[i*N_ + min_j];

    cpu_mask[i*N_ + min_j] = Dtype(1.0);
    cpu_count[min_j] += Dtype(1.0);
    cpu_min_distance[i] = min_dist;
    cpu_ind[i] = min_j;

    loss += min_dist;
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
  Sb /= Dtype(2);

  std:: cout << "pos: " << loss/N_/bottom[0]->num()/Dtype(2) 
             << "\tneg: " << lambda_ * std::max(beta - Sb/N_/bottom[0]->num(), Dtype(0)) 
             << std::endl;
  loss = loss/N_/bottom[0]->num()/Dtype(2) + lambda_ * std::max(beta - Sb/N_/bottom[0]->num(), Dtype(0));
  sign_ = beta - Sb/N_/bottom[0]->num() > Dtype(0);

  (*top)[0]->mutable_cpu_data()[0] = loss;
  (*top)[1]->mutable_cpu_data()[0] = acc/bottom[0]->num();

  //std::cout << "loss: " << loss << std::endl;
  //std::cout << "acc: " << acc/bottom[0]->num() << std::endl;
  //std::cout << "count: " << count_.stat_data() << std::endl;
}

template <typename Dtype>
void KmeansLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  Dtype weight = top[0]->cpu_diff()[0]/N_/(*bottom)[0]->num();		
  caffe_copy((*bottom)[0]->count(), (*bottom)[0]->gpu_data(), (*bottom)[0]->mutable_gpu_diff());
  caffe_gpu_gemm(CblasNoTrans, CblasTrans, (*bottom)[0]->num(), K_, N_, -weight,
    mask_.gpu_data(), this->blobs_[0]->gpu_data(), weight, (*bottom)[0]->mutable_gpu_diff());

  caffe_gpu_gemm(CblasTrans, CblasNoTrans, K_, N_, (*bottom)[0]->num(), Dtype(-1),
    (*bottom)[0]->gpu_diff(), mask_.gpu_data(), Dtype(0), this->blobs_[0]->mutable_gpu_diff());

  if (sign_) {
    caffe_gpu_axpy((*bottom)[0]->count(), -lambda_*weight, diff_.gpu_data(), (*bottom)[0]->mutable_gpu_diff());
  }

}

INSTANTIATE_CLASS(KmeansLossLayer);

}  // namespace caffe
