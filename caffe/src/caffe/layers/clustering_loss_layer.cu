#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype, int TILE_DIM>
__global__ void cluster_distance_kernel(const Dtype *x, const Dtype *y, Dtype *z, int m, int n, int p) {
    __shared__ Dtype sx[TILE_DIM][TILE_DIM+1];
    __shared__ Dtype sy[TILE_DIM][TILE_DIM+1];

    const int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    const int i = bx*TILE_DIM + tx;
    const int j = by*TILE_DIM + ty;

    Dtype c = 0.0;
    for (int k = 0; k < p; k += TILE_DIM) {
        sx[ty][tx] = x[j*p + k + tx];
        sy[ty][tx] = y[(k+ty)*n + i];
        __syncthreads();

        for (int kk = 0; kk < TILE_DIM; kk++) {
            c += (sx[ty][kk] - sy[kk][tx])*(sx[ty][kk] - sy[kk][tx]);
        }
        __syncthreads();
    }
    z[i+j*n] = sqrt(c/p);
}

/*
template<typename Dtype>
__global__ void coef_kernel(int m, int n, const Dtype *dist, const Dtype *y, const Dtype *margin, Dtype *coef, Dtype *loss, Dtype lambda) {
  CUDA_KERNEL_LOOP(index, m*n) {
    int i = index/n;
    int j = index%n;
    Dtype d = dist[index]/margin[j];
    Dtype sign = d < Dtype(1);
    coef[index] = ( lambda * y[i] - (Dtype(1)-lambda)*(Dtype(1)-y) ) * sign / (margin[j]*dist[index]*m*n);
    loss[index] = ( lambda * y[i] * min(d, Dtype(1)) + (Dtype(1)-lambda)*(Dtype(1)-y[i])*max(Dtype(1)-d, Dtype(0)) )/Dtype(m*n);
  } 
}
*/

template <typename Dtype>
void ClusteringLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  dim3 grid(N_/TILE_DIM, bottom[0]->num()/TILE_DIM, 1);
  dim3 block(TILE_DIM, TILE_DIM, 1);
  cluster_distance_kernel<Dtype, ClusteringLossLayer<Dtype>::TILE_DIM><<<grid, block>>>(bottom[0]->gpu_data(),
   this->blobs_[0]->gpu_data(), distance_.mutable_gpu_data(), bottom[0]->num(), N_, K_);
  CUDA_POST_KERNEL_CHECK;

	//std::cout << "distance: " << distance_.asum_data()/distance_.count() << std::endl;

	const Dtype *cpu_distance = distance_.cpu_data();
	Dtype *cpu_mask = mask_.mutable_cpu_data();
	Dtype *cpu_min_distance = (*top)[3]->mutable_cpu_data();
	const Dtype *cpu_label = bottom[1]->cpu_data();
  Dtype *cpu_margin = this->blobs_[1]->mutable_cpu_data();
	Dtype *cpu_coefm = coefm_.mutable_cpu_data();
  Dtype *cpu_coefn = coefn_.mutable_cpu_data();
  Dtype *cpu_ind = (*top)[2]->mutable_cpu_data();
  Dtype *cpu_count = count_.mutable_cpu_data();
  Dtype *cpu_pos_count = pos_count_.mutable_cpu_data();
  Dtype *cpu_coef_margin = coef_margin_.mutable_cpu_data();

  Dtype beta = this->layer_param_.clustering_loss_param().beta();

	Dtype loss = Dtype(0);
  caffe_memset(N_*sizeof(Dtype), 0, cpu_count);
  caffe_memset(N_*sizeof(Dtype), 0, cpu_coefn);
  caffe_memset(N_*sizeof(Dtype), 0, cpu_coef_margin);

  for (int j = 0; j < N_; ++j) if (cpu_margin[j] < Dtype(1.0)) cpu_margin[j] = Dtype(1.0);

  for (int i = 0; i < bottom[0]->num(); ++i) {
    int y = (int)cpu_label[i];
    cpu_pos_count[i] = Dtype(1e-10);
    if (y == 0) {
      for (int j = 0; j < N_; ++j) {
        Dtype d = cpu_distance[i*N_ + j]/cpu_margin[j];
        if (d < Dtype(1)) {
          cpu_count[j] += Dtype(1)-d;
        }
      }
    }else {
      for (int j = 0; j < N_; ++j) {
        Dtype d = cpu_distance[i*N_ + j]/cpu_margin[j];
        cpu_pos_count[i] += Dtype(1)-std::min(d, Dtype(1));
      }
    }
  }
  for (int j = 0; j < N_; ++j) cpu_count[j] = cpu_count[j]/bottom[0]->num()/beta - Dtype(1);

	for (int i = 0; i < bottom[0]->num(); ++i) {
    int y = (int)cpu_label[i];
    cpu_coefm[i] = Dtype(0);
		for (int j = 0; j < N_; ++j) {
      Dtype d = cpu_distance[i*N_ + j]/cpu_margin[j];
		  Dtype sign = d < Dtype(1);	
      cpu_ind[i*N_ + j] = sign*(Dtype(1)-y);
      cpu_min_distance[i*N_ + j] = d;
      if (y == 1) {
        cpu_mask[i*N_ + j] = (Dtype(1)-std::min(d, Dtype(1)))/cpu_pos_count[i] * lambda_*sign / (cpu_distance[i*N_ + j]*cpu_margin[j]);
        loss += (Dtype(1)-std::min(d, Dtype(1)))/cpu_pos_count[i] * lambda_ * y * std::min(d, Dtype(1));
        cpu_coef_margin[j] += -(Dtype(1)-std::min(d, Dtype(1)))/cpu_pos_count[i] * lambda_*sign*d/cpu_margin[j];
      }else {
        cpu_mask[i*N_ + j] = - (Dtype(1)-lambda_)*cpu_count[j]*sign / (beta*cpu_distance[i*N_ + j]*cpu_margin[j]);
        cpu_coef_margin[j] += (Dtype(1)-lambda_)*sign*cpu_count[j]*d/(beta*cpu_margin[j]);
      }
      cpu_coefm[i] += cpu_mask[i*N_ + j];
      cpu_coefn[j] += cpu_mask[i*N_ + j];
		}
  }

  loss /= bottom[0]->num();
  std::cout << "pos: " << loss;
  Dtype tt = 0.0;

  for (int j = 0; j < N_; ++j) {
    tt += (Dtype(1)-lambda_)*cpu_count[j]*cpu_count[j]/Dtype(2);
  }
  std::cout << " neg: " << tt << std::endl;
  loss += tt;
	(*top)[0]->mutable_cpu_data()[0] = loss;

  Dtype count_std;
  count_.stat_data(NULL, NULL, NULL, &count_std);
  (*top)[1]->mutable_cpu_data()[0] = count_std;


  std::cout << "margin: " << this->blobs_[1]->stat_data() << std::endl;
  std::cout << "count: " << count_.stat_data() << std::endl;
  std::cout << "dist: " << distance_.stat_data() << std::endl;
  std::cout << "dist/m: " << (*top)[3]->stat_data() << std::endl;
  std::cout << "mask: " << mask_.stat_data() << std::endl;
  std::cout << "ind: " << (*top)[2]->stat_data() << std::endl;
  std::cout << "K: " << K_ << std::endl;
  //std::cout << "margin_diff: " << coef_margin_.stat_data() << std::endl;

}

template <typename Dtype>
void ClusteringLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  Dtype weight = top[0]->cpu_diff()[0]/K_/(*bottom)[0]->num();		
  if (propagate_down[0]) {
    caffe_gpu_dgmm(CblasLeft, (*bottom)[0]->num(), K_, (*bottom)[0]->gpu_data(),
      coefm_.gpu_data(), (*bottom)[0]->mutable_gpu_diff());
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, mask_.num(), K_, N_, -weight, 
      mask_.gpu_data(), this->blobs_[0]->gpu_data(), weight, (*bottom)[0]->mutable_gpu_diff());
    LOG(INFO) << "bottom diff:" << (*bottom)[0]->stat_diff();
  }
  
  if (this->param_propagate_down_[0]) {
    caffe_gpu_dgmm(CblasRight, K_, N_, this->blobs_[0]->gpu_data(), coefn_.gpu_data(), this->blobs_[0]->mutable_gpu_diff());
  	caffe_gpu_gemm(CblasTrans, CblasNoTrans, K_, N_, mask_.num(), -weight, 
  		(*bottom)[0]->gpu_data(), mask_.gpu_data(), weight, this->blobs_[0]->mutable_gpu_diff());
    //std::cout << "diff: " <<  this->blobs_[0]->stat_diff() << std::endl;
    LOG(INFO) << "center diff:" << this->blobs_[0]->stat_diff();
  }
  
  if (this->param_propagate_down_[1]) {
    caffe_copy(coef_margin_.count(), coef_margin_.cpu_data(), this->blobs_[1]->mutable_gpu_diff());
    caffe_gpu_scal(coef_margin_.count(), top[0]->cpu_diff()[0]/(*bottom)[0]->num(), this->blobs_[1]->mutable_gpu_diff());
    LOG(INFO) << "center diff:" << this->blobs_[1]->stat_diff();
  } 
}

INSTANTIATE_CLASS(ClusteringLossLayer);

}  // namespace caffe
