#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>

#include <THC/THCDeviceUtils.cuh>
#include <THC/THCTensorMathReduce.cuh>
#include <THC/THCTensorSort.cuh>
#include <THC/THCThrustAllocator.cuh>

#include <thrust/execution_policy.h>
#include <thrust/unique.h>

namespace at { namespace native {

#ifdef __HIP_PLATFORM_HCC__
static const int WARP_SIZE = 64;
#else
static const int WARP_SIZE = 32;
#endif

// TODO: this is cut&paste
template <typename scalar_t>
__global__ void embedding_backward_kernel(
    int64_t* input, int64_t* indices, scalar_t* grad_output, scalar_t* grad_weight,
    int64_t numel, int64_t stride, int padding_idx) {

  using accscalar_t = acc_type<scalar_t, true>;
  int idx = blockIdx.x * 4 + threadIdx.y;

  // Each warp is responsible for an input into the LookupTable.
  // If the preceding input has the same as this input, then the warp
  // exits immediately. The warp also processes subsequent inputs with the
  // same value.
  //
  // Input Warp
  // 1     <warp 1>
  // 1     <warp 1> (<warp 2> exits without doing any work)
  // 5     <warp 3>
  // 8     <warp 4>

  // Number of values proceessed by each thread (grain size)
  const int SZ = 4;

  if (idx < numel
      && (idx == 0 || input[idx] != input[idx - 1])
      && input[idx] != padding_idx) {
    do {
      const int start_feature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
      const int weight_row = ((int) input[idx]) * stride;
      const int grad_row = ((int) indices[idx]) * stride;

      accscalar_t gradient[SZ];
      accscalar_t weight[SZ];

#pragma unroll
      for (int ii = 0; ii < SZ; ii++) {
        int feature_dim = start_feature + ii * WARP_SIZE;
        if (feature_dim < stride) {
          gradient[ii] = static_cast<accscalar_t>(grad_output[grad_row + feature_dim]);
          weight[ii] = static_cast<accscalar_t>(grad_weight[weight_row + feature_dim]);
        }
      }

#pragma unroll
      for (int ii = 0; ii < SZ; ii++) {
        weight[ii] += gradient[ii];
      }

#pragma unroll
      for (int ii = 0; ii < SZ; ii++) {
        int feature_dim = start_feature + ii * WARP_SIZE;
        if (feature_dim < stride) {
          grad_weight[weight_row + feature_dim] = static_cast<scalar_t>(weight[ii]);
        }
      }

      idx++;
    } while (idx < numel && input[idx] == input[idx - 1]);
  }
}


Tensor & index_put_cuda_(Tensor & self, TensorList indices_, const Tensor & values, bool) {
  auto grad_arg = TensorArg(values, "grad", 1);
  auto indices_arg = TensorArg(indices_[0], "indices", 1);
//  checkScalarType("index_put_cuda_", indices_arg, kLong);
  checkSameGPU("index_put_cuda_", grad_arg, indices_arg);

  int64_t num_weights = self.size(0);
  int64_t padding_idx = -1L;

  std::cerr << "indices_[0]" << std::endl;
  print(std::cerr, indices_[0], 120);
  std::cerr << indices_[0].sizes() << std::endl << std::endl;

  auto indices = indices_[0].toType(kByte);  // TODO

  std::cerr << "indices" << std::endl;
  print(std::cerr, indices, 120);
  std::cerr << indices.sizes() << std::endl << std::endl;

  auto num_indices = indices.numel();
  auto grad = values.contiguous().view({num_indices, values.size(-1)});

  int64_t stride = self.stride(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto sorted_indices = at::empty_like(indices);
  auto orig_indices = at::empty_like(indices);
  using device_ptr = thrust::device_ptr<int64_t>;

  // Sort the inputs into sorted with the corresponding indices; we
  // don't need a stable or multidimensional sort, so just use Thrust
  // directly
  {
    sorted_indices.copy_(indices);

    auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
    auto policy = thrust::cuda::par(allocator).on(stream);

    // Fill sortedOrigIndices with sequential indices
    auto count_iter = thrust::counting_iterator<int64_t>(0);
    auto orig_data = device_ptr(orig_indices.data<int64_t>());
    thrust::copy(policy, count_iter, count_iter + num_indices, orig_data);

    // Sort; a stable sort is not required
    auto sorted_data = device_ptr(sorted_indices.data<int64_t>());
    thrust::sort_by_key(policy, sorted_data, sorted_data + num_indices, orig_data,
        ThrustLTOp<int64_t>());
  }

  dim3 grid(THCCeilDiv(num_indices, (int64_t) 4), THCCeilDiv(stride, (int64_t) 128));
  dim3 block(32, 4);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.type(), "index_put_cuda_", [&] {
    embedding_backward_kernel<<<grid, block, 0, stream>>>(
      sorted_indices.data<int64_t>(),
      orig_indices.data<int64_t>(),
      grad.data<scalar_t>(),
      self.data<scalar_t>(),
      num_indices,
      stride,
      padding_idx);
  });
  THCudaCheck(cudaGetLastError());
  return self;
}

}}  // namespace at::native
