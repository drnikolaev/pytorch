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

namespace {

#ifdef __HIP_PLATFORM_HCC__
static const int WARP_SIZE = 64;
static const int BLOCKDIMY = 16;
#else
static const int WARP_SIZE = 32;
static const int BLOCKDIMY = 32;
#endif


template <typename scalar_t>
__global__ void embedding_backward_kernel(
    int64_t* input, int64_t* indices, scalar_t* grad_output, scalar_t* grad_weight,
    int64_t* count, int64_t numel, int64_t stride, int padding_idx) {

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
      const accscalar_t scale = count ? (accscalar_t)1.0 / count[idx] : 1.0;

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
        weight[ii] += gradient[ii] * scale;
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


Tensor embedding_dense_backward_cuda(const Tensor & grad_, const Tensor & indices,
                                     int64_t num_weights, int64_t padding_idx,
                                     bool scale_grad_by_freq) {
  auto grad_arg = TensorArg(grad_, "grad", 1);
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarType("embedding_backward", indices_arg, kLong);
  checkSameGPU("embedding_backward", grad_arg, indices_arg);

  std::cerr << "grad_" << std::endl;
  print(std::cerr, grad_, 120);
  std::cerr << grad_.sizes() << std::endl << std::endl;

  auto num_indices = indices.numel();
  auto grad = grad_.contiguous().view({num_indices, grad_.size(-1)});
  auto grad_weight = at::zeros({num_weights, grad_.size(-1)}, grad_.options());

  std::cerr << "grad.cnt" << std::endl;
  print(std::cerr, grad, 120);
  std::cerr << grad.sizes() << std::endl << std::endl;

  int64_t stride = grad_weight.stride(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (num_indices <= 768 && !scale_grad_by_freq) {

    std::cerr << "indices" << std::endl;
    print(std::cerr, indices, 120);
    std::cerr << indices.sizes() << std::endl << std::endl;

    auto indices_contig = indices.contiguous();

    std::cerr << "indices_contig" << std::endl;
    print(std::cerr, indices_contig, 120);
    std::cerr << indices_contig.sizes() << std::endl << std::endl;

    dim3 grid(THCCeilDiv(stride, (int64_t)WARP_SIZE));
    dim3 block(WARP_SIZE, BLOCKDIMY);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF
    (grad.type(),
     "embedding_backward",
     [&]
     {
       using accscalar_t = acc_type<scalar_t, true>;
       embedding_backward_feature_kernel<scalar_t, accscalar_t>
       <<<grid,
         block,
         sizeof(accscalar_t)*WARP_SIZE*BLOCKDIMY + sizeof(int)*WARP_SIZE*BLOCKDIMY,
         stream>>>
       (indices_contig.data<int64_t>(),
           grad.data<scalar_t>(),
           grad_weight.data<scalar_t>(),
           static_cast<int>(num_indices),
           static_cast<int64_t>(stride),
           static_cast<int>(padding_idx));
     });

    THCudaCheck(cudaGetLastError());

    std::cerr << "->grad_weight" << std::endl;
    print(std::cerr, grad_weight, 120);
    std::cerr << grad_weight.sizes() << std::endl << std::endl;

    return grad_weight;
  }

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

    std::cerr << "sorted_indices" << std::endl;
    print(std::cerr, sorted_indices, 120);
    std::cerr << sorted_indices.sizes() << std::endl << std::endl;

    std::cerr << "orig_indices" << std::endl;
    print(std::cerr, orig_indices, 120);
    std::cerr << orig_indices.sizes() << std::endl << std::endl;


  }

  Tensor count;
  if (scale_grad_by_freq) {
    count = at::empty_like(indices);

    auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
    auto policy = thrust::cuda::par(allocator).on(stream);

    // Compute an increasing sequence per unique item in sortedIndices:
    // sorted: 2 5 5 5 7 7 8 9 9
    //  count: 1 1 2 3 1 2 1 1 2
    auto sorted_data = device_ptr(sorted_indices.data<int64_t>());
    auto count_data = device_ptr(count.data<int64_t>());
    thrust::inclusive_scan_by_key(
        policy,
        sorted_data,
        sorted_data + num_indices,
        thrust::make_constant_iterator(1),
        count_data
    );

    // Take the maximum of each count per unique key in reverse:
    // sorted: 2 5 5 5 7 7 8 9 9
    //  count: 1 3 3 3 2 2 1 2 2
    thrust::inclusive_scan_by_key(
        policy,
        thrust::make_reverse_iterator(sorted_data + num_indices),
        thrust::make_reverse_iterator(sorted_data),
        thrust::make_reverse_iterator(count_data + num_indices),
        thrust::make_reverse_iterator(count_data + num_indices),
        thrust::equal_to<int64_t>(),
        thrust::maximum<int64_t>()
    );
  }

  dim3 grid(THCCeilDiv(num_indices, (int64_t) 4), THCCeilDiv(stride, (int64_t) 128));
  dim3 block(32, 4);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.type(), "embedding_backward", [&] {
    embedding_backward_kernel<<<grid, block, 0, stream>>>(
      sorted_indices.data<int64_t>(),
          orig_indices.data<int64_t>(),
          grad.data<scalar_t>(),
          grad_weight.data<scalar_t>(),
          count.defined() ? count.data<int64_t>() : nullptr,
          num_indices,
          stride,
          padding_idx);
  });
  THCudaCheck(cudaGetLastError());

  std::cerr << "->grad_weight" << std::endl;
  print(std::cerr, grad_weight, 120);
  std::cerr << grad_weight.sizes() << std::endl << std::endl;

  return grad_weight;
}



}}  // namespace at::native

