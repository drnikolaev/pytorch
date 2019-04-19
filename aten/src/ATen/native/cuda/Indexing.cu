// Indexing tensors by by tensors
//
// This corresponds to "advanced indexing" in NumPy. The two operations are:
//
//  index(Tensor self, indices) -> Tensor
//  index_put_(Tensor self, indices, value, accumulate=false)
//
// The index is a TensorList containg kLong or kByte tensors or nulls. Byte
// tensors (boolean masks) are expanded to long tensors via nonzero(). Null
// tensors signify that the dimension is not indexed.
//
// All indexes are broadcast together and iterated as *one*. From NumPy:
//
// result[i_1, ..., i_M] == x[ind_1[i_1, ..., i_M], ind_2[i_1, ..., i_M],
//                           ..., ind_N[i_1, ..., i_M]]
//
// Note 1: ByteTensors expand to index as many dimensions as there are in the
// mask.
//
// Note 2: The behavior is more complicated when the index tensors are not all
// adjacent (e.g. x[[0, 1], :, [2, 3]]). In this case, self and the index
// tensors are transposed to the front: x.transpose(1, 2)[[0, 1], [2, 3]]
//
// The code contains two implementations of indexing. The more efficient
// implementation treats indexing like an elementwise operation over the
// tensors `result`, `x`, `ind_1`, `ind_2`, etc. This implementation does
// not work for index_put_ with accumulate=True. The other implementation
// combines the indexed tensors into a single linear index that is used
// with Tensor.put_. This is used for index_put_ with accumulate=True.
//
// The more efficient implementation takes the following steps for the
// above operation:
//
// 1) Broadcast ind_1, ind_2, ind_3 together to a common shape
// 2) Record x.stride(i) for each indexed dimension `i`
// 3) Replace the indexed subspace of `x` with the shape of the corresponding
//    subspace of `result` but with stride 0
// 4) Add dimensions of size 1 to the index tensors (ind_1, ind_2, etc.) so
//    that their shape is compatible with the result shape
//
// The CPU or CUDA kernel then computes element-wise over the broadcasted
// and restrided result, x, ind_1,  ind_2, etc.:
//
//   result[...] = *(&x[...] +
//                   ind_1[...] * x.stride(1) +
//                   ind_2[...] * x.stride(2) +
//                   ...)
//
// where & and * represent the C-style address-of and indirection operations.

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/TensorUtils.h>
#include <ATen/ExpandUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/core/ScalarType.h>

#include <ATen/native/Indexing.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctions.h>

#include <THC/THCDeviceUtils.cuh>
#include <THC/THCTensorMathReduce.cuh>
#include <THC/THCTensorSort.cuh>
#include <THC/THCThrustAllocator.cuh>

#include <vector_types.h>

#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <torch/csrc/utils/tensor_flatten.h>

#include <ATen/cpu/vec256/vec256.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

//#ifdef __HIP_PLATFORM_HCC__
//#define WARP_SIZE 64
//#else
//#define WARP_SIZE 32
//#endif
#define GROUP_SIZE 8L

namespace at { namespace native {

static inline unsigned int nextP2(unsigned int v) {
  v--;
  v |= v >> 1U;
  v |= v >> 2U;
  v |= v >> 4U;
  v |= v >> 8U;
  v |= v >> 16U;
  v++;
  return v;
}

[[noreturn]]
static void invalid_mask(const Tensor & self, int64_t idx, const Tensor & mask, int64_t maskIdx) {
  std::stringstream ss;
  ss << "The shape of the mask " << mask.sizes() << " at index " << maskIdx;
  ss << " does not match the shape of the indexed tensor " << self.sizes();
  ss << " at index " << idx;
  AT_INDEX_ERROR(ss.str());
}

static void checkIndexTensorTypes(TensorList indices) {
  for (auto& tensor : indices) {
    if (tensor.defined()) {
      auto scalarType = tensor.scalar_type();
      if (scalarType != kLong && scalarType != kByte) {
        AT_INDEX_ERROR("tensors used as indices must be long or byte tensors");
      }
    }
  }
}

static std::vector<Tensor> expandByteTensors(const Tensor & self, TensorList indices) {
  // Expands byte tensors (masks) into the equivalent indexing by LongTensors
  std::vector<Tensor> result;
  for (auto & index : indices) {
    if (index.scalar_type() == kByte) {
      // The sizes of the ByteTensor mask must match the sizes of the
      // corresponding dimensions in self
      for (int64_t j = 0; j < index.dim(); j++) {
        int64_t srcIdx = result.size() + j;
        if (index.size(j) != self.size(srcIdx)) {
          invalid_mask(self, srcIdx, index, j);
        }
      }
      // Replace with nonzeros
      auto nonzero = index.nonzero();
      auto special_empty = false;
      for (int64_t j = 0; j < index.dim(); j++) {
        if (special_empty) {
          // We can't call select on an empty tensor so we just create an empty
          // tensor.
          result.emplace_back(at::empty({0}, nonzero.options()));
        } else {
          result.emplace_back(nonzero.select(1, j));
        }
      }
    } else {
      result.emplace_back(index);
    }
  }
  return result;
}

static bool hasContiguousSubspace(TensorList tl) {
  // true if all the non-null tensors are adjacent
  auto isDefined = [](const Tensor & tensor){ return tensor.defined(); };
  auto isNull = [](const Tensor & tensor){ return !tensor.defined(); };
  auto start = std::find_if(tl.begin(), tl.end(), isDefined);
  auto stop = std::find_if(tl.rbegin(), tl.rend(), isDefined);
  auto it = std::find_if(start, stop.base(), isNull);
  return it == stop.base();
}

// Transposes the tensor and indices together so that all the non-null indices
// index the first k dimensions of the tensor. Returns the transposed tensor
// and the reordered indices. For example:
//  transposeToFront(tensor, {nullptr, a, nullptr, b})
// returns
//  tensor.permute([1, 3, 0, 2]), {a, b, nullptr, nullptr}
static std::tuple<Tensor, std::vector<Tensor>>
transposeToFront(Tensor self, TensorList indices) {
  std::vector<int64_t> dims;
  std::vector<Tensor> transposedIndices;
  dims.reserve(self.dim());
  for (int64_t i = 0; i < self.dim(); i++) {
    if (indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back(indices[i]);
    }
  }
  for (int64_t i = 0; i < self.dim(); i++) {
    if (!indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back();
    }
  }
  return std::make_tuple(self.permute(dims), std::move(transposedIndices));
}

static std::vector<int64_t> computeLinearStride(const Tensor & tensor) {
  // computes the stride as if tensor were contigous
  auto sizes = tensor.sizes();
  std::vector<int64_t> stride(tensor.dim());
  stride[tensor.dim() - 1] = 1;
  std::partial_sum(sizes.rbegin(), sizes.rend() - 1, stride.rbegin() + 1, std::multiplies<int64_t>());
  return stride;
}

// Unsqueezes src `before` times at the front and `after` times at the end
static Tensor unsqueezeN(const Tensor & src, int64_t before, int64_t after) {
  auto srcSizes = src.sizes();
  auto nDim = src.dim();
  std::vector<int64_t> sizes(nDim + before + after, 1);
  for (int64_t i = 0; i < nDim; i++) {
    sizes[i + before] = srcSizes[i];
  }
  return src.view(sizes);
}

static std::tuple<Tensor, Tensor, Tensor, Tensor, int64_t, int64_t, int64_t, int64_t>
    makeLinearIndex(Tensor self, TensorList orig) {
  checkIndexTensorTypes(orig);
  // first expand ByteTensor (boolean masks) into 1 or more LongTensors
  auto indices = expandByteTensors(self, orig);
  // next broadcast all index tensors together
  indices = expand_outplace(indices);
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < (size_t)self.dim()) {
    indices.emplace_back();
  }
  // if the non-null indices are not all adjacent, transpose self and indices
  // together so that they're adjacent at the front
  if (!hasContiguousSubspace(indices)) {
    std::tie(self, indices) = transposeToFront(self, indices);
  }

  auto strides = computeLinearStride(self);

  // Compute the linear index by multiplying the indexing tensors by the
  // stride and summing them. All the indexing tensors have the same shape at
  // this point. We also compute the number of dimensions before and after that
  // are not being index.
  Tensor linearIndex;
  int64_t emptyBefore = 0, emptyAfter = 0, nElemBefore = 1, nElemAfter = 1;
  for (int64_t i = 0; i < self.dim(); i++) {
    if (indices[i].defined()) {
      if (linearIndex.defined()) {
        linearIndex += indices[i].remainder(self.size(i)) * strides[i];
      } else {
        linearIndex = indices[i].remainder(self.size(i)) * strides[i];
      }
    }
    else if (linearIndex.defined()) {
      emptyAfter++;
      nElemAfter *= self.size(i);
    } else {
      emptyBefore++;
      nElemBefore *= self.size(i);
    }
  }
  // Compute the linear indices for the parts of the tensor not being indexed
  // ...and not being sorted
  Tensor beforeIndex;
  if (emptyBefore > 0) {
    beforeIndex = at::arange(0, nElemBefore, self.options().dtype(kLong)) * strides[emptyBefore - 1];
    beforeIndex = beforeIndex.view(self.sizes().slice(0, emptyBefore));
    beforeIndex = unsqueezeN(beforeIndex, 0, linearIndex.dim() + emptyAfter);
  }
  Tensor afterIndex;
  if (emptyAfter > 0) {
    afterIndex = at::arange(0, nElemAfter, self.options().dtype(kLong));
    afterIndex = afterIndex.view(self.sizes().slice(self.dim() - emptyAfter, emptyAfter));
    afterIndex = unsqueezeN(afterIndex, linearIndex.dim() + emptyBefore, 0);
  }
  return std::make_tuple(self, linearIndex, beforeIndex, afterIndex,
                         emptyBefore, emptyAfter, nElemBefore, nElemAfter);
}

template <typename T, typename IndexType>
__device__ __forceinline__ IndexType indexToOffset(
    const at::cuda::detail::TensorInfo<T, IndexType>& info, IndexType linearIndex) {
  IndexType offset(0);
  for (int i = info.dims - 1; i > 0; --i) {
    offset += (linearIndex % info.sizes[i]) * info.strides[i];
    linearIndex /= info.sizes[i];
  }
  return offset + linearIndex * info.strides[0];
}

template <typename T, typename IndexType>
__device__ __forceinline__ IndexType indexToOffset(
    int dims,
    int64_t sizes[MAX_TENSORINFO_DIMS],
    int64_t strides[MAX_TENSORINFO_DIMS],
    IndexType linearIndex) {
  IndexType offset(0);
  for (int i = dims - 1; i > 0; --i) {
    offset += (linearIndex % sizes[i]) * strides[i];
    linearIndex /= sizes[i];
  }
  return offset + linearIndex * strides[0];
}


template <typename scalar_t>
__global__ void backward_indexing_kernel(
    int64_t* extendedIdx, int64_t* origOrder, scalar_t* gradValues,
    int64_t extendedIdxSize,
    int64_t dstSize, int64_t sortedStride, int64_t sortedSize,
    scalar_t* dstData, int dstDims,
    int64_t* dstSizes, int64_t* dstStrides) {
  using accscalar_t = acc_type<scalar_t, true>;
  thread_block block = this_thread_block();
  thread_block_tile<GROUP_SIZE> tile = tiled_partition<GROUP_SIZE>(block);
  int th = tile.thread_rank();
  int idx = block.thread_rank() / GROUP_SIZE;

  if (idx < extendedIdxSize &&
      (idx == 0 || extendedIdx[idx] != extendedIdx[idx - 1])) {
    do {
      const int64_t n = idx * GROUP_SIZE + th;
      int64_t offsetArr[GROUP_SIZE];
      accscalar_t valArr[GROUP_SIZE];
      if (n < extendedIdxSize) {
        const int64_t sortedno = n / sortedStride;
        const int64_t unsorted = origOrder[sortedno % sortedSize];
        const int64_t no = unsorted * sortedStride + (n % sortedStride);
        const scalar_t* pvalue = gradValues + no;
        const int64_t* pindex = extendedIdx + no;
        const int64_t linear_index = *pindex;
        offsetArr[th] =
        indexToOffset < scalar_t, int64_t > (dstDims, dstSizes, dstStrides, linear_index);
        valArr[th] = *pvalue;
      }
      tile.sync();
      if (tile.thread_rank() == 0) {
        #pragma unroll
        for (int t = 0; t < GROUP_SIZE; ++t) {
          dstData[offsetArr[t]] += valArr[t];
        }
      }
      ++idx;
    } while (idx < extendedIdxSize && extendedIdx[idx] == extendedIdx[idx - 1]);
  }
}

  //  printf("**** %lld %lld %lld %g  --  %lld %lld %lld %lld +++ %lld\n", n , no, *pindex, *pvalue,
  //    sortedStride, sortedSize, extendedIdxSize, dstSize, start_feature);

  //    if (pindex - (std::ptrdiff_t) extendedIdxSize >= 0) break;
  /// printf("...%lld %lld %lld %lld %lld %g\n", n , no, *pindex, offset, linear_index, info.data[offset]);

  // struct cudaLaunchParams {
  //  void *func;
  //  dim3 gridDim;
  //  dim3 blockDim;
  //  void **args;
  //  size_t sharedMem;
  //  cudaStream_t stream;
  //};

  // struct cudaLaunchParams* params = paramsList+i;
  // CUDACHECK(cudaSetDevice(cudaDevs[i]));
  // CUDACHECK(cudaLaunchKernel(params->func, params->gridDim, params->blockDim, params->args, params->sharedMem, params->stream));

  // kernel<type> <<<grid, block,shared,stream>>>(arg);
  // cudaLaunchKernel(kernel<type>, grid.x, grid.y, grid.z, block.x, block.y, block.z, arg, shared, stream);

  /*
  dim3 grid(THCCeilDiv(num_indices, (int64_t) 4), THCCeilDiv(stride, (int64_t)
  128)); dim3 block(32, 4);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.scalar_type(), "embedding_backward",
  [&] { embedding_backward_kernel<<<grid, block, 0, stream>>>(
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
  */

  // unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
  // unsigned int warpid = tid / warpSize;

  // template <class F, class... Args>
  // void for_each_argument(F f, Args&&... args) { [](...){}((f(std::forward<Args>(args)), 0)...); }

  //
  // template <typename... Params>
  // void bar(Params... params)
  //{
  //  /* etc etc */
  //  void* arguments_ptrs[sizeof...(Params)];
  //  auto arg_index = 0;
  //
  //  for_each_argument(
  //      [&](auto param) {arguments_ptrs[arg_index++] = &param;},
  //      params...);
  //
  //  cudaLaunchKernel<decltype(my_kernel)>(
  //      &my_kernel, grid_dims, block_dims, argument_ptrs, shmem_size, stream_id);
  //}

  template <typename T>
  struct TensorAccumMixedPutOp : thrust::unary_function<int64_t, T> {
    TensorAccumMixedPutOp(
        at::cuda::detail::TensorInfo<T, int64_t> info,
        const T* psrc,
        const int64_t* ext_idx_beg,
        const int64_t* ext_idx_end,
        const int64_t* orig_idx_beg,
        int64_t sortedStride,
        int64_t sortedSize,
        bool before)
        : info(info),
          p_source(psrc),
          ext_idx_beg(ext_idx_beg),
          ext_idx_end(ext_idx_end),
          orig_idx_beg(orig_idx_beg),
          sortedStride(sortedStride),
          sortedSize(sortedSize),
          before(before) {}

    __device__ __forceinline__ T operator()(int64_t n) {
      const int64_t sortedno = n / sortedStride;
      const int64_t unsorted = orig_idx_beg[sortedno % sortedSize];
      const int64_t no = unsorted * sortedStride + (n % sortedStride);
      const T* pvalue = p_source + no;
      const int64_t* pindex = ext_idx_beg + no;
      const int64_t linear_index = *pindex;
      const int64_t offset = indexToOffset<T, int64_t>(info, linear_index);

      // TODO!!!
      atomicAdd(info.data + offset, *pvalue);

      /*
      ///    if (pindex == ext_idx_beg || *pindex != *(pindex - 1)) {
            do {
              info.data[offset] += *pvalue;
              pindex += sortedStride;
              pvalue += sortedStride;

              printf("...%lld %lld %lld %lld %lld %g\n", n , no, *pindex,
      offset, linear_index, info.data[offset]); } while (pindex != ext_idx_end
      && *pindex == linear_index);
      ///    }
       */
      return 0; // discarded
    }

    at::cuda::detail::TensorInfo<T, int64_t> info;
    const T* p_source;
    const int64_t* ext_idx_beg;
    const int64_t* ext_idx_end;
    const int64_t* orig_idx_beg;
    const int64_t sortedStride;
    const int64_t sortedSize;
    const bool before;
  };

  template <typename T>
  struct TensorAccumFullyIndexedPutOp : thrust::unary_function<int64_t, T> {
    TensorAccumFullyIndexedPutOp(
        at::cuda::detail::TensorInfo<T, int64_t> info,
        const T* psrc,
        const int64_t* sorted_idx_beg,
        const int64_t* sorted_idx_end,
        const int64_t* orig_idx_beg)
        : info(info),
          p_source(psrc),
          sorted_idx_beg(sorted_idx_beg),
          sorted_idx_end(sorted_idx_end),
          orig_idx_beg(orig_idx_beg) {}

    __device__ __forceinline__ T operator()(int64_t n) {
      const int64_t no = orig_idx_beg[n]; // restore if flipped
      const T* pvalue = p_source + no;
      const int64_t* pindex = sorted_idx_beg + no;
      const int64_t linear_index = *pindex;
      const int64_t offset = indexToOffset<T, int64_t>(info, linear_index);

      if (pindex == sorted_idx_beg || *pindex != *(pindex - 1)) {
        do {
          info.data[offset] += *pvalue;
          pindex++;
          pvalue++;
        } while (pindex != sorted_idx_end && *pindex == linear_index);
      }
      return 0; // discarded
    }

    const at::cuda::detail::TensorInfo<T, int64_t> info;
    const T* p_source;
    const int64_t* sorted_idx_beg;
    const int64_t* sorted_idx_end;
    const int64_t* orig_idx_beg;
  };




  Tensor& index_put_cuda_(
      Tensor & self_,
      TensorList indices,
      const Tensor& value,
      bool accumulate) {
    if (indices.size() > (size_t)self_.dim()) {
      AT_INDEX_ERROR(
          "too many indices for tensor of dimension ",
          self_.dim(),
          " (got ",
          indices.size(),
          ")");
    }

    Tensor self, linearIndex; //, expandedValue;
    Tensor beforeIndex, afterIndex;
    int64_t emptyBefore = 0L, emptyAfter = 0L;
    int64_t nElemBefore = 1L, nElemAfter = 1L;

    std::tie(
        self,
        linearIndex,
        beforeIndex,
        afterIndex,
        emptyBefore,
        emptyAfter,
        nElemBefore,
        nElemAfter) = makeLinearIndex(self_, indices);

    auto sortedLinearIndex = linearIndex.clone();
    auto origCounters = at::empty_like(linearIndex);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
    auto policy = thrust::cuda::par(allocator).on(stream);

    int64_t idxSize = linearIndex.numel(); // const breaks usin cudaKernelLaunch
    int64_t dstSize = self.numel();
    AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "xput_cuda_1", [&] {
      int64_t* sortedLinearIndex_beg = sortedLinearIndex.data<int64_t>();
      int64_t* sortedLinearIndex_end = sortedLinearIndex_beg + idxSize;
      int64_t* origCounters_beg = origCounters.data<int64_t>();
      auto sortedLinearIndex_iter =
          thrust::device_ptr<int64_t>(sortedLinearIndex_beg);
      auto origCounters_iter = thrust::device_ptr<int64_t>(origCounters_beg);
      auto self_info = cuda::detail::getTensorInfo<scalar_t, int64_t>(self);
      TensorAccumFullyIndexedPutOp<scalar_t> aiPutOp(
          self_info,
          value.data<scalar_t>(),
          sortedLinearIndex_beg,
          sortedLinearIndex_end,
          origCounters_beg);

      thrust::sequence(policy, origCounters_iter, origCounters_iter + idxSize);

      thrust::sort_by_key(
          policy,
          sortedLinearIndex_iter,
          sortedLinearIndex_iter + idxSize,
          origCounters_iter,
          ThrustLTOp<int64_t>());

      if (!beforeIndex.defined() && !afterIndex.defined()) {
        // Full size index, done:
        thrust::counting_iterator<int64_t> first(0);
        thrust::counting_iterator<int64_t> last(idxSize);
        thrust::for_each(policy, first, last, aiPutOp);
      }
    });

    if (beforeIndex.defined() || afterIndex.defined()) {
      // trying to reuse device memory
      Tensor extendedLinearIndex = sortedLinearIndex;
      // Sum with broadcasting to compute the full index
      // using unsorted original
      linearIndex = unsqueezeN(linearIndex, emptyBefore, emptyAfter);

      // here we prefer to run 2*N^2 times rather than N^3
      for (int side = 0; side < 2; ++side) {
        if (side == 0 && emptyBefore > 0) {
          beforeIndex =
              unsqueezeN(beforeIndex, 0, linearIndex.dim()); // + emptyAfter);
          extendedLinearIndex = linearIndex + beforeIndex;

//          std::cout << "beforeIndex" << std::endl;
//          print(beforeIndex, 120);
//          std::cout << std::endl
//                    << "strides: " << computeLinearStride(beforeIndex)
//                    << std::endl
//                    << std::endl;

        } else if (side == 1 && emptyAfter > 0) {
          afterIndex =
              unsqueezeN(afterIndex, linearIndex.dim() /* + emptyBefore*/, 0);
          extendedLinearIndex = linearIndex + afterIndex;

//          std::cout << "afterIndex" << std::endl;
//          print(afterIndex, 120);
//          std::cout << std::endl
//                    << "strides: " << computeLinearStride(afterIndex)
//                    << std::endl
//                    << std::endl;

        } else {
          continue;
        }
        extendedLinearIndex.squeeze_();

//        std::cout << "extendedLinearIndex" << std::endl;
//        print(extendedLinearIndex, 120);
//        std::cout << std::endl
//                  << "strides: " << computeLinearStride(extendedLinearIndex)
//                  << std::endl
//                  << std::endl;

        AT_DISPATCH_FLOATING_TYPES(
            self.scalar_type(), "index_put_cuda_kernel_", [&] {
              cuda::detail::TensorInfo<scalar_t, int64_t> self_info =
                  cuda::detail::getTensorInfo<scalar_t, int64_t>(self);
              scalar_t* valuePtr = value.data<scalar_t>();
              int64_t extendedIdxSize = extendedLinearIndex.numel();
              int64_t* origCountersPtr = origCounters.data<int64_t>();
              int64_t* extendedLinearIndexPtr =
                  extendedLinearIndex.data<int64_t>();
              int64_t sortedStride = extendedIdxSize / idxSize;
              thrust::device_vector<int64_t> vSizes(
                  self_info.sizes, self_info.sizes + self_info.dims);
              thrust::device_vector<int64_t> vStrides(
                  self_info.strides, self_info.strides + self_info.dims);
              int64_t *dstSizes = vSizes.data().get();
              int64_t *dstStrides = vStrides.data().get();

              //        dim3 grid(THCCeilDiv(extendedIdxSize, (int64_t) GROUP_SIZE),
              //            THCCeilDiv(sortedStride, 128L));
              //        dim3 block(WARP_SIZE, GROUP_SIZE);

              //        int dev = 0, numBlocksPerSm = 0, numThreads = 0;
              int blockSize;
              int minGridSize;
              int gridSize;
              cudaOccupancyMaxPotentialBlockSize(
                  &minGridSize,
                  &blockSize,
                  (const void*)&backward_indexing_kernel<scalar_t>);
              gridSize = std::min(minGridSize,
                  (int) (extendedIdxSize + blockSize - 1) / blockSize);
              blockSize = std::min(blockSize,
                  (int) nextP2((unsigned int) extendedIdxSize));
//              printf("%d %d %d\n", minGridSize, gridSize, blockSize);

              void* args[] = {&extendedLinearIndexPtr,
                              &origCountersPtr,
                              &valuePtr,
                              &extendedIdxSize,
                              &dstSize,
                              &sortedStride,
                              &idxSize,
                              &self_info.data,
                              &self_info.dims,
                              &dstSizes,
                              &dstStrides};

              THCudaCheck(
                  cudaLaunchCooperativeKernel( //<decltype(backward_indexing_kernel<scalar_t>)>(
                      (const void*)&backward_indexing_kernel<scalar_t>,
                      gridSize,
                      blockSize,
                      args,
                      0,
                      stream));
              THCudaCheck(cudaStreamSynchronize(stream));
              THCudaCheck(cudaGetLastError());
            });
        /*

              AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "index_put_cuda_2",
           [&] { const scalar_t* pvalue = value.data<scalar_t>(); const int64_t
           extendedIdxSize = extendedLinearIndex.numel(); int64_t*
           extendedLinearIndex_beg = extendedLinearIndex.data<int64_t>();
                int64_t* extendedLinearIndex_end = extendedLinearIndex_beg +
           extendedIdxSize; int64_t* origCounters_beg =
           origCounters.data<int64_t>(); auto self_info =
           cuda::detail::getTensorInfo<scalar_t, int64_t>(self);
                TensorAccumMixedPutOp<scalar_t> amPutOp(self_info, pvalue,
           extendedLinearIndex_beg, extendedLinearIndex_end, origCounters_beg,
           extendedIdxSize / idxSize, idxSize, side == 0);  // <-- before

                thrust::counting_iterator <int64_t> first(0);
                thrust::counting_iterator <int64_t> last(extendedIdxSize);
                thrust::for_each(policy, first, last, amPutOp);

              });
        */
      }
    }
    return self_;
  }
}

}
