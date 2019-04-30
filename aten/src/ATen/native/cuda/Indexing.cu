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
#include <ATen/native/Indexing.h>
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

#ifdef __HIP_PLATFORM_HCC__
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif
#define GRID_SIZE 256

namespace at {
namespace native {

// FIXME
// MOVE THIS STUFF TO SOME COMMON PLACE SHARED BY .cpp and .cu

// STUFF BEGIN
[[noreturn]]
static void invalid_mask(const Tensor& self, int64_t idx, const Tensor& mask, int64_t maskIdx) {
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

static std::vector<Tensor> expandByteTensors(const Tensor& self, TensorList indices) {
  // Expands byte tensors (masks) into the equivalent indexing by LongTensors
  std::vector<Tensor> result;
  for (auto& index : indices) {
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
  auto isDefined = [](const Tensor& tensor) { return tensor.defined(); };
  auto isNull = [](const Tensor& tensor) { return !tensor.defined(); };
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
static std::tuple<Tensor, std::vector<Tensor>> transposeToFront(Tensor self, TensorList indices) {
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

static std::vector<int64_t> computeLinearStride(const Tensor& tensor) {
  // computes the stride as if tensor were contigous
  auto sizes = tensor.sizes();
  std::vector<int64_t> stride(tensor.dim());
  stride[tensor.dim() - 1] = 1;
  std::partial_sum(sizes.rbegin(), sizes.rend() - 1, stride.rbegin() + 1,
      std::multiplies<int64_t>());
  return stride;
}

// Unsqueezes src `before` times at the front and `after` times at the end
static Tensor unsqueezeN(const Tensor& src, int64_t before, int64_t after) {
  auto srcSizes = src.sizes();
  auto nDim = src.dim();
  std::vector<int64_t> sizes(nDim + before + after, 1);
  for (int64_t i = 0; i < nDim; i++) {
    sizes[i + before] = srcSizes[i];
  }
  return src.view(sizes);
}

// STUFF END


//long cnt = 0L;
//long total = 0L;



__global__ void arange_kernel(int64_t n, int64_t* a, int64_t multiplier) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
      i += blockDim.x * gridDim.x) {
    a[i] = i * multiplier;
  }
}

static std::tuple<Tensor, Tensor, Tensor, Tensor, int64_t, int64_t, int64_t, int64_t>
makeLinearIndex(Tensor self, TensorList orig) {
  checkIndexTensorTypes(orig);
  // first expand ByteTensor (boolean masks) into 1 or more LongTensors
  auto indices = expandByteTensors(self, orig);
  // next broadcast all index tensors together
  indices = expand_outplace(indices);
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < (size_t) self.dim()) {
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
      Tensor index = (indices[i].remainder(self.size(i)) * strides[i]).to(kLong);
      if (linearIndex.defined()) {
        linearIndex += index;
      } else {
        linearIndex = index;
      }
    } else if (linearIndex.defined()) {
      emptyAfter++;
      nElemAfter *= self.size(i);
    } else {
      emptyBefore++;
      nElemBefore *= self.size(i);
    }
  }

//  ++cnt;
//  auto start = std::chrono::high_resolution_clock::now();

  // Compute the linear indices for the parts of the tensor not being indexed
  // ...and not being sorted
  Tensor beforeIndex;
  if (emptyBefore > 0) {
    Tensor index = at::native::empty_cuda({nElemBefore},
        self.options().dtype(kLong).device(at::DeviceType::CUDA));
    int64_t *pData = index.data<int64_t>();
    int64_t multiplier = strides[emptyBefore - 1];
    void* args[] = {&nElemBefore, &pData, &multiplier};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    THCudaCheck(
        cudaLaunchKernel((const void*) &arange_kernel,
            GRID_SIZE, WARP_SIZE, args, 0, stream));
    THCudaCheck(cudaStreamSynchronize(stream));
    //index = index * strides[emptyBefore - 1];
    index = index.view(self.sizes().slice(0, emptyBefore));
    beforeIndex = unsqueezeN(index, 0, linearIndex.dim() + emptyAfter);
  }
  Tensor afterIndex;
  if (emptyAfter > 0) {
    Tensor index = at::native::empty_cuda({nElemAfter},
        self.options().dtype(kLong).device(at::DeviceType::CUDA));
    int64_t *pData = index.data<int64_t>();
    int64_t multiplier = 1L;
    void* args[] = {&nElemAfter, &pData, &multiplier};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    THCudaCheck(
        cudaLaunchKernel((const void*) &arange_kernel,
            GRID_SIZE, WARP_SIZE, args, 0, stream));
    THCudaCheck(cudaStreamSynchronize(stream));
    index = index.view(self.sizes().slice(self.dim() - emptyAfter, emptyAfter));
    afterIndex = unsqueezeN(index, linearIndex.dim() + emptyBefore, 0);
  }

//  auto finish = std::chrono::high_resolution_clock::now();
//  total += std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
//  if (cnt % 100 == 0) {
//    std::cout << "kernel pars: " << total / cnt /1000 << "us" << std::endl;
//  }


  return std::make_tuple(self, linearIndex, beforeIndex, afterIndex, emptyBefore, emptyAfter,
      nElemBefore, nElemAfter);
}

template<typename IndexType>
__device__ __forceinline__
IndexType dstOffset(IndexType dims, const IndexType* sizes,
    const IndexType* strides, IndexType linearIndex) {
  IndexType offset(0);
  for (IndexType i = dims - 1; i > 0; --i) {
    offset += (linearIndex % sizes[i]) * strides[i];
    linearIndex /= sizes[i];
  }
  return offset + linearIndex* strides[0];
}

template<typename index_t>
__device__ __forceinline__
index_t extended_idx_ordered(index_t nseq, index_t sortedSize,
    index_t extendedSize, index_t extendedStride,
    const int64_t* origOrder) {
//  index_t aPart = nseq % sortedSize;
//  index_t aPartOrdered = origOrder[aPart];
//  index_t ret = nseq + (aPartOrdered - aPart) * extendedStride;
//  if (ret < 0) {
//    ret += extendedSize;
//  } else if (ret >= extendedSize) {
//    ret -= extendedSize;
//  }

/*
  index_t row = nseq % sortedSize;
  index_t col = nseq / sortedSize;

//  printf("EXT %d %d %d %d   r %d     c %d\n", nseq, sortedSize, extendedSize,extendedStride, row, col);

  row = origOrder[row];
//  index_t ret = col * sortedSize + row;
index_t ret = col * sortedSize + row;
*/

//index_t row = nseq % sortedSize;
//index_t ret = nseq - row + origOrder[row];

//index_t row = nseq / extendedStride;

//row = origOrder[row];

index_t ret = (nseq  % extendedStride) + origOrder[nseq / extendedStride] * extendedStride;

//printf("EXT %d-%d    row %d   or %lld     : %d\n", nseq, (nseq  % sortedSize), row, origOrder[row] ,ret);

return ret;
}

template<typename scalar_t>
__global__
void backward_indexing_kernel(const int64_t* extendedIdx,
    const int64_t* origOrder, const scalar_t* gradValues,
    int64_t extendedSize, int64_t extendedStride, int64_t sortedSize,
    scalar_t* dstData, int64_t dstDims,
    const int64_t* dstSizes, const int64_t* dstStrides,
    bool accumulate) {
  using accscalar_t = acc_type<scalar_t, true>;
  const int realGroup = extendedSize / extendedStride;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= extendedSize) {
    return;
  }
//  printf("** %lld %lld %lld %lld\n", extendedSize, extendedStride, sortedSize, dstDims);

//  for (int k = 0; k < extendedSize && i == 0; ++k)
//    printf("************ %lld ", extendedIdx[k]);

  int dstIdxPrev = -1;
  bool done = false;
  bool bonusTrack = false;
  int dstIdx = -1;
  int extIdx = -1;
  int ii = i;
  do {
    int iii = ii;
    do {
      // j goes along sorted dimensions.
      int j = (iii % realGroup) * extendedStride + iii / realGroup;
//      int extIdx_ = extended_idx_ordered<int>(j, sortedSize,
//          extendedSize, extendedStride,
//          origOrder);

      int extIdx_ = (j % extendedStride) + origOrder[j / extendedStride] * extendedStride;

      int dstIdx_ = extendedIdx[extIdx_];
      if (dstIdxPrev < 0) {
        dstIdxPrev = dstIdx_;
        extIdx = extIdx_;
        dstIdx = dstIdx_;
  //      printf("iii j START %d %d %d %d %d\n", iii, j, dstIdxPrev,extIdx_, dstIdx_);
      } else if (bonusTrack) {
        if (dstIdxPrev != dstIdx_) {
          // (i+1)-th thread will handle this, exit
    //      printf("iii j dstIdxPrev %d %d %d %d\n", iii, j, dstIdxPrev, dstIdx_);
          done = true;
        } else {
          // keep rolling
          extIdx = extIdx_;
          dstIdx = dstIdx_;
        }
        break;
      } else {
        // (i-1)-th thread will handle this, exit
        done = (dstIdxPrev == dstIdx_);
//if (done)        printf("iii j (i-1) %d %d %d %d\n", i, j, dstIdxPrev, dstIdx_);
        break;
      }
      --iii;
    } while (iii >= 0);
    if (done) {
  //    printf("iii %d DONE\n", i);
      break;
    }
    int offset = dstOffset<int64_t>(dstDims, dstSizes, dstStrides, dstIdx);

/*
    IndexType dstOffset(IndexType dims, const IndexType* sizes,
        const IndexType* strides, IndexType linearIndex) {
      IndexType offset(0);
      for (IndexType i = dims - 1; i > 0; --i) {
        offset += (linearIndex % sizes[i]) * strides[i];
        linearIndex /= sizes[i];
      }
      return offset + linearIndex* strides[0];
*/

//    int offset = 0;
//    int li = dstIdx;
//    for (int k = dstDims - 1; k > 0; --k) {
//      offset += (li % dstSizes[k]) * dstStrides[k];
//      li /= dstSizes[k];
//    }
//    offset += li * dstStrides[0];

    if (accumulate) {
    // it's now safe, one thread comes here
      dstData[offset] += gradValues[extIdx];
    } else {
      dstData[offset] = gradValues[extIdx];
    }

//    printf("offset->data %d %d %g\n", iii, offset, gradValues[extIdx]);

    dstIdxPrev = dstIdx;
    ++ii;
    bonusTrack = true;
  } while (ii < extendedSize);
}

Tensor& index_put_cuda_(Tensor& self, TensorList indices, const Tensor& value,
    bool accumulate) {
  if (indices.size() > (size_t) self.dim()) {
    AT_INDEX_ERROR("too many indices for tensor of dimension ", self.dim(), " (got ",
        indices.size(), ")");
  }

  Tensor linearIndex;
  Tensor beforeIndex, afterIndex;
  int64_t emptyBefore = 0L, emptyAfter = 0L;
  int64_t nElemBefore = 1L, nElemAfter = 1L;

  std::tie(self, linearIndex, beforeIndex, afterIndex, emptyBefore, emptyAfter, nElemBefore,
      nElemAfter) = makeLinearIndex(self, indices);

  auto sortedLinearIndex = linearIndex.clone();
  auto origCounters = at::empty_like(linearIndex);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  int64_t idxSize = linearIndex.numel(); // const breaks cudaKernelLaunch
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "xput_cuda_1", [&] {
    int64_t* sortedLinearIndex_beg = sortedLinearIndex.data<int64_t>();
    int64_t* origCounters_beg = origCounters.data<int64_t>();
    auto sortedLinearIndex_iter = thrust::device_ptr<int64_t>(sortedLinearIndex_beg);
    auto origCounters_iter = thrust::device_ptr<int64_t>(origCounters_beg);
    auto self_info = cuda::detail::getTensorInfo<scalar_t, int64_t>(self);
    int64_t multiplier = 1L;
    void* args[] = {&idxSize, &origCounters_beg, &multiplier};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    THCudaCheck(
        cudaLaunchKernel((const void*) &arange_kernel,
            WARP_SIZE, GRID_SIZE, args, 0, stream));
    THCudaCheck(cudaStreamSynchronize(stream));

    thrust::sort_by_key(policy, sortedLinearIndex_iter, sortedLinearIndex_iter + idxSize,
        origCounters_iter, ThrustLTOp<int64_t>());
  });
//    std::cout << "----> linearIndex" << std::endl
//      << linearIndex  << std::endl
//      << std::endl << linearIndex.sizes() << std::endl
//      << "strides: " << computeLinearStride(linearIndex)
//      << std::endl << std::endl;
//
//  std::cout << "----> origCounters" << std::endl
//            << origCounters
//            << std::endl << std::endl;


  // Sum with broadcasting to compute the full index
  // using unsorted original
  linearIndex = unsqueezeN(linearIndex, emptyBefore, emptyAfter);
  if (emptyBefore > 0) {
    linearIndex = linearIndex + beforeIndex;

//  std::cout << "*********** linearIndex+ beforeIndex;" << std::endl
//    << linearIndex  << std::endl
//    << std::endl << linearIndex.sizes() << std::endl
//    << "strides: " << computeLinearStride(linearIndex)
//    << std::endl << std::endl;


  }
  if (emptyAfter > 0) {
    linearIndex = linearIndex + afterIndex;

//    std::cout << "*********** linearIndex+ afterIndex;" << std::endl
//              << linearIndex  << std::endl
//              << std::endl << linearIndex.sizes() << std::endl
//              << "strides: " << computeLinearStride(linearIndex)
//              << std::endl << std::endl;

  }

  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "index_put_cuda_kernel_", [&] {
    cuda::detail::TensorInfo <scalar_t, int64_t> self_info =
        cuda::detail::getTensorInfo<scalar_t, int64_t>(self);
    int64_t dstDims = self_info.dims;
    Tensor dstSizes =
        CPU(kLong).tensorFromBlob(self_info.sizes, {dstDims});
    Tensor dstStrides =
        CPU(kLong).tensorFromBlob(self_info.strides, {dstDims});
    dstSizes = dstSizes.to(at::DeviceType::CUDA, kLong, true, true);
    dstStrides = dstStrides.to(at::DeviceType::CUDA, kLong, true, true);

    scalar_t* valuePtr = value.data<scalar_t>();
    int64_t extendedSize = linearIndex.numel();
    int64_t* origCountersPtr = origCounters.data<int64_t>();
    int64_t* extendedLinearIndexPtr = linearIndex.data<int64_t>();
    int64_t* dstSizesPtr = dstSizes.data<int64_t>();
    int64_t* dstStridesPtr = dstStrides.data<int64_t>();

//    std::cout << "nElemAfter " << std::endl
//              << nElemAfter  << std::endl << std::endl;


    dim3 gridSize(GRID_SIZE);
    dim3 blockSize(WARP_SIZE);
    void* args[] = {&extendedLinearIndexPtr, &origCountersPtr, &valuePtr,
                    &extendedSize, &nElemAfter, &idxSize,
                    &self_info.data, &dstDims, &dstSizesPtr, &dstStridesPtr,
                    &accumulate};
    THCudaCheck(
        cudaLaunchKernel((const void*) &backward_indexing_kernel<scalar_t>, gridSize, blockSize,
            args, 0, stream));
    THCudaCheck(cudaStreamSynchronize(stream));
  });
  return self;
}

}}


//long cnt = 0L;
//long total = 0L;
//
//

//++cnt;
//auto start = std::chrono::high_resolution_clock::now();
//
//auto finish = std::chrono::high_resolution_clock::now();
//total += std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
//if (cnt % 100 == 0) {
//std::cout << "AFTER indices: " << total / cnt << "ns" << std::endl;
//}

//    std::cout << "afterIndex" << std::endl;
//                        print(afterIndex, 120);
//                        std::cout << std::endl
//                                  << "strides: " << computeLinearStride(afterIndex)
//                                  << std::endl
//                                  << std::endl;

//std::cout << "*********** beforeIndex" << std::endl;
//print(beforeIndex, 120);
//std::cout << beforeIndex.sizes() << std::endl
//<< "strides: " << computeLinearStride(beforeIndex)
//<< std::endl
//<< std::endl;
