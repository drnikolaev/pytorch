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

namespace at {
namespace native {

// FIXME
// MOVE THIS STUFF TO SOME COMMON PLACE SHARED BY .cpp and .cu

// STUFF BEGIN
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

// STUFF END


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
      if (linearIndex.defined()) {
        linearIndex += indices[i].remainder(self.size(i)) * strides[i];
      } else {
        linearIndex = indices[i].remainder(self.size(i)) * strides[i];
      }
    } else if (linearIndex.defined()) {
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
    beforeIndex =
        at::arange(0, nElemBefore, self.options().dtype(kLong)) * strides[emptyBefore - 1];
    beforeIndex = beforeIndex.view(self.sizes().slice(0, emptyBefore));
    beforeIndex = unsqueezeN(beforeIndex, 0, linearIndex.dim() + emptyAfter);
  }
  Tensor afterIndex;
  if (emptyAfter > 0) {
    afterIndex = at::arange(0, nElemAfter, self.options().dtype(kLong));
    afterIndex = afterIndex.view(self.sizes().slice(self.dim() - emptyAfter, emptyAfter));
    afterIndex = unsqueezeN(afterIndex, linearIndex.dim() + emptyBefore, 0);
  }
  return std::make_tuple(self, linearIndex, beforeIndex, afterIndex, emptyBefore, emptyAfter,
      nElemBefore, nElemAfter);
}





template<typename T, typename IndexType>
__device__ __forceinline__
IndexType indexToOffset(const at::cuda::detail::TensorInfo<T, IndexType>& info,
    IndexType linearIndex) {
  IndexType offset(0);
  for (int i = info.dims - 1; i > 0; --i) {
    offset += (linearIndex % info.sizes[i]) * info.strides[i];
    linearIndex /= info.sizes[i];
  }
  return offset + linearIndex * info.strides[0];
}

template<typename T, typename IndexType>
__device__ __forceinline__
IndexType indexToOffset(IndexType dims, IndexType* sizes, IndexType* strides,
    IndexType linearIndex) {
  IndexType offset(0);
  for (IndexType i = dims - 1; i > 0; --i) {
    offset += (linearIndex % sizes[i]) * strides[i];
    linearIndex /= sizes[i];
  }
  return offset + linearIndex* strides[0];
}

template<typename index_t>
__device__ __forceinline__
index_t extended_pos(index_t nseq, index_t sortedStride, index_t sortedSize,
    const int64_t* origOrder) {
return nseq / sortedSize + origOrder[nseq % sortedSize] * sortedStride;
  //return origOrder[nseq / sortedSize + (nseq % sortedSize) * sortedStride];

  //const int n = idx * blockSize + currentThreadInBlock;
//const int sortedno = n / sortedStride;
//const int unsorted = origOrder[sortedno % sortedSize];
//return unsorted * sortedStride + (n % sortedStride);
}

template<typename index_t>
__device__ __forceinline__
    index_t extended_pos(index_t idx, index_t blockSize, index_t currentThreadInBlock,
    index_t sortedStride, index_t sortedSize, const int64_t* origOrder) {
  const index_t nseq = idx * blockSize + currentThreadInBlock;
  return extended_pos(nseq, sortedStride, sortedSize, origOrder);
//const int n = idx * blockSize + currentThreadInBlock;
//const int sortedno = n / sortedStride;
//const int unsorted = origOrder[sortedno % sortedSize];
//return unsorted * sortedStride + (n % sortedStride);
}

template<typename scalar_t>
__global__ void
backward_indexing_kernel(const int64_t* extendedIdx, int64_t* origOrder, scalar_t* gradValues,
    int64_t extendedIdxSize, int64_t sortedStride, int64_t sortedSize, scalar_t* dstData,
    int dstDims, int64_t* dstSizes, int64_t* dstStrides) {
  using accscalar_t = acc_type<scalar_t, true>;
//  thread_block block = this_thread_block();
//  thread_block_tile<GROUP_SIZE> tile = tiled_partition<GROUP_SIZE>(block);
//  int th = tile.thread_rank();
//  int idx = block.thread_rank() / GROUP_SIZE;
  int blockSize = blockDim.x * blockDim.y * blockDim.z;
//  int thRank = (threadIdx.z * blockDim.y * blockDim.x) +
//      (threadIdx.y * blockDim.x) + threadIdx.x;
  int idx = blockIdx.x;// * blockDim.x;
  int th = threadIdx.x;
//  int gridSize = blockDim.x * gridDim.x;
  //printf("...%d %d %d %d %d %lld\n", blockSize, thRank, idx, th, gridSize, extendedIdxSize);

//  int blockHeadPos = backward_indexing_extended_pos<int64_t>(idx,
//      blockSize, 0L, sortedStride, sortedSize, origOrder);
//  if (blockHeadPos < extendedIdxSize &&
//    (idx == 0 || blockHeadPos != backward_indexing_extended_pos<int64_t>(idx - 1,
//      blockSize, 0L, sortedStride, sortedSize, origOrder))) {

    int blockHeadPos = extended_pos<int64_t>(idx,
        blockSize, 0L, sortedStride, sortedSize, origOrder);
    if (blockHeadPos < extendedIdxSize &&
        (idx == 0 || blockHeadPos != extended_pos<int64_t>(idx - 1,
            blockSize, 0L, sortedStride, sortedSize, origOrder))) {

    const int blocksDone = idx * WARP_SIZE;
    const int nseq = blocksDone + th;
//    const int no = nseq / sortedSize + (nseq % sortedSize) * sortedStride; // to step along sorted
    const int no = extended_pos<int>(nseq, sortedStride, sortedSize, origOrder);
    do {
      __shared__ int offsetArr[WARP_SIZE];
      __shared__ accscalar_t valArr[WARP_SIZE];
      if (no < extendedIdxSize) {
        const int linear_index = extendedIdx[no];
        offsetArr[th] = indexToOffset<scalar_t, int64_t>(dstDims, dstSizes, dstStrides,
            linear_index);
        valArr[th] = gradValues[no];
      }
      __syncthreads();
      if (th == 0) {
        int currentBlockEnd = extendedIdxSize - blocksDone;
//#pragma unroll
        for (int t = 0; t < WARP_SIZE; ++t) {
          if (t >= currentBlockEnd) {
            break;
          }

//          printf("...%d %d %d %d %d\n", nseq, idx, th, t, offsetArr[t]);

          dstData[offsetArr[t]] += valArr[t];
        }
      }
      ++idx;
      const int blockHeadPosNext = extended_pos<int64_t>(idx,
          blockSize, 0L, sortedStride, sortedSize, origOrder);
      if (blockHeadPosNext >= extendedIdxSize || blockHeadPosNext != blockHeadPos) {
        // next block processing other images indexes
        break;
      }
      blockHeadPos = blockHeadPosNext;  // keep going in the same warp
    } while (true);
  }
}


template<typename T>
struct TensorAccumFullyIndexedPutOp : thrust::unary_function<int64_t, T> {
  TensorAccumFullyIndexedPutOp(at::cuda::detail::TensorInfo<T, int64_t> info, const T* psrc,
      const int64_t* sorted_idx_beg, const int64_t* sorted_idx_end, const int64_t* orig_idx_beg)
      : info(info), p_source(psrc), sorted_idx_beg(sorted_idx_beg), sorted_idx_end(sorted_idx_end),
        orig_idx_beg(orig_idx_beg) {}

  __device__ __forceinline__ T
  operator()(int64_t n) {
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


long long mtotal = 0;
long long xtotal = 0;
long long x2total = 0;
long pcnt = 0;
long xcnt = 0;
long x2cnt = 0;


Tensor& index_put_cuda_(Tensor& self_, TensorList indices, const Tensor& value, bool accumulate) {
  if (indices.size() > (size_t) self_.dim()) {
    AT_INDEX_ERROR("too many indices for tensor of dimension ", self_.dim(), " (got ",
        indices.size(), ")");
  }

//  Tensor & set_(Storage source);
//  Tensor & set_(Storage source);
  Tensor self, linearIndex; //, expandedValue;
  Tensor beforeIndex, afterIndex;
  int64_t emptyBefore = 0L, emptyAfter = 0L;
  int64_t nElemBefore = 1L, nElemAfter = 1L;

  ++x2cnt;
  auto start2 = std::chrono::high_resolution_clock::now();


  std::tie(self, linearIndex, beforeIndex, afterIndex, emptyBefore, emptyAfter, nElemBefore,
      nElemAfter) = makeLinearIndex(self_, indices);










/*

//  inline std::vector<Tensor> expand_outplace(TensorList to_expand)

//  static std::tuple<Tensor, Tensor, Tensor, Tensor, int64_t, int64_t, int64_t, int64_t>
//  makeLinearIndex(Tensor self, TensorList orig) {
    checkIndexTensorTypes(indices_);
    // first expand ByteTensor (boolean masks) into 1 or more LongTensors
    auto indices = expandTensors(self_, indices_); //expandByteTensors(self, indices);
    // next broadcast all index tensors together
    indices = expand_outplace(TensorList(indices));
    // add missing null Tensors so that it matches self.dim()
    while (indices.size() < (size_t) self_.dim()) {
      indices.emplace_back();
    }
    // if the non-null indices are not all adjacent, transpose self and indices
    // together so that they're adjacent at the front
 // TensorList indices(indicesArr);

  Tensor self;// = self_.clone();

    std::cout << "--> self_" << std::endl;
    print(self_, 120);
    std::cout << std::endl
              << "strides: " << computeLinearStride(self_)
              << std::endl << self_.is_contiguous()
              << std::endl
              << std::endl;

  //  std::cerr << "############## " << (self.is_alias_of(self_)) << std::endl;

    //std::tuple<Tensor, TensorList> t = std::make_tuple(self, indices);
    if (!hasContiguousSubspace(indices)) {
//      self = self_.clone();
      std::tie(self, indices) = transposeToFront(self_, indices);
  //    self_.set_(self);
    }

  self_.resize_as_(self);
  self_ = self_.permute(self.dim());


  std::cout << "<-- self" << std::endl;
  print(self, 120);
  std::cout << std::endl
            << "strides: " << computeLinearStride(self)
            << std::endl
            << std::endl;

  //  std::cerr << "!!!!!!!!!!!!!!!!!!11 " << (self.is_alias_of(self_)) << std::endl;

  // To prevent
  // RuntimeError: self__impl_saved == self_.getIntrusivePtr()
  // ASSERT FAILED at ../torch/csrc/autograd/generated/VariableType_1.cpp:11490,
  // please report a bug to PyTorch.
//  Tensor self = self_.clone();

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
      } else if (linearIndex.defined()) {
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
      beforeIndex =
          at::arange(0, nElemBefore, self.options().dtype(kLong)) * strides[emptyBefore - 1];
      beforeIndex = beforeIndex.view(self.sizes().slice(0, emptyBefore));
      beforeIndex = unsqueezeN(beforeIndex, 0, linearIndex.dim() + emptyAfter);
    }
    Tensor afterIndex;
    if (emptyAfter > 0) {
      afterIndex = at::arange(0, nElemAfter, self.options().dtype(kLong));
      afterIndex = afterIndex.view(self.sizes().slice(self.dim() - emptyAfter, emptyAfter));
      afterIndex = unsqueezeN(afterIndex, linearIndex.dim() + emptyBefore, 0);
    }

*/

















  auto finish2 = std::chrono::high_resolution_clock::now();
  x2total += std::chrono::duration_cast<std::chrono::nanoseconds>(finish2 - start2).count();
  if (x2cnt % 100 == 0) {
    std::cout << "makeLinearIndex time: " << x2total / x2cnt / 1000 << "us" << std::endl;
  }


//  std::cout << "UNsorted linearIndex" << std::endl;
//  print(linearIndex, 120);
//  std::cout << std::endl
//            << "strides: " << computeLinearStride(linearIndex)
//            << std::endl
//            << std::endl;




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
    auto sortedLinearIndex_iter = thrust::device_ptr<int64_t>(sortedLinearIndex_beg);
    auto origCounters_iter = thrust::device_ptr<int64_t>(origCounters_beg);
    auto self_info = cuda::detail::getTensorInfo<scalar_t, int64_t>(self);
    TensorAccumFullyIndexedPutOp<scalar_t> aiPutOp(self_info, value.data<scalar_t>(),
        sortedLinearIndex_beg, sortedLinearIndex_end, origCounters_beg);

    ++xcnt;
    auto start = std::chrono::high_resolution_clock::now();


    thrust::sequence(policy, origCounters_iter, origCounters_iter + idxSize);

    thrust::sort_by_key(policy, sortedLinearIndex_iter, sortedLinearIndex_iter + idxSize,
        origCounters_iter, ThrustLTOp<int64_t>());



    auto finish = std::chrono::high_resolution_clock::now();
    xtotal += std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
    if (xcnt % 100 == 0) {
      std::cout << "sort time: " << xtotal / xcnt / 1000 << "us" << std::endl;
    }



    if (!beforeIndex.defined() && !afterIndex.defined()) {
      // Full size index, done:
      thrust::counting_iterator<int64_t> first(0);
      thrust::counting_iterator<int64_t> last(idxSize);
      thrust::for_each(policy, first, last, aiPutOp);
    }
  });


//  std::cout << "sortedLinearIndex" << std::endl;
//  print(sortedLinearIndex, 120);
//  std::cout << std::endl
//            << "strides: " << computeLinearStride(sortedLinearIndex)
//            << std::endl
//            << std::endl;


  if (beforeIndex.defined() || afterIndex.defined()) {
    // trying to reuse device memory
    //Tensor extendedLinearIndex = sortedLinearIndex;
    // Sum with broadcasting to compute the full index
    // using unsorted original
    linearIndex = unsqueezeN(linearIndex, emptyBefore, emptyAfter);

//    std::cout << "linearIndex" << std::endl;
//    print(linearIndex, 120);
//    std::cout << std::endl
//              << "strides: " << computeLinearStride(linearIndex)
//              << std::endl
//              << std::endl;

     if (emptyBefore > 0) {
        beforeIndex = unsqueezeN(beforeIndex, 0, linearIndex.dim() + emptyAfter);
        linearIndex  = linearIndex + beforeIndex;

//                  std::cout << "beforeIndex" << std::endl;
//                  print(beforeIndex, 120);
//                  std::cout << std::endl
//                            << "strides: " << computeLinearStride(beforeIndex)
//                            << std::endl
//                            << std::endl;

      }

      if (emptyAfter > 0) {
        afterIndex = unsqueezeN(afterIndex, linearIndex.dim() + emptyBefore, 0);
        linearIndex  = linearIndex + afterIndex;

//                  std::cout << "afterIndex" << std::endl;
//                  print(afterIndex, 120);
//                  std::cout << std::endl
//                            << "strides: " << computeLinearStride(afterIndex)
//                            << std::endl
//                            << std::endl;

      }
      linearIndex.squeeze_();

//              std::cout << "extended linearIndex" << std::endl;
//              print(linearIndex , 120);
//              std::cout << std::endl
//                        << "strides: " << computeLinearStride(linearIndex )
//                        << std::endl
//                        << std::endl;
//


      AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "index_put_cuda_kernel_", [&] {



        cuda::detail::TensorInfo <scalar_t, int64_t> self_info =
            cuda::detail::getTensorInfo<scalar_t, int64_t>(self);
        scalar_t* valuePtr = value.data<scalar_t>();
        int64_t extendedIdxSize = linearIndex .numel();
        int64_t* origCountersPtr = origCounters.data<int64_t>();
        int64_t* extendedLinearIndexPtr = linearIndex .data<int64_t>();
        int64_t sortedStride = extendedIdxSize / idxSize;

//        printf("lambda %ld %ld %ld\n", idxSize, extendedIdxSize, sortedStride);


        thrust::device_vector<int64_t> vSizes(self_info.sizes, self_info.sizes + self_info.dims);
        thrust::device_vector<int64_t> vStrides(self_info.strides,
            self_info.strides + self_info.dims);
        int64_t* dstSizes = vSizes.data().get();
        int64_t* dstStrides = vStrides.data().get();




        int blockSize;
//        int minGridSize;
        int gridSize;
//        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
//            (const void*) &backward_indexing_kernel<scalar_t>);
        blockSize = WARP_SIZE;
//        gridSize = std::min(minGridSize, (int) (extendedIdxSize + blockSize - 1) / blockSize);
        gridSize = (extendedIdxSize + blockSize - 1) / blockSize;
//        blockSize = std::min(blockSize, (int) nextP2((unsigned int) extendedIdxSize));
//                      printf("%d %d %d\n", minGridSize, gridSize, blockSize);

        void* args[] = {&extendedLinearIndexPtr, &origCountersPtr, &valuePtr, &extendedIdxSize,
                        &sortedStride, &idxSize, &self_info.data, &self_info.dims, &dstSizes,
                        &dstStrides};

        ++pcnt;
        auto start = std::chrono::high_resolution_clock::now();


        THCudaCheck(cudaLaunchKernel(
            (const void*) &backward_indexing_kernel<scalar_t>, gridSize, blockSize, args,
            blockSize * 16, stream));
        THCudaCheck(cudaStreamSynchronize(stream));
        THCudaCheck(cudaGetLastError());


        auto finish = std::chrono::high_resolution_clock::now();
        mtotal += std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
        if (pcnt % 100 == 0) {
          std::cout << "kernel time: " << mtotal / pcnt / 1000 << "us" << std::endl;
        }


      });
  }

//                    std::cout << "self" << std::endl;
//                    print(self, 120);
//                    std::cout << std::endl
//                              << "strides: " << computeLinearStride(self)
//                              << std::endl
//                              << std::endl;


  //  std::cerr << "!!!!!!!!!!!!!!!!!!222 " << (self.is_alias_of(self_)) << std::endl;
//  std::cerr << "!!!!!!!!!!!!!!!!!!222 " << (self_.is_alias_of(self)) << std::endl;

//  self_.resize_as_(self);
//  self_ = self_.view_as(self);
//  self_.view_as(self).copy_(self);
  return self_;
}

}}




//static std::vector<Tensor> expandByteTensors(const Tensor& self, TensorList indices) {
//  // Expands byte tensors (masks) into the equivalent indexing by LongTensors
//  std::vector<Tensor> result;
//  for (auto& index : indices) {
//    if (index.scalar_type() == kByte) {
//      // The sizes of the ByteTensor mask must match the sizes of the
//      // corresponding dimensions in self
//      for (int64_t j = 0; j < index.dim(); j++) {
//        int64_t srcIdx = result.size() + j;
//        if (index.size(j) != self.size(srcIdx)) {
//          invalid_mask(self, srcIdx, index, j);
//        }
//      }
//      // Replace with nonzeros
//      auto nonzero = index.nonzero();
//      auto special_empty = false;
//      for (int64_t j = 0; j < index.dim(); j++) {
//        if (special_empty) {
//          // We can't call select on an empty tensor so we just create an empty
//          // tensor.
//          result.emplace_back(at::empty({0}, nonzero.options()));
//        } else {
//          result.emplace_back(nonzero.select(1, j));
//        }
//      }
//    } else {
//      result.emplace_back(index);
//    }
//  }
//  return result;
//}







//      std::cout << "self " <<  std::endl;
//      print(self, 120);
//      std::cout << std::endl
//                << "strides: " << computeLinearStride(self)
//                << std::endl
//                << std::endl;


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
//}




//static inline unsigned int nextP2(unsigned int v) {
//  v--;
//  v |= v >> 1U;
//  v |= v >> 2U;
//  v |= v >> 4U;
//  v |= v >> 8U;
//  v |= v >> 16U;
//  v++;
//  return v;
//}
//
//[[noreturn]]
//static void invalid_mask(const Tensor& self, int64_t idx, const Tensor& mask, int64_t maskIdx) {
//  std::stringstream ss;
//  ss << "The shape of the mask " << mask.sizes() << " at index " << maskIdx;
//  ss << " does not match the shape of the indexed tensor " << self.sizes();
//  ss << " at index " << idx;
//  AT_INDEX_ERROR(ss.str());
//}
//
//static void checkIndexTensorTypes(TensorList indices) {
//  for (auto& tensor : indices) {
//    if (tensor.defined()) {
//      auto scalarType = tensor.scalar_type();
//      if (scalarType != kLong && scalarType != kByte) {
//        AT_INDEX_ERROR("tensors used as indices must be long or byte tensors");
//      }
//    }
//  }
//}
//
//static std::vector<Tensor> expandByteTensors(const Tensor& self, TensorList indices) {
//  // Expands byte tensors (masks) into the equivalent indexing by LongTensors
//  std::vector<Tensor> result;
//  for (auto& index : indices) {
//    if (index.scalar_type() == kByte) {
//      // The sizes of the ByteTensor mask must match the sizes of the
//      // corresponding dimensions in self
//      for (int64_t j = 0; j < index.dim(); j++) {
//        int64_t srcIdx = result.size() + j;
//        if (index.size(j) != self.size(srcIdx)) {
//          invalid_mask(self, srcIdx, index, j);
//        }
//      }
//      // Replace with nonzeros
//      auto nonzero = index.nonzero();
//      auto special_empty = false;
//      for (int64_t j = 0; j < index.dim(); j++) {
//        if (special_empty) {
//          // We can't call select on an empty tensor so we just create an empty
//          // tensor.
//          result.emplace_back(at::empty({0}, nonzero.options()));
//        } else {
//          result.emplace_back(nonzero.select(1, j));
//        }
//      }
//    } else {
//      result.emplace_back(index);
//    }
//  }
//  return result;
//}
//
//static bool hasContiguousSubspace(TensorList tl) {
//  // true if all the non-null tensors are adjacent
//  auto isDefined = [](const Tensor& tensor) { return tensor.defined(); };
//  auto isNull = [](const Tensor& tensor) { return !tensor.defined(); };
//  auto start = std::find_if(tl.begin(), tl.end(), isDefined);
//  auto stop = std::find_if(tl.rbegin(), tl.rend(), isDefined);
//  auto it = std::find_if(start, stop.base(), isNull);
//  return it == stop.base();
//}
//
// Transposes the tensor and indices together so that all the non-null indices
// index the first k dimensions of the tensor. Returns the transposed tensor
// and the reordered indices. For example:
//  transposeToFront(tensor, {nullptr, a, nullptr, b})
// returns
//  tensor.permute([1, 3, 0, 2]), {a, b, nullptr, nullptr}
//static std::tuple<Tensor, std::vector<Tensor>> transposeToFront(Tensor self, TensorList indices) {
//  std::vector<int64_t> dims;
//  std::vector<Tensor> transposedIndices;
//  dims.reserve(self.dim());
//  for (int64_t i = 0; i < self.dim(); i++) {
//    if (indices[i].defined()) {
//      dims.push_back(i);
//      transposedIndices.emplace_back(indices[i]);
//    }
//  }
//  for (int64_t i = 0; i < self.dim(); i++) {
//    if (!indices[i].defined()) {
//      dims.push_back(i);
//      transposedIndices.emplace_back();
//    }
//  }
//  return std::make_tuple(self.permute(dims), std::move(transposedIndices));
//}
//
//static std::vector<int64_t> computeLinearStride(const Tensor& tensor) {
//  // computes the stride as if tensor were contigous
//  auto sizes = tensor.sizes();
//  std::vector<int64_t> stride(tensor.dim());
//  stride[tensor.dim() - 1] = 1;
//  std::partial_sum(sizes.rbegin(), sizes.rend() - 1, stride.rbegin() + 1,
//      std::multiplies<int64_t>());
//  return stride;
//}

// Unsqueezes src `before` times at the front and `after` times at the end
//static Tensor unsqueezeN(const Tensor& src, int64_t before, int64_t after) {
//  auto srcSizes = src.sizes();
//  auto nDim = src.dim();
//  std::vector<int64_t> sizes(nDim + before + after, 1);
//  for (int64_t i = 0; i < nDim; i++) {
//    sizes[i + before] = srcSizes[i];
//  }
//  return src.view(sizes);
//}

//static std::tuple<Tensor, Tensor, Tensor, Tensor, int64_t, int64_t, int64_t, int64_t>
//makeLinearIndex(Tensor self, TensorList orig) {
//  checkIndexTensorTypes(orig);
//  // first expand ByteTensor (boolean masks) into 1 or more LongTensors
//  auto indices = expandByteTensors(self, orig);
//  // next broadcast all index tensors together
//  indices = expand_outplace(indices);
//  // add missing null Tensors so that it matches self.dim()
//  while (indices.size() < (size_t) self.dim()) {
//    indices.emplace_back();
//  }
//  // if the non-null indices are not all adjacent, transpose self and indices
//  // together so that they're adjacent at the front
//  if (!hasContiguousSubspace(indices)) {
//    std::tie(self, indices) = transposeToFront(self, indices);
//  }
//
//  auto strides = computeLinearStride(self);
//
//  // Compute the linear index by multiplying the indexing tensors by the
//  // stride and summing them. All the indexing tensors have the same shape at
//  // this point. We also compute the number of dimensions before and after that
//  // are not being index.
//  Tensor linearIndex;
//  int64_t emptyBefore = 0, emptyAfter = 0, nElemBefore = 1, nElemAfter = 1;
//  for (int64_t i = 0; i < self.dim(); i++) {
//    if (indices[i].defined()) {
//      if (linearIndex.defined()) {
//        linearIndex += indices[i].remainder(self.size(i)) * strides[i];
//      } else {
//        linearIndex = indices[i].remainder(self.size(i)) * strides[i];
//      }
//    } else if (linearIndex.defined()) {
//      emptyAfter++;
//      nElemAfter *= self.size(i);
//    } else {
//      emptyBefore++;
//      nElemBefore *= self.size(i);
//    }
//  }
//  // Compute the linear indices for the parts of the tensor not being indexed
//  // ...and not being sorted
//  Tensor beforeIndex;
//  if (emptyBefore > 0) {
//    beforeIndex =
//        at::arange(0, nElemBefore, self.options().dtype(kLong)) * strides[emptyBefore - 1];
//    beforeIndex = beforeIndex.view(self.sizes().slice(0, emptyBefore));
//    beforeIndex = unsqueezeN(beforeIndex, 0, linearIndex.dim() + emptyAfter);
//  }
//  Tensor afterIndex;
//  if (emptyAfter > 0) {
//    afterIndex = at::arange(0, nElemAfter, self.options().dtype(kLong));
//    afterIndex = afterIndex.view(self.sizes().slice(self.dim() - emptyAfter, emptyAfter));
//    afterIndex = unsqueezeN(afterIndex, linearIndex.dim() + emptyBefore, 0);
//  }
//  return std::make_tuple(self, linearIndex, beforeIndex, afterIndex, emptyBefore, emptyAfter,
//      nElemBefore, nElemAfter);
//}


//template<typename scalar_t>
//__global__ void
//backward_indexing_kernel_coop(int64_t* extendedIdx, int64_t* origOrder, scalar_t* gradValues,
//    int64_t extendedIdxSize, int64_t sortedStride, int64_t sortedSize, scalar_t* dstData,
//    int dstDims, int64_t* dstSizes, int64_t* dstStrides) {
//  using accscalar_t = acc_type<scalar_t, true>;
//  thread_block block = this_thread_block();
//  thread_block_tile<GROUP_SIZE> tile = tiled_partition<GROUP_SIZE>(block);
//  int th = tile.thread_rank();
//  int idx = block.thread_rank() / GROUP_SIZE;
//
//  if (idx < extendedIdxSize && (idx == 0 || extendedIdx[idx] != extendedIdx[idx - 1])) {
//    do {
//      const int n = idx * GROUP_SIZE + th;
//      __shared__ int offsetArr[GROUP_SIZE];
//      __shared__ accscalar_t valArr[GROUP_SIZE];
//      if (n < extendedIdxSize) {
//        const int sortedno = n / sortedStride;
//        const int unsorted = origOrder[sortedno % sortedSize];
//        const int no = unsorted * sortedStride + (n % sortedStride);
//        const int linear_index = extendedIdx[no];
//        offsetArr[th] = indexToOffset<scalar_t, int64_t>(dstDims, dstSizes, dstStrides,
//            linear_index);
//        valArr[th] = gradValues[no];
//      }
//      tile.sync();
//      if (tile.thread_rank() == 0) {
//        #pragma unroll
//        for (int t = 0; t < GROUP_SIZE; ++t) {
//          dstData[offsetArr[t]] += valArr[t];
//        }
//      }
//      ++idx;
//    } while (idx < extendedIdxSize && extendedIdx[idx] == extendedIdx[idx - 1]);
//  }
//}

//class thread_block : public thread_group {
// public:
//  __device__ unsigned int size() const {
//    return blockDim.x * blockDim.y * blockDim.z;
//  }
//
//  __device__ unsigned int thread_rank() const {
//    return (threadIdx.z * blockDim.y * blockDim.x) + (threadIdx.y * blockDim.x) + threadIdx.x;
//  }
//};

//int i = blockIdx.x*blockDim.x + threadIdx.x;
//template<typename index_t>
//__device__ __forceinline__ index_t
//backward_indexing_extended_pos(index_t idx,
//    index_t blockSize,
//    index_t currentThreadInBlock,
//    index_t sortedStride, index_t sortedSize,
//    const int64_t* origOrder) {
//  const int n = idx * blockSize + currentThreadInBlock;
//  const int sortedno = n / sortedStride;
//  const int unsorted = origOrder[sortedno % sortedSize];
//  return unsorted * sortedStride + (n % sortedStride);
//}



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

//template<typename T>
//struct TensorAccumMixedPutOp : thrust::unary_function<int64_t, T> {
//  TensorAccumMixedPutOp(at::cuda::detail::TensorInfo<T, int64_t> info, const T* psrc,
//      const int64_t* ext_idx_beg, const int64_t* ext_idx_end, const int64_t* orig_idx_beg,
//      int64_t sortedStride, int64_t sortedSize, bool before) : info(info), p_source(psrc),
//                                                               ext_idx_beg(ext_idx_beg),
//                                                               ext_idx_end(ext_idx_end),
//                                                               orig_idx_beg(orig_idx_beg),
//                                                               sortedStride(sortedStride),
//                                                               sortedSize(sortedSize),
//                                                               before(before) {}
//
//  __device__ __forceinline__ T
//
//  operator()(int64_t n) {
//    const int64_t sortedno = n / sortedStride;
//    const int64_t unsorted = orig_idx_beg[sortedno % sortedSize];
//    const int64_t no = unsorted * sortedStride + (n % sortedStride);
//    const T* pvalue = p_source + no;
//    const int64_t* pindex = ext_idx_beg + no;
//    const int64_t linear_index = *pindex;
//    const int64_t offset = indexToOffset<T, int64_t>(info, linear_index);
//
//    // TODO!!!
//    atomicAdd(info.data + offset, *pvalue);
//
//    /*
//    ///    if (pindex == ext_idx_beg || *pindex != *(pindex - 1)) {
//          do {
//            info.data[offset] += *pvalue;
//            pindex += sortedStride;
//            pvalue += sortedStride;
//
//            printf("...%lld %lld %lld %lld %lld %g\n", n , no, *pindex,
//    offset, linear_index, info.data[offset]); } while (pindex != ext_idx_end
//    && *pindex == linear_index);
//    ///    }
//     */
//    return 0; // discarded
//  }
//
//  at::cuda::detail::TensorInfo<T, int64_t> info;
//  const T* p_source;
//  const int64_t* ext_idx_beg;
//  const int64_t* ext_idx_end;
//  const int64_t* orig_idx_beg;
//  const int64_t sortedStride;
//  const int64_t sortedSize;
//  const bool before;
//};
