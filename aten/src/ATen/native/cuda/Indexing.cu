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
#include "../../../../../c10/core/DeviceType.h"

namespace at { namespace native {

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

template <typename T>
struct TensorAccumMixedPutOp : thrust::unary_function<int64_t, T> {
  TensorAccumMixedPutOp(at::cuda::detail::TensorInfo<T, int64_t> info,
  const T* psrc,
  const int64_t* ext_idx_beg, const int64_t* ext_idx_end,
  const int64_t* orig_idx_beg, int64_t sortedStride,
  int64_t sortedSize)
  : info(info), p_source(psrc),
  ext_idx_beg(ext_idx_beg), ext_idx_end(ext_idx_end),
  orig_idx_beg(orig_idx_beg), sortedStride(sortedStride),
  sortedSize(sortedSize) {}

//  __device__ __forceinline__
//  T operator()(thrust::tuple <int64_t, T> tuple_idx) {
//    const int64_t idx = thrust::get<0>(tuple_idx);
//    const int64_t offset = indexToOffset < T, int64_t>(info, idx);
//    const T value = thrust::get<1>(tuple_idx);  // orig_idx
//    info.data[offset] += value;
//    return 0;  // discarded
//  }

//  int64_t no = orig_idx_beg[n / lindexStride] * lindexStride + (n % lindexStride);

  __device__ __forceinline__
  T operator()(int64_t n) {

    printf("%lld %lld %lld %lld \n",n , sortedStride, n / sortedStride, orig_idx_beg[n]);

    const int64_t no = orig_idx_beg[n / sortedStride] * sortedStride + (n % sortedStride);
//    const int64_t ino = no / sortedSize;
    const T* pvalue = p_source + no;
    const int64_t* pindex = ext_idx_beg + no;
    const int64_t linear_index = *pindex;
    const int64_t offset = indexToOffset<T, int64_t>(info, linear_index);

    printf("%lld %lld %lld %lld %g\n", n , no, linear_index, offset, *pvalue);


    if (pindex == ext_idx_beg || *pindex != *(pindex - 1)) {
      do {
        info.data[offset] += *pvalue;
        pindex++;
        pvalue++;

        printf("...%lld %lld %lld %g\n", n , no, *pindex, offset, info.data[offset]);
      } while (pindex != ext_idx_end && *pindex == linear_index);
    }
    return 0;  // discarded
  }

  at::cuda::detail::TensorInfo<T, int64_t> info;
  const T* p_source;
  const int64_t* ext_idx_beg;
  const int64_t* ext_idx_end;
  const int64_t* orig_idx_beg;
  const int64_t sortedStride;
  const int64_t sortedSize;
};

template <typename T>
struct TensorAccumFullyIndexedPutOp : thrust::unary_function<int64_t, T> {
  TensorAccumFullyIndexedPutOp(at::cuda::detail::TensorInfo<T, int64_t> info,
      const T* psrc,
      const int64_t* sorted_idx_beg, const int64_t* sorted_idx_end,
      const int64_t* orig_idx_beg)
    : info(info), p_source(psrc),
      sorted_idx_beg(sorted_idx_beg), sorted_idx_end(sorted_idx_end),
      orig_idx_beg(orig_idx_beg) {}

  __device__ __forceinline__ T operator()(int64_t n) {
    const int64_t no = orig_idx_beg[n];  // restore if flipped
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
    return 0;  // discarded
  }

  const at::cuda::detail::TensorInfo<T, int64_t> info;
  const T* p_source;
  const int64_t* sorted_idx_beg;
  const int64_t* sorted_idx_end;
  const int64_t* orig_idx_beg;
};

Tensor & index_put_cuda_(Tensor & self_, TensorList indices, const Tensor & value, bool accumulate) {
  if (indices.size() > (size_t)self_.dim()) {
    AT_INDEX_ERROR("too many indices for tensor of dimension ", self_.dim(), " (got ", indices.size(), ")");
  }

  Tensor self, linearIndex, expandedValue;
  Tensor beforeIndex, afterIndex;
  int64_t emptyBefore = 0L, emptyAfter = 0L;
  int64_t nElemBefore = 1L, nElemAfter = 1L;

  std::tie(self, linearIndex, beforeIndex, afterIndex,
      emptyBefore, emptyAfter, nElemBefore, nElemAfter) = makeLinearIndex(self_, indices);

  auto sortedLinearIndex = linearIndex;
  auto origCounters = at::empty_like(linearIndex);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  const int64_t idxSize = linearIndex.numel();
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "xput_cuda_1", [&] {
    int64_t* sortedLinearIndex_beg = sortedLinearIndex.data<int64_t>();
    int64_t* sortedLinearIndex_end = sortedLinearIndex_beg + idxSize;
    int64_t* origCounters_beg = origCounters.data<int64_t>();
    auto sortedLinearIndex_iter = thrust::device_ptr<int64_t>(sortedLinearIndex_beg);
    auto origCounters_iter = thrust::device_ptr<int64_t>(origCounters_beg);
    auto self_info = cuda::detail::getTensorInfo<scalar_t, int64_t>(self);
    TensorAccumFullyIndexedPutOp<scalar_t> aiPutOp(self_info,
        value.data<scalar_t>(), sortedLinearIndex_beg, sortedLinearIndex_end,
        origCounters_beg);

    thrust::sequence(policy,
        origCounters_iter, origCounters_iter + idxSize);

    thrust::sort_by_key(
        policy,
        sortedLinearIndex_iter, sortedLinearIndex_iter + idxSize,
        origCounters_iter, ThrustLTOp<int64_t>());

    std::cout << "origCounters" << std::endl;
    print(origCounters, 120);
    std::cout << std::endl << "strides: "
              << computeLinearStride(origCounters) << std::endl << std::endl;
    std::cout << "value" << std::endl;
    print(value, 120);
    std::cout << std::endl << "strides: "
              << computeLinearStride(value) << std::endl << std::endl;
    std::cout << "sortedLinearIndex" << std::endl;
    print(sortedLinearIndex, 120);
    std::cout << std::endl << "strides: "
              << computeLinearStride(sortedLinearIndex) << std::endl << std::endl;

    if (!beforeIndex.defined() && !afterIndex.defined()) {
      // Full size index, done:
      thrust::counting_iterator<int64_t> first(0);
      thrust::counting_iterator<int64_t> last(idxSize);
      thrust::for_each(policy, first, last, aiPutOp);
    }
  });

  if (beforeIndex.defined() || afterIndex.defined()) {
    Tensor extendedLinearIndex;
    // Sum with broadcasting to compute the full index
    linearIndex = unsqueezeN(linearIndex, emptyBefore, emptyAfter);
    if (emptyBefore > 0) {
      beforeIndex = unsqueezeN(beforeIndex, 0, linearIndex.dim() + emptyAfter);

      std::cout << "beforeIndex" << std::endl;
      print(beforeIndex, 120);
      std::cout << std::endl << "strides: "
                << computeLinearStride(beforeIndex) << std::endl << std::endl;

      extendedLinearIndex = linearIndex + beforeIndex;
    }
    if (emptyAfter > 0) {
      assert(false);
      afterIndex = unsqueezeN(afterIndex, linearIndex.dim() + emptyBefore, 0);

      std::cout << "afterIndex" << std::endl;
      print(afterIndex, 120);
      std::cout << std::endl << "strides: "
                << computeLinearStride(afterIndex) << std::endl << std::endl;

      linearIndex = linearIndex + afterIndex;

    }
    extendedLinearIndex.squeeze_();

    std::cout << "extendedLinearIndex" << std::endl;
    print(extendedLinearIndex, 120);
    std::cout << std::endl << "strides: "
              << computeLinearStride(extendedLinearIndex) << std::endl << std::endl;

    AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "index_put_cuda_2", [&] {
      const scalar_t *pvalue = value.data<scalar_t>();
      const int64_t extendedIdxSize = extendedLinearIndex.numel();
      int64_t* extendedLinearIndex_beg = extendedLinearIndex.data<int64_t>();
      int64_t* extendedLinearIndex_end = extendedLinearIndex_beg + extendedIdxSize;
      int64_t* origCounters_beg = origCounters.data<int64_t>();
      auto self_info = cuda::detail::getTensorInfo<scalar_t, int64_t>(self);
      TensorAccumMixedPutOp<scalar_t> amPutOp(self_info,
        pvalue, extendedLinearIndex_beg, extendedLinearIndex_end,
        origCounters_beg, extendedIdxSize / idxSize, idxSize);

//      auto first = thrust::make_zip_iterator(
//          thrust::make_tuple(plindex, pvalue));
//      auto last = thrust::make_zip_iterator(
//          thrust::make_tuple(plindex + extendedIdxSize, pvalue + extendedIdxSize));
      thrust::counting_iterator<int64_t> first(0);
      thrust::counting_iterator<int64_t> last(extendedIdxSize);
      thrust::for_each(policy, first, last, amPutOp);

    });
  }
  return self_;
}

}}




//  printf("%lld %lld %lld %lld %g\n", n , no, linear_index,
//  offset, *pvalue);
//  int64_t no = orig_idx_beg[n / lindexStride] * lindexStride + (n % lindexStride);
//  const int64_t lindex = plindex[no];
//  const int64_t headIdx = lindex / headStride;
//  const int64_t shPos = lindex % headStride;
//  const int64_t offset = headIdx * headStride + shPos;
//
//  printf("%lld %lld %lld %lld %lld %lld %lld\n", n , no, lindex, headStride, headIdx, shPos, offset);
//  int64_t sortedIndexSize;
//  const int64_t* beforeIndex;
//  const int64_t elementsBefore;
//  const int64_t* afterIndex;
//  const int64_t elementsAfter;


//  std::cout << "strides" << std::endl;
//  std::cout << strides << std::endl;
//  std::cout << std::endl << std::endl;

//    std::cout << "beforeIndex" << std::endl;
//    print(beforeIndex, 120);
//    std::cout << std::endl << "strides: "
//              << computeLinearStride(beforeIndex) << std::endl << std::endl;

//    std::cout << "afterIndex" << std::endl;
//    print(afterIndex, 120);
//    std::cout << std::endl << "strides: "
//              << computeLinearStride(afterIndex) << std::endl << std::endl;
//  std::cout << "linearIndex" << std::endl;
//  print(linearIndex, 120);
//  std::cout << std::endl << "strides: "
//      << computeLinearStride(linearIndex) << std::endl << std::endl;

//  Tensor lll = unsqueezeN(linearIndex, emptyBefore, emptyAfter) + afterIndex;
//  std::cout << "lll" << std::endl;
//  print(lll, 120);
//  std::cout << std::endl << "strides: "
//            << computeLinearStride(linearIndex) << std::endl << std::endl;





//    const T* psrc, const int64_t* plindex)
//    const int64_t* orig_idx_beg, int64_t origIdxSize,
//      int64_t headSize, int64_t headStride, int64_t lindexStride)
//   , p_source(psrc), plindex(plindex) {}
//      orig_idx_beg(orig_idx_beg), origIdxSize(origIdxSize),
//      headSize(headSize), headStride(headStride), lindexStride(lindexStride) {}


//    atomicAdd(address, value);
//    T* address = info.data + offset;
//    atomicAdd(address, value);

//__device__ __forceinline__ T operator()(int64_t n) {
//
//
//    const T value = p_source[n];
////    const int64_t lindex = plindex[n];
//    const int64_t offset = indexToOffset(info, plindex[n]);
//    T* address = info.data + offset;
//
////    printf("%lld %lld %lld %g %g\n", n , lindex, offset,
////        info.data[offset], value);
//
//    atomicAdd(address, value);
//
//  return 0;  // discarded
//  }


//  const T* p_source;
//  const int64_t* plindex;
//  const int64_t* orig_idx_beg;
//  const int64_t origIdxSize;
//  const int64_t headSize; // size of sorted linearIndex (aka "head")
//  const int64_t headStride; // distance between shoulder slices
//  const int64_t lindexStride; // full index major stride

//long long mtotal = 0;
//long long xtotal = 0;
//long pcnt = 0;

//////////////////////////////////////////////////
//  ++pcnt;
//  auto start = std::chrono::high_resolution_clock::now();
/////////////////////////////////////////////////////////////

//  auto finish = std::chrono::high_resolution_clock::now();
//
//  mtotal += std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
//  if (pcnt % 100 == 0) {
//    std::cout << "makeLinearIndex: "
//              << mtotal/pcnt/1000
//              << "us" << std::endl;;
//  }


//  std::tie(expandedValue) = expand_inplace(linearIndex, value);

//  start = std::chrono::high_resolution_clock::now();

//  Tensor & ret = src.xput_(linearIndex, value, accumulate, beforeIndex, afterIndex,
//                           emptyBefore, emptyAfter);
//  Tensor & xput_cuda_(Tensor & self, Tensor & linearIndex, const Tensor & source, bool accumulate,
//      Tensor & beforeIndex, Tensor & afterIndex, int64_t emptyBefore, int64_t emptyAfter)

//  std::cout << "value" << std::endl;
//  print(value, 120);
//  std::cout << std::endl << "strides: "
//            << computeLinearStride(value) << std::endl << std::endl;

//  std::cout << "self" << std::endl;
//  print(self, 120);
//  std::cout << std::endl << "strides: "
//            << computeLinearStride(self) << std::endl << std::endl;

//  auto sortedLinearIndex = linearIndex;//.clone();
//  auto orig_index = at::empty_like(linearIndex);


//  if (beforeIndex.defined() || afterIndex.defined()) {

//    const int64_t *beforeIndexPtr = beforeIndex.defined() ? beforeIndex.data<int64_t>() : nullptr;
//    const int64_t elementsBefore = beforeIndex.defined() ? beforeIndex.numel() : 0L;
//    const int64_t *afterIndexPtr = afterIndex.defined() ? afterIndex.data<int64_t>() : nullptr;
//    const int64_t elementsAfter = afterIndex.defined() ? afterIndex.numel() : 0L;


//      std::cout << "beforeIndex" << std::endl;
//      print(beforeIndex, 120);
//      std::cout << std::endl << "strides: "
//                << computeLinearStride(beforeIndex) << std::endl << std::endl;

//      std::cout << "afterIndex" << std::endl;
//      print(afterIndex, 120);
//      std::cout << std::endl << "strides: "
//                << computeLinearStride(afterIndex) << std::endl << std::endl;

//    std::cout << "linearIndex GPU" << std::endl;
//    print(linearIndex, 120);
//    std::cout << std::endl << "strides: "
//              << computeLinearStride(linearIndex) << std::endl << std::endl;


//    const int64_t* orig_idx_beg = orig_index.data<int64_t>();
//    const int64_t* orig_idx_end = orig_idx_beg + idxSize;
//    auto full_index_iter = thrust::device_ptr<int64_t>(linearIndex.data<int64_t>());
//    int64_t liStride = fullIdxSize / idxSize;
//    const int64_t headStride = dstSize / (nElemBefore * nElemAfter);

//, value.data<scalar_t>(), plindex);
//      , orig_idx_beg,
//          idxSize, idxSize, headStride, liStride);

//
//
//
//
//  finish = std::chrono::high_resolution_clock::now();
//
//  xtotal += std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
//  if (pcnt % 100 == 0) {
//    std::cout << "xput: "
//              << xtotal/pcnt/1000
//              << "us" << std::endl;;
//  }



//  return src.xput_(linearIndex, value, accumulate);

//      thrust::counting_iterator<int64_t> full_index_iter(0);
//      thrust::for_each_n(policy, full_index_iter, fullIdxSize, amPutOp);

//    printf("< %lld %lld info.data[lindex]=%g value=%g\n",
//        n, lindex, info.data[lindex], value);

//    if (elementsBefore == 0 && elementsAfter == 0) {
// all dimensions indexed
//      int64_t* pindex = &sorted_idx;
//      int64_t linear_index = *pindex;
//      do {
//        info.data[offset] += *pvalue;
//        pindex++;
//        pvalue++;
//      } while (pindex != sorted_idx_end && *pindex == linear_index);
//      return 0;  // discarded
//    }
/*
    pvalue = psrc + orig_idx;
    for (int64_t e = 0; e < elementsBefore; ++e) {
      int64_t eoffset = beforeIndex[e];
      if (accumulate) {
        info.data[offset + eoffset] += *pvalue;
      } else {
        info.data[offset + eoffset] = *pvalue;
      }
      pvalue += elementsBefore;
    }

    pvalue = psrc + orig_idx * elementsAfter;
    for (int64_t e = 0; e < elementsAfter; ++e) {
      int64_t eoffset = afterIndex[e];
      if (accumulate) {
        info.data[offset + eoffset] += *pvalue;
      } else {
        info.data[offset + eoffset] = *pvalue;
      }
      pvalue++;
    }
*/

//    assert( < dstCopyDimSize);
//    int64_t nn = n / lindexStride;
//    nn = orig_idx_beg[n / lindexStride];
//    const int64_t srcIdx = orig_idx_beg[headIdx] + (lindex % lindexStride);


/*
    int64_t no = orig_idx_beg[n / lindexStride] * lindexStride + (n % lindexStride);
    const int64_t lindex = plindex[no];
    const int64_t headIdx = lindex / headStride;
    const int64_t shPos = lindex % headStride;
    const int64_t offset = headIdx * headStride + shPos;

    printf("%lld %lld %lld %lld %lld %lld %lld\n", n , no, lindex, headStride, headIdx, shPos, offset);



    const T value = p_source[no];
    T* address = info.data + offset;
//    info.data[lindex] += value;

*/

//__device__ __forceinline__ T operator()(int64_t& lindex) {
//    const T* pvalue = p_source + orig_idx;
//    const int64_t offset = indexToOffset<T, int64_t>(info, sorted_idx);

//printf("elementsBefore %lld elementsAfter %lld s=%lld o=%lld\n",
//    elementsBefore, elementsAfter, sorted_idx, orig_idx);

//    if (elementsBefore == 0 && elementsAfter == 0) {
// all dimensions indexed
//      int64_t* pindex = &sorted_idx;
//      int64_t linear_index = *pindex;
//      do {
//        info.data[offset] += *pvalue;
//        pindex++;
//        pvalue++;
//      } while (pindex != sorted_idx_end && *pindex == linear_index);
//      return 0;  // discarded
//    }


//    pvalue = psrc + orig_idx;
//    for (int64_t e = 0; e < elementsBefore; ++e) {
//      int64_t eoffset = beforeIndex[e];
//      if (accumulate) {
//        info.data[offset + eoffset] += *pvalue;
//      } else {
//        info.data[offset + eoffset] = *pvalue;
//      }
//      pvalue += elementsBefore;
//    }
//
//    pvalue = psrc + orig_idx * elementsAfter;
//    for (int64_t e = 0; e < elementsAfter; ++e) {
//      int64_t eoffset = afterIndex[e];
//      if (accumulate) {
//        info.data[offset + eoffset] += *pvalue;
//      } else {
//        info.data[offset + eoffset] = *pvalue;
//      }
//      pvalue++;
//    }


