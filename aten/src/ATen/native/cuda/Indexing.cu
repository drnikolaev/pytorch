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
#include <torch/csrc/utils/tensor_flatten.h>

#include <ATen/cpu/vec256/vec256.h>

namespace at { namespace native {

//DEFINE_DISPATCH(index_stub);
//DEFINE_DISPATCH(index_put_stub);

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

static Tensor wrapIndexOnce(const Tensor & index, int64_t dim, int64_t dim_size) {
  if (index.numel() != 0) {
    auto max_idx = index.max().item<int64_t>();
    auto min_idx = index.min().item<int64_t>();
    if (max_idx >= dim_size) {
      AT_INDEX_ERROR("index ", max_idx, " is out of bounds for dimension ", dim, " with size ", dim_size);
    }
    if (min_idx < -dim_size) {
      AT_INDEX_ERROR("index ", min_idx, " is out of bounds for dimension ", dim, " with size ", dim_size);
    }
  }
  return index.remainder(dim_size);
}

static std::tuple<Tensor, Tensor, int64_t, int64_t> makeLinearIndex(Tensor self, TensorList orig) {
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

  for (size_t i = 0; i < orig.size(); i++) {
    if (!orig[i].defined()) {
      continue;
    }
  }
  for (size_t i = 0; i < indices.size(); i++) {
    if (!indices[i].defined()) {
      continue;
    }
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
      // Cast index to the longType matching self's backend
      // This allows us to support ie indexing a cuda tensor with a cpu tensor
      Tensor index = (wrapIndexOnce(indices[i], i, self.size(i)) * strides[i]).to(kLong);
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

  // Compute the linear indices for the parts of the tensor not being indexed
  Tensor beforeIndex;
  if (emptyBefore > 0) {
    auto index = at::arange(0, nElemBefore, self.options().dtype(kLong)) * strides[emptyBefore - 1];
    index = index.view(self.sizes().slice(0, emptyBefore));
    beforeIndex = unsqueezeN(index, 0, linearIndex.dim() + emptyAfter);
  }
  Tensor afterIndex;
  if (emptyAfter > 0) {
    auto index = at::arange(0, nElemAfter, self.options().dtype(kLong));
    index = index.view(self.sizes().slice(self.dim() - emptyAfter, emptyAfter));
    afterIndex = unsqueezeN(index, linearIndex.dim() + emptyBefore, 0);
  }

  // Sum with broadcasting to compute the full index
  linearIndex = unsqueezeN(linearIndex, emptyBefore, emptyAfter);
  if (beforeIndex.defined()) {
    linearIndex = linearIndex + beforeIndex;
  }
  if (afterIndex.defined()) {
    linearIndex = linearIndex + afterIndex;
  }

  return std::make_tuple(self, linearIndex, emptyBefore, emptyAfter);
}

struct WrapIndexOp  : thrust::unary_function<int64_t, int64_t> {
  WrapIndexOp(int64_t size) : size(size) {}

  __device__ __forceinline__ int64_t operator()(int64_t idx) {
//    if(!(idx < size && idx >= -size)) {
//      printf("!!!!!!!!!!!! %lld %lld\n", idx, size);
//    }
    assert(idx < size && idx >= -size);
    return idx < 0 ? idx + size : idx;
  }

  int64_t size;
};

// Translate a linear index for the apply to a T* offset;
// specialized on `Dims` to reduce nvcc compilation time
template <typename T, typename IndexType>
struct IndexToOffset {
  static __host__ __device__ IndexType get(
      IndexType linearId,
      const at::cuda::detail::TensorInfo<T, IndexType>& info) {
    IndexType offset = 0;
    for (int i = info.dims - 1; i > 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;
      linearId /= info.sizes[i];
    }
    return offset + linearId * info.strides[0];
  }
};

template <typename T, typename IndexType>
__device__ __forceinline__ IndexType indexToOffset(
    const at::cuda::detail::TensorInfo<T, IndexType>& info,
    int64_t index,
    IndexType size)
{
  IndexType linearIndex = static_cast<IndexType>(index);

//  if (!(linearIndex < size && linearIndex >= -size)) {
//    printf("*************** %lld %lld\n", linearIndex, size);
//  }

  assert(linearIndex < size && linearIndex >= -size);
  if (linearIndex < 0) {
    linearIndex += size;
  }
  return IndexToOffset<T, IndexType>::get(linearIndex, info);
}

template <typename T>
struct TensorPutOp : thrust::binary_function<int64_t, T, T> {
  TensorPutOp(at::cuda::detail::TensorInfo<T, int64_t> info, int64_t numel, int64_t*, int64_t*)
      : info(info), numel(numel) {}

  __device__ __forceinline__ T operator()(int64_t& index, T& value) {
    auto offset = indexToOffset<T, int64_t>(info, index, numel);
    info.data[offset] = value;
    return 0;  // discarded
  }

  const at::cuda::detail::TensorInfo<T, int64_t> info;
  int64_t numel;
};

template <typename T>
struct TensorPutAccumulateOp : thrust::binary_function<int64_t, T, T> {
  TensorPutAccumulateOp(at::cuda::detail::TensorInfo<T, int64_t> info, int64_t numel,
      int64_t* start, int64_t* end)
    : info(info), numel(numel), start(start), end(end) {}

  __device__ __forceinline__ T operator()(int64_t& index, T& value) {
    int64_t* pindex = &index;
    T* pvalue = &value;
    if (pindex == start || *pindex != *(pindex - 1)) {
      int64_t linear_index = *pindex;
      int64_t offset = indexToOffset<T, int64_t>(info, linear_index, numel);
      do {
        info.data[offset] += *pvalue;
        pindex++;
        pvalue++;
      } while (pindex != end && *pindex == linear_index);
    }
    return 0;  // discarded
  }

  const at::cuda::detail::TensorInfo<T, int64_t> info;
  int64_t numel;
  int64_t* start;
  int64_t* end;
};

Tensor & xput_cuda_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate,
    int64_t emptyBefore, int64_t emptyAfter) {
  auto sorted_index = at::empty_like(index);
  auto orig_index = at::empty_like(index);
  int64_t dstSize = self.numel();
  int64_t idxSize = index.numel();
  orig_index.copy_(index);

  auto index_iter = thrust::device_ptr<int64_t>(orig_index.data<int64_t>());
  auto sorted_iter = thrust::device_ptr<int64_t>(sorted_index.data<int64_t>());
  auto numel = source.numel();

  if (numel != idxSize) {
    AT_INDEX_ERROR("src should have the same number of elements as index: ",
        numel, " != ", idxSize);
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "xput_cuda_", [&] {
    auto src_iter = thrust::device_ptr<scalar_t>(source.data<scalar_t>());
    auto dst_iter = thrust::make_discard_iterator(); // we directly write to info.data
    auto self_info = cuda::detail::getTensorInfo<scalar_t, int64_t>(self);
    //self_info.collapseDims();
    int64_t* raw_sorted_iter = sorted_index.data<int64_t>();

    if (accumulate) {
      WrapIndexOp wrapIndexOp(dstSize);
      thrust::transform(
          policy,
          index_iter, index_iter + idxSize, sorted_iter, wrapIndexOp);

      thrust::sort_by_key(
          policy,
          sorted_iter, sorted_iter + idxSize, src_iter, ThrustLTOp<int64_t>());

      TensorPutAccumulateOp<scalar_t> putAccumulateOp(self_info,
          dstSize, raw_sorted_iter, raw_sorted_iter + idxSize);

      thrust::transform(
          policy,
          sorted_iter, sorted_iter + idxSize, src_iter, dst_iter, putAccumulateOp);
    } else {
      TensorPutOp<scalar_t> putOp(self_info,
          dstSize, raw_sorted_iter, raw_sorted_iter + idxSize);

      thrust::transform(
          policy,
          index_iter, index_iter + idxSize, src_iter, dst_iter, putOp);
    }
  });

  return self;
}

Tensor & index_put_cuda_(Tensor & self, TensorList indices, const Tensor & value, bool accumulate) {
  if (indices.size() > (size_t)self.dim()) {
    AT_INDEX_ERROR("too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  }

  Tensor src, linearIndex, expandedValue;
  int64_t emptyBefore, emptyAfter;
  std::tie(src, linearIndex, emptyBefore, emptyAfter) = makeLinearIndex(self, indices);
  std::tie(expandedValue) = expand_inplace(linearIndex, value);
  return src.xput_(linearIndex, expandedValue, accumulate, emptyBefore, emptyAfter);
}

}}
