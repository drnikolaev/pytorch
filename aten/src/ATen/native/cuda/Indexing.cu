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

#ifdef __HIP_PLATFORM_HCC__
static const int WARP_SIZE = 64;
#else
static const int WARP_SIZE = 32;
#endif

DEFINE_DISPATCH(index_stub);
DEFINE_DISPATCH(index_put_stub);

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
//    std::cerr << "orig[i]  GPU " << i << std::endl;
//    print(std::cerr, orig[i], 120);
//    std::cerr << orig[i].sizes() << std::endl << std::endl;
  }
  for (size_t i = 0; i < indices.size(); i++) {
    if (!indices[i].defined()) {
      continue;
    }
//    std::cerr << "indices[i]  GPU " << i << std::endl;
//    print(std::cerr, indices[i], 120);
//    std::cerr << indices[i].sizes() << std::endl << std::endl;
  }
//  std::cerr << "self!  GPU " << std::endl;
//  print(std::cerr, self, 120);
//  std::cerr << self.sizes() << std::endl << std::endl;

//  auto linearIndex = computeLinearIndex(self, indices);
  auto strides = computeLinearStride(self);

//  std::cerr << "self" << std::endl;
//  print(std::cerr, self, 120);
//  std::cerr << self.sizes() << std::endl << std::endl;

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

//      std::cerr << i << "-th index  " << std::endl << indices[i] << std::endl;
//      std::cerr << i << "-th stride " << std::endl << strides[i] << std::endl << std::endl;
//
//      std::cerr << "index" << std::endl;
//      print(std::cerr, index, 120);
//      std::cerr << index.sizes() << std::endl << std::endl;

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

//    std::cerr << "beforeIndex  --->>" << std::endl;
//    print(std::cerr, beforeIndex, 120);
//    std::cerr << beforeIndex.sizes() << " emptyBefore: " << emptyBefore
//              << " nElemBefore: " << nElemBefore<< std::endl << std::endl;

  }
  Tensor afterIndex;
  if (emptyAfter > 0) {
    auto index = at::arange(0, nElemAfter, self.options().dtype(kLong));
    index = index.view(self.sizes().slice(self.dim() - emptyAfter, emptyAfter));
    afterIndex = unsqueezeN(index, linearIndex.dim() + emptyBefore, 0);

//    std::cerr << "afterIndex  --->>" << std::endl;
//    print(std::cerr, afterIndex, 120);
//    std::cerr << afterIndex.sizes() << " emptyAfter: " << emptyAfter
//              << " nElemAfter: " << nElemAfter<< std::endl << std::endl;

  }

//  std::cerr << "linearIndex  --->>" << std::endl;
//  print(std::cerr, linearIndex, 120);
//  std::cerr << linearIndex.sizes() << std::endl << std::endl;

  // Sum with broadcasting to compute the full index
  linearIndex = unsqueezeN(linearIndex, emptyBefore, emptyAfter);
  if (beforeIndex.defined()) {
    linearIndex = linearIndex + beforeIndex;
  }
  if (afterIndex.defined()) {
    linearIndex = linearIndex + afterIndex;
  }

//  std::cerr << "linearIndex  <<---" << std::endl;
//  print(std::cerr, linearIndex, 120);
//  std::cerr << linearIndex.sizes() << std::endl << std::endl;

  return std::make_tuple(self, linearIndex, emptyBefore, emptyAfter);
}

static bool all_strides_match(TensorList tensors) {
  AT_ASSERT(tensors.size() >= 1);
  auto strides = tensors[0].strides();
  for (auto& tensor : tensors.slice(1)) {
    if (!strides.equals(tensor.strides())) {
      return false;
    }
  }
  return true;
}

struct AdvancedIndex {
  AdvancedIndex(const Tensor& src, TensorList indices);

  Tensor src;
  std::vector<Tensor> indices;
  DimVector indexed_sizes;
  DimVector indexed_strides;
  int64_t dims_before;
  int64_t dims_after;
};

// Replace indexed dimensions in src with stride 0 and the size of the result tensor.
// The offset in these dimensions is computed by the kernel using the index tensor's
// values and the stride of src. The new shape is not meaningful. It's used to make
// the shape compatible with the result tensor.
static Tensor restride_src(const Tensor& src, int64_t dims_before, int64_t dims_indexed,
    IntArrayRef replacement_shape) {
  auto shape = DimVector(src.sizes());
  auto strides = DimVector(src.strides());
  int64_t end = dims_before + dims_indexed;
  shape.erase(shape.begin() + dims_before, shape.begin() + end);
  strides.erase(strides.begin() + dims_before, strides.begin() + end);
  shape.insert(shape.begin() + dims_before, replacement_shape.begin(), replacement_shape.end());
  strides.insert(strides.begin() + dims_before, replacement_shape.size(), 0);
  return src.as_strided(shape, strides);
}

// Add dimensions of size 1 to an index tensor so that it can be broadcast to the result
// shape and iterated over element-wise like the result tensor and the restrided src.
static Tensor reshape_indexer(const Tensor& index, int64_t dims_before, int64_t dims_after) {
  auto orig_shape = index.sizes();
  auto shape = DimVector();
  shape.append(dims_before, 1);
  shape.append(orig_shape.begin(), orig_shape.end());
  shape.append(dims_after, 1);
  return index.reshape(shape);
}

AdvancedIndex::AdvancedIndex(const Tensor& src, TensorList indices_list)
{
  int64_t element_size_bytes = src.element_size();
  int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
  IntArrayRef replacement_shape;
  for (size_t dim = 0; dim < indices_list.size(); dim++) {
    if (!indices_list[dim].defined()) {
      if (dims_indexed == 0) {
        dims_before++;
      } else {
        dims_after++;
      }
    } else {
      dims_indexed++;
      replacement_shape = indices_list[dim].sizes();
      indexed_sizes.push_back(src.size(dim));
      indexed_strides.push_back(src.stride(dim) * element_size_bytes);
    }
  }

  // Check if the indexed subspace contains a dim of size 0, but the replacement
  // shape does not. This implies that an index is out of bounds, because there
  // is no number that's a valid index for an empty tensor. Normally, out of
  // bounds is handled in the indexing kernel, but this case fails earlier in
  // restride_src with an unhelpful error message.
  if (std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) != indexed_sizes.end() &&
      std::find(replacement_shape.begin(), replacement_shape.end(), 0) == replacement_shape.end()) {
    AT_INDEX_ERROR("index is out of bounds for dimension with size 0");
  }

  this->dims_before = dims_before;
  this->dims_after = dims_after;
  this->src = restride_src(src, dims_before, dims_indexed, replacement_shape);

  for (auto& index : indices_list) {
    if (index.defined()) {
      indices.push_back(reshape_indexer(index, dims_before, dims_after));
    }
  }

  // For CUDA tensors, force all index tensors to have the same striding to
  // simplify the CUDA kernel.
  if (indices.size() >= 2 && this->src.type().device_type() == kCUDA) {
    if (!all_strides_match(indices)) {
      for (size_t i = 0; i < indices.size(); i++) {
        indices[i] = indices[i].contiguous();
      }
    }
  }
}



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

#if false
Tensor & index_put_cuda_(Tensor & self, TensorList indices_, const Tensor & values, bool accumulate) {
  if (values.numel() == 0 || values.numel() == 1) {
    return self;
  }

// first expand ByteTensor (boolean masks) into 1 or more LongTensors
  auto indices__ = expandByteTensors2(self, indices_);

  auto grad_arg = TensorArg(values, "grad", 1);
  auto indices_arg = TensorArg(indices__[0], "indices", 1);
  checkScalarType("index_put_cuda_", indices_arg, kLong);
  checkSameGPU("index_put_cuda_", grad_arg, indices_arg);

  int padding_idx = -1;

  std::vector<int64_t> dims;
  std::vector<int64_t> sizes;
  Tensor self_;
  Tensor indices;
  std::tie(self_, indices) = flattenToFront(self, indices__, dims, sizes);

  auto num_indices = indices[0].numel();
  auto grad = values.contiguous().view({num_indices, values.size(-1)});
  int64_t stride = self_.stride(0);
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
          self_.data<scalar_t>(),
          num_indices,
          stride,
          padding_idx);
  });
  THCudaCheck(cudaGetLastError());
  self.copy_(self_.view({sizes}));

//  std::cerr << "self  --->>" << std::endl;
//  print(std::cerr, self, 120);
//  std::cerr << self.sizes() << std::endl << std::endl;
  return self;
}
#endif



/*
static void THCTensor_(sort_indices)(THCState *state, THCudaLongTensor *index, THCTensor *src) {
  THCThrustAllocator thrustAlloc(state);

  auto index_iter = thrust::device_ptr<int64_t>(THCudaLongTensor_data(state, index));
  auto src_iter = thrust::device_ptr<scalar_t>(THCTensor_(data)(state, src));
  auto numel = THCTensor_(numel)(state, src);

  thrust::sort_by_key(
      thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
      index_iter, index_iter + numel,
      src_iter, ThrustLTOp<int64_t>());
}


// wrap indices so to replace negative indices
THCudaLongTensor* sorted_index = THCudaLongTensor_new(state);
THCudaLongTensor_resizeAs(state, sorted_index, index);
THC_pointwiseApply2<int64_t, int64_t>(state, sorted_index, index, WrapIndexOp(dstSize));

THCTensor* sorted_src = THCTensor_(newClone)(state, src);
THCTensor_(sort_indices)(state, sorted_index, sorted_src);

dispatchTakePut<scalar_t, TensorPutAccumulateOp>(state, dst, sorted_src, sorted_index);

*/

struct WrapIndexOp  : thrust::unary_function<int64_t, int64_t> {
  WrapIndexOp(int64_t size) : size(size) {}

  __device__ __forceinline__ int64_t operator()(int64_t idx) {
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

template <typename T, typename IndexType>
struct TensorPutOp {
  TensorPutOp(at::cuda::detail::TensorInfo<T, IndexType> info, IndexType numel, int64_t*, int64_t*)
      : info(info), numel(numel) {}

  __device__ __forceinline__ void operator()(T* value, int64_t* index) {
    auto offset = indexToOffset(info, *index, numel);
    info.data[offset] = *value;
  }

  const at::cuda::detail::TensorInfo<T, IndexType> info;
  IndexType numel;
};

//template <typename T, typename IndexType, int Dims>
//struct TensorPutAccumulateOp {
//  TensorPutAccumulateOp(TensorInfo<T, IndexType> info, IndexType numel, int64_t* start, int64_t* end)
//      : info(info), numel(numel), start(start), end(end) {}
//
//  __device__ __forceinline__ void operator()(T* value, int64_t* index) {
//    if (index == start || *index != *(index - 1)) {
//      int64_t linear_index = *index;
//      auto offset = indexToOffset<Dims>(info, linear_index, numel);
//      do {
//        info.data[offset] = THCNumerics<T>::add(info.data[offset], *value);
//        index++;
//        value++;
//      } while (index != end && *index == linear_index);
//    }
//  }

template <typename T>
struct TensorPutAccumulateOp : thrust::binary_function<int64_t, T, T> {
  TensorPutAccumulateOp(at::cuda::detail::TensorInfo<T, int64_t> info, int64_t numel,
      int64_t* start, int64_t* end)
    : info(info), numel(numel), start(start), end(end) {}

//    __device__ __forceinline__ bool operator()(thrust::tuple<signed long, int64_t> ci) {
//      entry = ci.get<0>();
//      int64_t index = ci.get<1>();
////      printf("CIT %ld %lld\n", entry, index);
//      return true;
//    }

    __device__ __forceinline__ T operator()(int64_t& index, T& value) {
      int64_t* pindex = &index;
      T* pvalue = &value;

      //printf("???? %lld %g\n", *pindex, *pvalue);

      if (pindex == start || *pindex != *(pindex - 1)) {
        int64_t linear_index = *pindex;
        int64_t offset = indexToOffset<T, int64_t>(info, linear_index, numel);

//        printf("~~~~~ %ld %lld %lld %g %g\n", entry, offset, linear_index, value, outptr[offset]);

        //int i = 0;
        do {
          info.data[offset] += *pvalue; //THCNumerics<T>::add(info.data[offset], *value);
//          outptr[offset] += *pvalue; //THCNumerics<T>::add(info.data[offset], *value);
          pindex++;
          pvalue++;
          //++i;
        } while (pindex != end && *pindex == linear_index);
      }
//      else {
//        printf("!!! %lld %g\n", index, value);
//      }
    return 0;
  }

  const at::cuda::detail::TensorInfo<T, int64_t> info;
  int64_t numel;
  int64_t* start;
  int64_t* end;
//  signed long entry;
//  T* outptr;
};


Tensor & xput_cuda_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) {
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
      auto dst_iter = thrust::make_discard_iterator(); // thrust::device_ptr<scalar_t>(self.data<scalar_t>());

      if (accumulate) {
        WrapIndexOp wrapIndexOp(dstSize);
        thrust::transform(
            policy,
            index_iter, index_iter + idxSize, sorted_iter, wrapIndexOp);

        thrust::sort_by_key(
            policy,
            sorted_iter, sorted_iter + idxSize, src_iter, ThrustLTOp<int64_t>());

        auto self_info = cuda::detail::getTensorInfo<scalar_t, int64_t>(self);
        //self_info.collapseDims();
        int64_t* raw_sorted_iter = sorted_index.data<int64_t>();
        TensorPutAccumulateOp<scalar_t> putAccumulateOp(self_info,
            dstSize, raw_sorted_iter, raw_sorted_iter + idxSize);

        thrust::transform(
            policy,
            sorted_iter, sorted_iter + idxSize, src_iter, dst_iter, putAccumulateOp);
      } else {
  //    TensorPutOp<float, int64_t, 25> putOp();
  //    thrust::transform(policy,
  //        index_iter, index_iter + numel, sorted_iter, dst_iter, pop);
      }
    });




#if false
  std::cout << "-> sorted_index GPU" << std::endl;
  print(std::cout, sorted_index, 120);
  std::cout << sorted_index.sizes() << std::endl << std::endl;


  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "xput_cuda_", [&] {
      auto src_iter = thrust::device_ptr<scalar_t>(source.data<scalar_t>());
      auto dst_iter = thrust::device_ptr<scalar_t>(self.data<scalar_t>());

      WrapIndexOp wrapIndexOp(dstSize);

      auto index_iter_end = index_iter;
      thrust::advance(index_iter_end, idxSize);

      thrust::transform(policy, index_iter, index_iter_end, sorted_iter, wrapIndexOp);
  });

  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "xput_cuda_", [&] {
    auto src_iter = thrust::device_ptr<scalar_t>(source.data<scalar_t>());
    auto dst_iter = thrust::device_ptr<scalar_t>(self.data<scalar_t>());

    thrust::sort_by_key(
        policy,
        sorted_iter, sorted_iter + idxSize, src_iter, ThrustLTOp<int64_t>());
  });

//  std::cout << "source GPU" << std::endl;
//  print(std::cout, source, 120);
//  std::cout << source.sizes() << std::endl << std::endl;
//
//  std::cout << "self GPU" << std::endl;
//  print(std::cout, self, 120);
//  std::cout << self.sizes() << std::endl << std::endl;

//  auto self_infoo = cuda::detail::getTensorInfo<float, int64_t>(self);
//  self_infoo.collapseDims();

  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "xput_cuda_", [&] {
    auto src_iter = thrust::device_ptr<scalar_t>(source.data<scalar_t>());
    auto dst_iter = thrust::make_discard_iterator(); // thrust::device_ptr<scalar_t>(self.data<scalar_t>());

    auto self_info = cuda::detail::getTensorInfo<scalar_t, int64_t>(self);
    self_info.collapseDims();
    int64_t* raw_sorted_iter = sorted_index.data<int64_t>();
    TensorPutAccumulateOp<scalar_t> putAccumulateOp(self_info,
        dstSize, raw_sorted_iter, raw_sorted_iter + idxSize);//, self.data<scalar_t>());
//    thrust::identity<int64_t> identity;
//    auto count_iter = thrust::counting_iterator<int64_t>(0);

//    thrust::zip_iterator<thrust::tuple<
//        thrust::counting_iterator<int64_t>,
//        thrust::device_ptr<int64_t>
//        >>
//        count_index_itr(thrust::make_tuple(count_iter, sorted_iter));

//    thrust::transform_if(
//        policy,
//        sorted_iter, sorted_iter + idxSize, src_iter,
//        count_index_itr, dst_iter, putAccumulateOp, putAccumulateOp);

    thrust::transform(
        policy,
        sorted_iter, sorted_iter + idxSize, src_iter, dst_iter, putAccumulateOp);

  });

//  std::cout << "---> source GPU" << std::endl;
//  print(std::cout, source, 120);
//  std::cout << source.sizes() << std::endl << std::endl;
//
//  std::cout << "---> self GPU" << std::endl;
//  print(std::cout, self, 120);
//  std::cout << self.sizes() << std::endl << std::endl;

#endif



  return self;
}


Tensor & index_put_cuda_(Tensor & self, TensorList indices, const Tensor & value, bool accumulate) {
//  return at::native::index_put_(self, indices, value, accumulate);
  if (indices.size() > (size_t)self.dim()) {
    AT_INDEX_ERROR("too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  }
  if (accumulate) { // && self.type().device_type() == kCUDA) {
    Tensor src, linearIndex, expandedValue;
    int64_t emptyBefore, emptyAfter;
    std::tie(src, linearIndex, emptyBefore, emptyAfter) = makeLinearIndex(self, indices);


    /*
        Tensor effectiveIndex = linearIndex.select(emptyAfter, 0L);
        Tensor effectiveValue = value.select(emptyAfter, 0L);

        std::cerr << "effectiveIndex" << std::endl;
        print(std::cerr, effectiveIndex, 120);
        std::cerr << effectiveIndex.sizes() << std::endl << std::endl;
        std::cerr << "effectiveValue" << std::endl;
        print(std::cerr, effectiveValue, 120);
        std::cerr << effectiveValue.sizes() << std::endl << std::endl;

        self.xput_(effectiveIndex, effectiveValue, true);
    */
    std::tie(expandedValue) = expand_inplace(linearIndex, value);
    return src.xput_(linearIndex, expandedValue, true);
  }
  return self;
}

}}


//    std::tie(expandedValue) = expand_inplace(linearIndex, value);
//    std::tie(expandedValue) = expand_inplace(effectiveIndex, effectiveValue);
//    index.view(src.sizes().slice(src.dim() - emptyAfter, emptyAfter));
//    std::cerr << "expandedValue" << std::endl;
//    print(std::cerr, expandedValue, 120);
//    std::cerr << expandedValue.sizes() << std::endl << std::endl;
//    self.view(self.sizes().slice(0L, src.dim() - emptyAfter)).put_(linearIndex, expandedValue, true);
//    self.put_(effectiveIndex, expandedValue, true);
//    return src.put_(linearIndex, value, true);
