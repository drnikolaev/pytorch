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
#include <c10/util/Exception.h>

#include <ATen/native/Indexing.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctions.h>

#include <THC/THCDeviceUtils.cuh>
#include <THC/THCTensorMathReduce.cuh>
#include <THC/THCTensorSort.cuh>
#include <THC/THCThrustAllocator.cuh>

#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <torch/csrc/utils/tensor_flatten.h>

#include <ATen/cpu/vec256/vec256.h>

namespace at { namespace native {

//Tensor& index_put_cuda_(Tensor& self, TensorList indices, const Tensor& value, bool accumulate) {
//  return self;
//}




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

static Tensor computeLinearIndex(const Tensor & src, TensorList indices) {
  auto strides = computeLinearStride(src);
  Type& longType = src.type().toScalarType(kLong);

  std::cerr << "src" << std::endl;
  print(std::cerr, src, 120);
  std::cerr << src.sizes() << std::endl << std::endl;

  // Compute the linear index by multiplying the indexing tensors by the
  // stride and summing them. All the indexing tensors have the same shape at
  // this point. We also compute the number of dimensions before and after that
  // are not being index.
  Tensor linearIndex;
  int64_t emptyBefore = 0, emptyAfter = 0, nElemBefore = 1, nElemAfter = 1;
  for (int64_t i = 0; i < src.dim(); i++) {
    if (indices[i].defined()) {
      // Cast index to the longType matching src's backend
      // This allows us to support ie indexing a cuda tensor with a cpu tensor
      Tensor index = (wrapIndexOnce(indices[i], i, src.size(i)) * strides[i]).toType(longType);

      std::cerr << i << " " << indices[i] << std::endl << std::endl;
      std::cerr << i << " " << strides[i] << std::endl << std::endl;

      std::cerr << "index" << std::endl;
      print(std::cerr, index, 120);
      std::cerr << index.sizes() << std::endl << std::endl;

      if (linearIndex.defined()) {
        linearIndex += index;
      } else {
        linearIndex = index;
      }
    } else if (linearIndex.defined()) {
      emptyAfter++;
      nElemAfter *= src.size(i);
    } else {
      emptyBefore++;
      nElemBefore *= src.size(i);
    }
  }

  std::cerr << "linearIndex" << std::endl;
  print(std::cerr, linearIndex, 120);
  std::cerr << linearIndex.sizes() << std::endl << std::endl;

  // Compute the linear indices for the parts of the tensor not being indexed
  //  Tensor beforeIndex;
  //  if (emptyBefore > 0) {
  //    auto index = at::arange(0, nElemBefore, longType) * strides[emptyBefore - 1];
  //    index = index.view(src.sizes().slice(0, emptyBefore));
  //    beforeIndex = unsqueezeN(index, 0, linearIndex.dim() + emptyAfter);
  //  }
  //  Tensor afterIndex;
  //  if (emptyAfter > 0) {
  //    auto index = at::arange(0, nElemAfter, longType);
  //    index = index.view(src.sizes().slice(src.dim() - emptyAfter, emptyAfter));
  //    afterIndex = unsqueezeN(index, linearIndex.dim() + emptyBefore, 0);
  //  }
  //
  //  // Sum with broadcasting to compute the full index
  //  linearIndex = unsqueezeN(linearIndex, emptyBefore, emptyAfter);
  //  if (beforeIndex.defined()) {
  //    linearIndex = linearIndex + beforeIndex;
  //  }
  //  if (afterIndex.defined()) {
  //    linearIndex = linearIndex + afterIndex;
  //  }

  return linearIndex;
}

static std::tuple<Tensor, Tensor> makeLinearIndex(Tensor self, TensorList orig) {
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
    std::cerr << "orig[i]  CPU " << i << std::endl;
    print(std::cerr, orig[i], 120);
    std::cerr << orig[i].sizes() << std::endl << std::endl;
  }
  for (size_t i = 0; i < indices.size(); i++) {
    if (!indices[i].defined()) {
      continue;
    }
    std::cerr << "indices[i]  CPU " << i << std::endl;
    print(std::cerr, indices[i], 120);
    std::cerr << indices[i].sizes() << std::endl << std::endl;
  }
  std::cerr << "self!" << std::endl;
  print(std::cerr, self, 120);
  std::cerr << self.sizes() << std::endl << std::endl;

  auto linearIndex = computeLinearIndex(self, indices);
  return std::make_tuple(self, linearIndex);
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

static std::string shapes_as_str(TensorList tensors) {
  std::ostringstream os;
  bool first = true;
  for (auto& tensor : tensors) {
    if (tensor.defined()) {
      if (!first) {
        os << ", ";
      }
      os << tensor.sizes();
      first = false;
    }
  }
  return os.str();
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

static AdvancedIndex make_info(Tensor self, TensorList orig) {
  checkIndexTensorTypes(orig);
  // first expand ByteTensor (boolean masks) into 1 or more LongTensors
  auto indices = expandByteTensors(self, orig);
  // next broadcast all index tensors together
  try {
    indices = expand_outplace(indices);
  } catch (std::exception& e) {
    AT_INDEX_ERROR("shape mismatch: indexing tensors could not be broadcast together"
                   " with shapes ", shapes_as_str(indices));
  }
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < (size_t)self.dim()) {
    indices.emplace_back();
  }
  // if the non-null indices are not all adjacent, transpose self and indices
  // together so that they're adjacent at the front
  if (!hasContiguousSubspace(indices)) {
    std::tie(self, indices) = transposeToFront(self, indices);
  }
  // Ensure indices are on the same device as self
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i].defined() && indices[i].device() != self.device()) {
      indices[i] = indices[i].to(self.device());
    }
  }
  return AdvancedIndex(self, indices);
}

static std::unique_ptr<TensorIterator> make_index_iterator(const AdvancedIndex& info) {
  auto builder = TensorIterator::Builder();
  builder.dont_compute_common_dtype();
  builder.add_output(Tensor(), &info.src.type());
  builder.add_input(info.src);
  for (auto& index : info.indices) {
    builder.add_input(index);
  }
  return builder.build();
}

static std::unique_ptr<TensorIterator> make_index_put_iterator(const AdvancedIndex& info, const Tensor& value) {
  if (!is_expandable_to(value.sizes(), info.src.sizes())) {
    AT_ERROR("shape mismatch: value tensor of shape ", value.sizes(),
        " cannot be broadcast to indexing result of shape ", info.src.sizes());
  }
  auto builder = TensorIterator::Builder();
  builder.dont_compute_common_dtype();
  builder.dont_resize_outputs();
  builder.add_output(info.src);
  builder.add_input(value, &info.src.type());
  for (auto& index : info.indices) {
    builder.add_input(index);
  }
  return builder.build();
}

//Tensor index(const Tensor & self, TensorList indices) {
//  if (indices.size() > (size_t)self.dim()) {
//    AT_INDEX_ERROR("too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
//  }
//
//  auto info = make_info(self, indices);
//  auto iter = make_index_iterator(info);
//  index_stub(iter->device_type(), *iter, info.indexed_sizes, info.indexed_strides);
//  return iter->output();
//}
//
//Tensor index_put(const Tensor & self, TensorList indices, const Tensor & value, bool accumulate) {
//  return self.clone().index_put_(indices, value, accumulate);
//}










// TODO: this is cut&paste
template <typename scalar_t>
__global__ void embedding_backward_kernel2(
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




Tensor & index_put_cuda__(Tensor & self, Tensor & index, const Tensor & values, bool accumulate) {
  if (values.numel() == 0 || values.numel() == 1) {
    return self;
  }

  // first expand ByteTensor (boolean masks) into 1 or more LongTensors
  //  auto indices__ = expandByteTensors2(self, indices_);
  //
  //  auto grad_arg = TensorArg(values, "grad", 1);
  //  auto indices_arg = TensorArg(indices__[0], "indices", 1);
  //  checkScalarType("index_put_cuda_", indices_arg, kLong);
  //  checkSameGPU("index_put_cuda_", grad_arg, indices_arg);

  int padding_idx = -1;

  std::vector<int64_t> dims;
  std::vector<int64_t> sizes;
  Tensor self_;
  //  Tensor indices;
  //  std::tie(self_, indices) = flattenToFront(self, indices__, dims, sizes);

  auto num_index = index.numel();
  auto grad = values.contiguous().view({num_index, values.size(-1)});
  int64_t stride = self_.stride(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto sorted_index = at::empty_like(index);
  auto orig_index = at::empty_like(index);
  using device_ptr = thrust::device_ptr<int64_t>;

  // Sort the inputs into sorted with the corresponding index; we
  // don't need a stable or multidimensional sort, so just use Thrust
  // directly
  {
    sorted_index.copy_(index);

    auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
    auto policy = thrust::cuda::par(allocator).on(stream);

    // Fill sortedOrigIndices with sequential index
    auto count_iter = thrust::counting_iterator<int64_t>(0);
    auto orig_data = device_ptr(orig_index.data<int64_t>());
    thrust::copy(policy, count_iter, count_iter + num_index, orig_data);

    // Sort; a stable sort is not required
    auto sorted_data = device_ptr(sorted_index.data<int64_t>());
    thrust::sort_by_key(policy, sorted_data, sorted_data + num_index, orig_data,
        ThrustLTOp<int64_t>());
  }

  dim3 grid(THCCeilDiv(num_index, (int64_t) 4), THCCeilDiv(stride, (int64_t) 128));
  dim3 block(32, 4);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.type(), "index_put_cuda_", [&] {
    embedding_backward_kernel2<<<grid, block, 0, stream>>>(
        sorted_index.data<int64_t>(),
            orig_index.data<int64_t>(),
            grad.data<scalar_t>(),
            self_.data<scalar_t>(),
            num_index,
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



//Tensor & index_put_cuda__(Tensor & self, Tensor & index, const Tensor & values, bool accumulate);

Tensor & index_put_cuda_(Tensor & self, TensorList indices, const Tensor & value, bool accumulate) {
  if (indices.size() > (size_t)self.dim()) {
    AT_INDEX_ERROR("too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  }
  if (accumulate) { //} && self.type().device_type() == kCUDA) {
    //    Tensor src, linearIndex, expandedValue;
    //    std::tie(src, linearIndex) = makeLinearIndex(self, indices);
    //    std::tie(expandedValue) = expand_inplace(linearIndex, value);
    //    return src.put_(linearIndex, expandedValue, true);


    Tensor src, linearIndex;
    std::tie(src, linearIndex) = makeLinearIndex(self, indices);

    return index_put_cuda__(self, linearIndex, value, accumulate);

  }
  //  auto info = make_info(self, indices);
  //  auto iter = make_index_put_iterator(info, value);
  //  index_put_stub(value.device().type(), *iter, info.indexed_sizes, info.indexed_strides, value, accumulate);
  return self;
}




}}

#if false



//Tensor & index_put_cuda_(Tensor & self, Tensor & index, const Tensor & values, bool accumulate) {
//  if (values.numel() == 0 || values.numel() == 1) {
//    return self;
//  }
//
//  // first expand ByteTensor (boolean masks) into 1 or more LongTensors
//  //  auto indices__ = expandByteTensors2(self, indices_);
//  //
//  //  auto grad_arg = TensorArg(values, "grad", 1);
//  //  auto indices_arg = TensorArg(indices__[0], "indices", 1);
//  //  checkScalarType("index_put_cuda_", indices_arg, kLong);
//  //  checkSameGPU("index_put_cuda_", grad_arg, indices_arg);
//
//  int padding_idx = -1;
//
//  std::vector<int64_t> dims;
//  std::vector<int64_t> sizes;
//  Tensor self_;
//  //  Tensor indices;
//  //  std::tie(self_, indices) = flattenToFront(self, indices__, dims, sizes);
//
//  auto num_index = index.numel();
//  auto grad = values.contiguous().view({num_index, values.size(-1)});
//  int64_t stride = self_.stride(0);
//  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//
//  auto sorted_index = at::empty_like(index);
//  auto orig_index = at::empty_like(index);
//  using device_ptr = thrust::device_ptr<int64_t>;
//
//  // Sort the inputs into sorted with the corresponding index; we
//  // don't need a stable or multidimensional sort, so just use Thrust
//  // directly
//  {
//    sorted_index.copy_(index);
//
//    auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
//    auto policy = thrust::cuda::par(allocator).on(stream);
//
//    // Fill sortedOrigIndices with sequential index
//    auto count_iter = thrust::counting_iterator<int64_t>(0);
//    auto orig_data = device_ptr(orig_index.data<int64_t>());
//    thrust::copy(policy, count_iter, count_iter + num_index, orig_data);
//
//    // Sort; a stable sort is not required
//    auto sorted_data = device_ptr(sorted_index.data<int64_t>());
//    thrust::sort_by_key(policy, sorted_data, sorted_data + num_index, orig_data,
//        ThrustLTOp<int64_t>());
//  }
//
//  dim3 grid(THCCeilDiv(num_index, (int64_t) 4), THCCeilDiv(stride, (int64_t) 128));
//  dim3 block(32, 4);
//
//  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.type(), "index_put_cuda_", [&] {
//    embedding_backward_kernel2<<<grid, block, 0, stream>>>(
//        sorted_index.data<int64_t>(),
//            orig_index.data<int64_t>(),
//            grad.data<scalar_t>(),
//            self_.data<scalar_t>(),
//            num_index,
//            stride,
//            padding_idx);
//  });
//  THCudaCheck(cudaGetLastError());
//  self.copy_(self_.view({sizes}));
//
//  //  std::cerr << "self  --->>" << std::endl;
//  //  print(std::cerr, self, 120);
//  //  std::cerr << self.sizes() << std::endl << std::endl;
//  return self;
//}



}} // at::native
#endif




#if false
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/TensorUtils.h>
#include <ATen/ExpandUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/TensorIterator.h>
#include <c10/util/Exception.h>

#include <ATen/native/Indexing.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctions.h>

#include <THC/THCDeviceUtils.cuh>
#include <THC/THCTensorMathReduce.cuh>
#include <THC/THCTensorSort.cuh>
#include <THC/THCThrustAllocator.cuh>

#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <torch/csrc/utils/tensor_flatten.h>

#include <ATen/cpu/vec256/vec256.h>

namespace at { namespace native {

#ifdef __HIP_PLATFORM_HCC__
static const int WARP_SIZE = 64;
#else
static const int WARP_SIZE = 32;
#endif

[[noreturn]]
static void invalid_mask2(const Tensor & self, int64_t idx, const Tensor & mask, int64_t maskIdx) {
  std::stringstream ss;
  ss << "The shape of the mask " << mask.sizes() << " at index " << maskIdx;
  ss << " does not match the shape of the indexed tensor " << self.sizes();
  ss << " at index " << idx;
  AT_INDEX_ERROR(ss.str());
}

static std::vector<Tensor> expandByteTensors2(const Tensor & self, TensorList indices) {
  // Expands byte tensors (masks) into the equivalent indexing by LongTensors
  std::vector<Tensor> result;
  for (auto & index : indices) {
    if (index.scalar_type() == kByte) {
      // The sizes of the ByteTensor mask must match the sizes of the
      // corresponding dimensions in self
      for (int64_t j = 0; j < index.dim(); j++) {
        int64_t srcIdx = result.size() + j;
        if (index.size(j) != self.size(srcIdx)) {
          invalid_mask2(self, srcIdx, index, j);
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

// Transposes the tensor and indices together so that all the non-null indices
// index the first k dimensions of the tensor. Returns the transposed tensor
// and the reordered indices. For example:
//  transposeToFront(tensor, {nullptr, a, nullptr, b})
// returns
//  tensor.permute([1, 3, 0, 2]), {a, b, nullptr, nullptr}
static std::tuple<Tensor, Tensor>
flattenToFront(const Tensor & self, TensorList indices, std::vector<int64_t>& dims,
    std::vector<int64_t>& sizes) {
  std::vector<Tensor> transposedIndices;
  dims.reserve(self.dim());
  sizes.resize(self.dim());
  for (int64_t i = 0; i < std::min<int64_t>(indices.size(), self.dim()); i++) {
    if (indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back(indices[i]);
    }
  }
  for (int64_t i = 0L; i < self.dim(); i++) {
    if (!indices[i].defined()) {
      dims.push_back(i);
    }
  }
  int64_t n1 = 1L;
  for (int64_t i = 0L; i < self.dim(); i++) {
    if (std::find(dims.begin(), dims.end(), i) == dims.end()) {
      dims.push_back(i);
    }
    if (i + 1 < self.dim()) {
      n1 *= self.size(i);
    }
    sizes[i] = self.size(i);
  }
  return std::make_tuple(self.view({n1,-1}), //permute(dims),//torch::utils::flatten_dense_tensors(transposedIndices));
      std::move(at::stack(transposedIndices, 0)));
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


Tensor & index_put_cuda_(Tensor & self, TensorList indices_, const Tensor & values, bool accumulate) {
  if (values.numel() == 0 || values.numel() == 1) {
    return self;
  }

  // first expand ByteTensor (boolean masks) into 1 or more LongTensors
//  auto indices__ = expandByteTensors2(self, indices_);
//
//  auto grad_arg = TensorArg(values, "grad", 1);
//  auto indices_arg = TensorArg(indices__[0], "indices", 1);
//  checkScalarType("index_put_cuda_", indices_arg, kLong);
//  checkSameGPU("index_put_cuda_", grad_arg, indices_arg);

  int padding_idx = -1;

  std::vector<int64_t> dims;
  std::vector<int64_t> sizes;
  Tensor self_;
//  Tensor indices;
//  std::tie(self_, indices) = flattenToFront(self, indices__, dims, sizes);

  auto num_index = index.numel();
  auto grad = values.contiguous().view({num_index, values.size(-1)});
  int64_t stride = self_.stride(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto sorted_index = at::empty_like(index);
  auto orig_index = at::empty_like(index);
  using device_ptr = thrust::device_ptr<int64_t>;

  // Sort the inputs into sorted with the corresponding index; we
  // don't need a stable or multidimensional sort, so just use Thrust
  // directly
  {
    sorted_index.copy_(index);

    auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
    auto policy = thrust::cuda::par(allocator).on(stream);

    // Fill sortedOrigIndices with sequential index
    auto count_iter = thrust::counting_iterator<int64_t>(0);
    auto orig_data = device_ptr(orig_index.data<int64_t>());
    thrust::copy(policy, count_iter, count_iter + num_index, orig_data);

    // Sort; a stable sort is not required
    auto sorted_data = device_ptr(sorted_index.data<int64_t>());
    thrust::sort_by_key(policy, sorted_data, sorted_data + num_index, orig_data,
        ThrustLTOp<int64_t>());
  }

  dim3 grid(THCCeilDiv(num_index, (int64_t) 4), THCCeilDiv(stride, (int64_t) 128));
  dim3 block(32, 4);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.type(), "index_put_cuda_", [&] {
    embedding_backward_kernel<<<grid, block, 0, stream>>>(
        sorted_index.data<int64_t>(),
            orig_index.data<int64_t>(),
            grad.data<scalar_t>(),
            self_.data<scalar_t>(),
            num_index,
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





//Tensor & index_put_cuda_(Tensor & self, TensorList indices_, const Tensor & values, bool accumulate) {
//  if (values.numel() == 0 || values.numel() == 1) {
//    return self;
//  }
//
//// first expand ByteTensor (boolean masks) into 1 or more LongTensors
//  auto indices__ = expandByteTensors2(self, indices_);
//
//  auto grad_arg = TensorArg(values, "grad", 1);
//  auto indices_arg = TensorArg(indices__[0], "indices", 1);
//  checkScalarType("index_put_cuda_", indices_arg, kLong);
//  checkSameGPU("index_put_cuda_", grad_arg, indices_arg);
//
//  int padding_idx = -1;
//
//  std::vector<int64_t> dims;
//  std::vector<int64_t> sizes;
//  Tensor self_;
//  Tensor indices;
//  std::tie(self_, indices) = flattenToFront(self, indices__, dims, sizes);
//
//  auto num_indices = indices[0].numel();
//  auto grad = values.contiguous().view({num_indices, values.size(-1)});
//  int64_t stride = self_.stride(0);
//  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//
//  auto sorted_indices = at::empty_like(indices);
//  auto orig_indices = at::empty_like(indices);
//  using device_ptr = thrust::device_ptr<int64_t>;
//
//  // Sort the inputs into sorted with the corresponding indices; we
//  // don't need a stable or multidimensional sort, so just use Thrust
//  // directly
//  {
//    sorted_indices.copy_(indices);
//
//    auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
//    auto policy = thrust::cuda::par(allocator).on(stream);
//
//    // Fill sortedOrigIndices with sequential indices
//    auto count_iter = thrust::counting_iterator<int64_t>(0);
//    auto orig_data = device_ptr(orig_indices.data<int64_t>());
//    thrust::copy(policy, count_iter, count_iter + num_indices, orig_data);
//
//    // Sort; a stable sort is not required
//    auto sorted_data = device_ptr(sorted_indices.data<int64_t>());
//    thrust::sort_by_key(policy, sorted_data, sorted_data + num_indices, orig_data,
//        ThrustLTOp<int64_t>());
//  }
//
//  dim3 grid(THCCeilDiv(num_indices, (int64_t) 4), THCCeilDiv(stride, (int64_t) 128));
//  dim3 block(32, 4);
//
//  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.type(), "index_put_cuda_", [&] {
//    embedding_backward_kernel<<<grid, block, 0, stream>>>(
//      sorted_indices.data<int64_t>(),
//      orig_indices.data<int64_t>(),
//      grad.data<scalar_t>(),
//      self_.data<scalar_t>(),
//      num_indices,
//      stride,
//      padding_idx);
//  });
//  THCudaCheck(cudaGetLastError());
//  self.copy_(self_.view({sizes}));
//
////  std::cerr << "self  --->>" << std::endl;
////  print(std::cerr, self, 120);
////  std::cerr << self.sizes() << std::endl << std::endl;
//  return self;
//}

}}  // namespace at::native
#endif
