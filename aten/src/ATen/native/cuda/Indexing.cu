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

//DEFINE_DISPATCH(index_stub);
//DEFINE_DISPATCH(index_put_stub);
//
//void index_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride);
//void index_put_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate);
//
//
//void index(TensorIterator &, IntArrayRef indexed_sizes, IntArrayRef indexed_strides);
//using index_put_fn = void(*)(TensorIterator &, IntArrayRef indexed_sizes, IntArrayRef indexed_strides, bool accumulate);
//
//
//static std::string shapes_as_str(TensorList tensors) {
//  std::ostringstream os;
//  bool first = true;
//  for (auto& tensor : tensors) {
//    if (tensor.defined()) {
//      if (!first) {
//        os << ", ";
//      }
//      os << tensor.sizes();
//      first = false;
//    }
//  }
//  return os.str();
//}
//
//
//static bool all_strides_match(TensorList tensors) {
//  AT_ASSERT(tensors.size() >= 1);
//  auto strides = tensors[0].strides();
//  for (auto& tensor : tensors.slice(1)) {
//    if (!strides.equals(tensor.strides())) {
//      return false;
//    }
//  }
//  return true;
//}
//
//static bool hasContiguousSubspace(TensorList tl) {
//  // true if all the non-null tensors are adjacent
//  auto isDefined = [](const Tensor & tensor){ return tensor.defined(); };
//  auto isNull = [](const Tensor & tensor){ return !tensor.defined(); };
//  auto start = std::find_if(tl.begin(), tl.end(), isDefined);
//  auto stop = std::find_if(tl.rbegin(), tl.rend(), isDefined);
//  auto it = std::find_if(start, stop.base(), isNull);
//  return it == stop.base();
//}

// Transposes the tensor and indices together so that all the non-null indices
// index the first k dimensions of the tensor. Returns the transposed tensor
// and the reordered indices. For example:
//  transposeToFront(tensor, {nullptr, a, nullptr, b})
// returns
//  tensor.permute([1, 3, 0, 2]), {a, b, nullptr, nullptr}
//static std::tuple<Tensor, std::vector<Tensor>>
//transposeToFront(Tensor self, TensorList indices) {
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
//
//struct AdvancedIndex {
//  AdvancedIndex(const Tensor& src, TensorList indices);
//
//  Tensor src;
//  std::vector<Tensor> indices;
//  DimVector indexed_sizes;
//  DimVector indexed_strides;
//  int64_t dims_before;
//  int64_t dims_after;
//};
//
//// Replace indexed dimensions in src with stride 0 and the size of the result tensor.
//// The offset in these dimensions is computed by the kernel using the index tensor's
//// values and the stride of src. The new shape is not meaningful. It's used to make
//// the shape compatible with the result tensor.
//static Tensor restride_src(const Tensor& src, int64_t dims_before, int64_t dims_indexed,
//    IntArrayRef replacement_shape) {
//  auto shape = DimVector(src.sizes());
//  auto strides = DimVector(src.strides());
//  int64_t end = dims_before + dims_indexed;
//  shape.erase(shape.begin() + dims_before, shape.begin() + end);
//  strides.erase(strides.begin() + dims_before, strides.begin() + end);
//  shape.insert(shape.begin() + dims_before, replacement_shape.begin(), replacement_shape.end());
//  strides.insert(strides.begin() + dims_before, replacement_shape.size(), 0);
//  return src.as_strided(shape, strides);
//}
//
//// Add dimensions of size 1 to an index tensor so that it can be broadcast to the result
//// shape and iterated over element-wise like the result tensor and the restrided src.
//static Tensor reshape_indexer(const Tensor& index, int64_t dims_before, int64_t dims_after) {
//  auto orig_shape = index.sizes();
//  auto shape = DimVector();
//  shape.append(dims_before, 1);
//  shape.append(orig_shape.begin(), orig_shape.end());
//  shape.append(dims_after, 1);
//  return index.reshape(shape);
//}
//

[[noreturn]]
static void invalid_mask2(const Tensor & self, int64_t idx, const Tensor & mask, int64_t maskIdx) {
  std::stringstream ss;
  ss << "The shape of the mask " << mask.sizes() << " at index " << maskIdx;
  ss << " does not match the shape of the indexed tensor " << self.sizes();
  ss << " at index " << idx;
  AT_INDEX_ERROR(ss.str());
}

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
static Tensor
flattenToFront(Tensor& self, TensorList indices) {
  std::vector<int64_t> dims;
  std::vector<Tensor> transposedIndices;
//  dims.reserve(self.dim());
  dims.reserve(indices.size());
  for (int64_t i = 0; i < indices.size()/*self.dim()*/; i++) {
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
  self = self.flatten(0L, dims.back());
  return torch::utils::flatten_dense_tensors(transposedIndices);
}

//static AdvancedIndex make_info(Tensor self, TensorList orig) {
//  checkIndexTensorTypes(orig);
//  // first expand ByteTensor (boolean masks) into 1 or more LongTensors
//  auto indices = expandByteTensors2(self, orig);
//  // next broadcast all index tensors together
//  try {
//    indices = expand_outplace(indices);
//  } catch (std::exception& e) {
//    AT_INDEX_ERROR("shape mismatch: indexing tensors could not be broadcast together"
//                   " with shapes ", shapes_as_str(indices));
//  }
//  // add missing null Tensors so that it matches self.dim()
//  while (indices.size() < (size_t)self.dim()) {
//    indices.emplace_back();
//  }
//  // if the non-null indices are not all adjacent, transpose self and indices
//  // together so that they're adjacent at the front
//  if (!hasContiguousSubspace(indices)) {
//    std::tie(self, indices) = transposeToFront(self, indices);
//  }
//  // Ensure indices are on the same device as self
//  for (size_t i = 0; i < indices.size(); i++) {
//    if (indices[i].defined() && indices[i].device() != self.device()) {
//      indices[i] = indices[i].to(self.device());
//    }
//  }
//  return AdvancedIndex(self, indices);
//}
//
//
//AdvancedIndex::AdvancedIndex(const Tensor& src, TensorList indices_list)
//{
//  int64_t element_size_bytes = src.element_size();
//  int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
//  IntArrayRef replacement_shape;
//  for (size_t dim = 0; dim < indices_list.size(); dim++) {
//    if (!indices_list[dim].defined()) {
//      if (dims_indexed == 0) {
//        dims_before++;
//      } else {
//        dims_after++;
//      }
//    } else {
//      dims_indexed++;
//      replacement_shape = indices_list[dim].sizes();
//      indexed_sizes.push_back(src.size(dim));
//      indexed_strides.push_back(src.stride(dim) * element_size_bytes);
//    }
//  }
//
//  // Check if the indexed subspace contains a dim of size 0, but the replacement
//  // shape does not. This implies that an index is out of bounds, because there
//  // is no number that's a valid index for an empty tensor. Normally, out of
//  // bounds is handled in the indexing kernel, but this case fails earlier in
//  // restride_src with an unhelpful error message.
//  if (std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) != indexed_sizes.end() &&
//      std::find(replacement_shape.begin(), replacement_shape.end(), 0) == replacement_shape.end()) {
//    AT_INDEX_ERROR("index is out of bounds for dimension with size 0");
//  }
//
//  this->dims_before = dims_before;
//  this->dims_after = dims_after;
//  this->src = restride_src(src, dims_before, dims_indexed, replacement_shape);
//
//  for (auto& index : indices_list) {
//    if (index.defined()) {
//      indices.push_back(reshape_indexer(index, dims_before, dims_after));
//    }
//  }
//
//  // For CUDA tensors, force all index tensors to have the same striding to
//  // simplify the CUDA kernel.
//  if (indices.size() >= 2 && this->src.type().device_type() == kCUDA) {
//    if (!all_strides_match(indices)) {
//      for (size_t i = 0; i < indices.size(); i++) {
//        indices[i] = indices[i].contiguous();
//      }
//    }
//  }
//}
//
//
//static std::unique_ptr<TensorIterator> make_index_put_iterator(const AdvancedIndex& info, const Tensor& value) {
//  if (!is_expandable_to(value.sizes(), info.src.sizes())) {
//    AT_ERROR("shape mismatch: value tensor of shape ", value.sizes(),
//        " cannot be broadcast to indexing result of shape ", info.src.sizes());
//  }
//  auto builder = TensorIterator::Builder();
//  builder.dont_compute_common_dtype();
//  builder.dont_resize_outputs();
//  builder.add_output(info.src);
//  builder.add_input(value, &info.src.type());
//  for (auto& index : info.indices) {
//    builder.add_input(index);
//  }
//  return builder.build();
//}


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
  if (values.numel() == 0) {
    return self;
  }

// first expand ByteTensor (boolean masks) into 1 or more LongTensors
  auto indices__ = expandByteTensors2(self, indices_);

  auto grad_arg = TensorArg(values, "grad", 1);
  auto indices_arg = TensorArg(indices__[0], "indices", 1);
  checkScalarType("index_put_cuda_", indices_arg, kLong);
  checkSameGPU("index_put_cuda_", grad_arg, indices_arg);

  int64_t num_weights = self.size(0);
  int64_t padding_idx = -1L;

  auto indices = flattenToFront(self, indices__);

  std::cerr << "self" << std::endl;
  print(std::cerr, self, 120);
  std::cerr << self.sizes() << std::endl << std::endl;

  std::cerr << "indices" << std::endl;
  print(std::cerr, indices, 120);
  std::cerr << indices.sizes() << std::endl << std::endl;

  std::cerr << "values" << std::endl;
  print(std::cerr, values, 120);
  std::cerr << values.sizes() << std::endl << std::endl;

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


//namespace {
//
//using namespace vec256;
//
//struct Indexer {
//  Indexer(int64_t num_indexers, char** indexers, const int64_t* indexer_strides,
//      IntArrayRef original_sizes, IntArrayRef original_strides)
//      : num_indexers(num_indexers)
//      , indexers(indexers)
//      , indexer_strides(indexer_strides)
//      , original_strides(original_strides.data())
//      , original_sizes(original_sizes.data()) {
//    AT_ASSERT(original_strides.size() == num_indexers);
//    AT_ASSERT(original_sizes.size() == num_indexers);
//  }
//
//  int64_t num_indexers;
//  char** indexers;
//  const int64_t* indexer_strides;
//  const int64_t* original_strides;
//  const int64_t* original_sizes;
//
//  int64_t get(int64_t idx) {
//    int64_t offset = 0;
//    for (int j = 0; j < num_indexers; j++) {
//      int64_t value = *(int64_t*)&indexers[j][idx * indexer_strides[j]];
//      int64_t size = original_sizes[j];
//      if (value < -size || value >= size) {
//        AT_INDEX_ERROR("index ", value, " is out of bounds for dimension ", j, " with size ", size);
//      }
//      if (value < 0) {
//        value += size;
//      }
//      offset += value * original_strides[j];
//    }
//    return offset;
//  }
//};
//
//static bool is_constant_index(int ntensor, const int64_t* strides) {
//  AT_ASSERT(ntensor >= 3);
//  for (int arg = 2; arg < ntensor; arg++) {
//    if (strides[arg] != 0) {
//      return false;
//    }
//  }
//  return true;
//}
//
//template <typename scalar_t, typename func_t>
//void cpu_index_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride,
//    const func_t& f, bool serial_execution=false)
//{
//  auto loop = [&](int ntensor, char** data, const int64_t* strides, int64_t n) {
//    auto indexer = Indexer(ntensor - 2, &data[2], &strides[2], index_size, index_stride);
//    char* dst = data[0];
//    char* src = data[1];
//    if (is_constant_index(ntensor, strides)) {
//      // specialization for when every element uses the same index
//      int64_t offset = indexer.get(0);
//      if (strides[0] == sizeof(scalar_t) && strides[1] == sizeof(scalar_t)) {
//        for (int64_t i = 0; i < n; i++) {
//          f(dst + strides[0] * i, src + strides[1] * i, offset);
//        }
//      } else {
//        for (int64_t i = 0; i < n; i++) {
//          f(dst + strides[0] * i, src + strides[1] * i, offset);
//        }
//      }
//    } else {
//      for (int64_t i = 0; i < n; i++) {
//        int64_t offset = indexer.get(i);
//        f(dst + strides[0] * i, src + strides[1] * i, offset);
//      }
//    }
//  };
//  if (serial_execution) {
//    iter.serial_for_each(loop, {0, iter.numel()});
//  } else {
//    iter.for_each(loop);
//  }
//}
//
//void index_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride) {
//  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, iter.dtype(), "index_cpu", [&] {
//    cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
//      *(scalar_t*)dst = *(scalar_t*)(src + offset);
//    });
//  });
//}
//
//void index_put_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate) {
//  // NOTE: duplicate indices are only supported if accumulate is true.
//  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, iter.dtype(), "index_put", [&] {
//    if (accumulate) {
//      // TODO: investigate parallelization of the accumulate kernel. Unlike the non-accumulate case,
//      // this needs to be thread-safe.
//      cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
//        *(scalar_t*)(dst + offset) += *(scalar_t*)src;
//      }, /*serial_execution=*/true);
//    } else {
//      cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
//        *(scalar_t*)(dst + offset) = *(scalar_t*)src;
//      });
//    }
//  });
//}
//
//
////void index_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride) {
////  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, iter.dtype(), "index_cpu", [&] {
////    cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
////      *(scalar_t*)dst = *(scalar_t*)(src + offset);
////    });
////  });
////}
////
////void index_put_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate) {
////  // NOTE: duplicate indices are only supported if accumulate is true.
////  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, iter.dtype(), "index_put", [&] {
////    if (accumulate) {
////      // TODO: investigate parallelization of the accumulate kernel. Unlike the non-accumulate case,
////      // this needs to be thread-safe.
////      cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
////        *(scalar_t*)(dst + offset) += *(scalar_t*)src;
////      }, /*serial_execution=*/true);
////    } else {
////      cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
////        *(scalar_t*)(dst + offset) = *(scalar_t*)src;
////      });
////    }
////  });
////}
//
//
//} // anonymous namespace

//REGISTER_DISPATCH(index_stub, &index_kernel);
//REGISTER_DISPATCH(index_put_stub, &index_put_kernel);

}}  // namespace at::native
