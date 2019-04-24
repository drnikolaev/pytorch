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
#define GRID_SIZE 128
#define GROUP_SIZE 32

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


__global__ void arange_kernel(int64_t n, int64_t* a) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
      i += blockDim.x * gridDim.x) {
    a[i] = i;
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

  // Compute the linear indices for the parts of the tensor not being indexed
  // ...and not being sorted
  Tensor beforeIndex;
  if (emptyBefore > 0) {
    Tensor index = at::native::empty_cuda({nElemBefore},
        self.options().dtype(kLong).device(at::DeviceType::CUDA));
    int64_t *pData = index.data<int64_t>();
    void* args[] = {&nElemBefore, &pData};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    THCudaCheck(
        cudaLaunchKernel((const void*) &arange_kernel,
            WARP_SIZE, GRID_SIZE, args, 0, stream));
    THCudaCheck(cudaStreamSynchronize(stream));
    index = index * strides[emptyBefore - 1];
    index = index.view(self.sizes().slice(0, emptyBefore));
    beforeIndex = unsqueezeN(index, 0, linearIndex.dim() + emptyAfter);
  }
  Tensor afterIndex;
  if (emptyAfter > 0) {
    Tensor index = at::native::empty_cuda({nElemAfter},
        self.options().dtype(kLong).device(at::DeviceType::CUDA));
    int64_t *pData = index.data<int64_t>();
    void* args[] = {&nElemAfter, &pData};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    THCudaCheck(
        cudaLaunchKernel((const void*) &arange_kernel,
            WARP_SIZE, GRID_SIZE, args, 0, stream));
    THCudaCheck(cudaStreamSynchronize(stream));
    index = index.view(self.sizes().slice(self.dim() - emptyAfter, emptyAfter));
    afterIndex = unsqueezeN(index, linearIndex.dim() + emptyBefore, 0);
  }

  return std::make_tuple(self, linearIndex, beforeIndex, afterIndex, emptyBefore, emptyAfter,
      nElemBefore, nElemAfter);
}

template<typename T, typename IndexType>
__device__ __forceinline__ IndexType
indexToOffset(const at::cuda::detail::TensorInfo<T, IndexType>& info, IndexType linearIndex) {
  IndexType offset(0);
  for (int i = info.dims - 1; i > 0; --i) {
    offset += (linearIndex % info.sizes[i]) * info.strides[i];
    linearIndex /= info.sizes[i];
  }
  return offset + linearIndex * info.strides[0];
}

//template<typename IndexType>
//__device__ __forceinline__
//IndexType indexToOffset(IndexType dims, IndexType* sizes, IndexType* strides,
//    IndexType linearIndex) {
//  IndexType offset(0);
//  for (IndexType i = dims - 1; i > 0; --i) {
//    offset += (linearIndex % sizes[i]) * strides[i];
//    linearIndex /= sizes[i];
//  }
//  return offset + linearIndex* strides[0];
//}

template<typename index_t>
__device__ __forceinline__
index_t extended_pos(index_t nseq, index_t baStride, index_t sortedSize,
    const int64_t* origOrder) {
  return nseq / sortedSize + origOrder[nseq % sortedSize] * baStride;
}

template<typename index_t>
__device__ __forceinline__
index_t extended_pos(index_t idx, index_t blockSize, index_t currentThreadInBlock,
    index_t baStride, index_t sortedSize, const int64_t* origOrder) {
  const index_t nseq = idx * blockSize + currentThreadInBlock;
  return extended_pos(nseq, baStride, sortedSize, origOrder);
}

template<typename scalar_t>
__global__
void backward_indexing_kernel(const int64_t* extendedIdx,
    int64_t* origOrder, scalar_t* gradValues, int64_t extendedIdxSize,
    int64_t baStride, int64_t sortedSize, scalar_t* dstData) {
  using accscalar_t = acc_type<scalar_t, true>;

  int blockSize = blockDim.x * blockDim.y * blockDim.z;
  int idxMax = (extendedIdxSize + blockSize - 1) / blockSize;
  int idx = blockIdx.x;

  int blockHeadPos = extended_pos<int64_t>(idx, blockSize, 0L,
      baStride, sortedSize, origOrder);

//  printf("BBBBBBBB blockHeadPos %d \n", blockHeadPos); // TODO!!!

  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < extendedIdxSize;
       i += blockDim.x * gridDim.x) {

    if (blockHeadPos < extendedIdxSize &&
        (idx == 0 || blockHeadPos != extended_pos<int64_t>(idx - 1, blockSize, 0L,
                                                           baStride, sortedSize, origOrder))) {
      int th = i % GROUP_SIZE;
      const int no = extended_pos<int>(
          idx, blockSize, th, baStride, sortedSize, origOrder);

      //       printf("****** %d %d %d %d\n", idx, i, th, no); // TODO!!!

      __shared__ int offsetArr[GROUP_SIZE];
      __shared__ accscalar_t valArr[GROUP_SIZE];

      offsetArr[th] = extendedIdx[no];
      valArr[th] = gradValues[no];
      __syncthreads();
      if (th == 0 && idx < idxMax) {
        int currentBlockEnd = extendedIdxSize - idx * GROUP_SIZE;
        currentBlockEnd =
            currentBlockEnd < GROUP_SIZE ? currentBlockEnd : GROUP_SIZE;

        //     printf("                %d %d %d %d\n", idx, i, th, currentBlockEnd); // TODO!!!

        for (int t = 0; t < currentBlockEnd; ++t) {
          dstData[offsetArr[t]] += valArr[t];

          //        printf("%d %d %d\n", idx, t, offsetArr[t]);
        }
        __threadfence();
      }
      ++idx;
      const int blockHeadPosNext = extended_pos<int64_t>(
          idx, blockSize, 0L, baStride, sortedSize, origOrder);
      if (blockHeadPosNext >= extendedIdxSize ||
          blockHeadPosNext != blockHeadPos) {
        // next block is processing other images' indexes, exit.
        break;
      }
      blockHeadPos = blockHeadPosNext; // keep going in the same warp
    }
  }
//  __threadfence_block();
}


/*

  //int idx = blockIdx.x * GROUP_SIZE + threadIdx.y;
  int blockSize = blockDim.x * blockDim.y * blockDim.z;

//  if (idx % GROUP_SIZE > 0) return;

  int blockHeadPos = (int) extended_pos<int64_t>(idx, baStride, sortedSize, origOrder);


  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < extendedIdxSize; i += blockDim.x * gridDim.x) {
    int th = i % WARP_SIZE;



  if (i < extendedIdxSize && (i == 0 || blockHeadPos !=
      extended_pos<int64_t>(i - 1, baStride, sortedSize, origOrder))) {


    do {
//      const int ft = threadIdx.x + blockIdx.y * blockDim.x * GROUP_SIZE;


//      int offsetArr[GROUP_SIZE];
//      accscalar_t valArr[GROUP_SIZE];

      #pragma unroll
      for (int g = 0; g < GROUP_SIZE; g++) {

//        int feature_dim = ft + i * WARP_SIZE;
//        const int th = i + i;

        printf("****** %d %d %d   %d %d \n", idx, i, th,
               ft, feature_dim); // TODO!!!

        const int no = extended_pos<int>(i + g, baStride, sortedSize, origOrder);
//        offsetArr[i] = extendedIdx[no];
//        valArr[i] = gradValues[no];

        dstData[extendedIdx[no]] += gradValues[no];

      }
      idx++;
      const int blockHeadPosNext =
          extended_pos<int64_t>(idx, baStride, sortedSize, origOrder);
      if (blockHeadPosNext >= extendedIdxSize || blockHeadPosNext != blockHeadPos) {
        // next block is processing other images' indexes, exit.
        break;
      }
      blockHeadPos = blockHeadPosNext;  // keep going in the same warp
    } while (true);
  }
*/

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

Tensor& index_put_cuda_(Tensor& self_, TensorList indices, const Tensor& value, bool accumulate) {
  if (indices.size() > (size_t) self_.dim()) {
    AT_INDEX_ERROR("too many indices for tensor of dimension ", self_.dim(), " (got ",
        indices.size(), ")");
  }
  Tensor self, linearIndex;
  Tensor beforeIndex, afterIndex;
  int64_t emptyBefore = 0L, emptyAfter = 0L;
  int64_t nElemBefore = 1L, nElemAfter = 1L;

  std::tie(self, linearIndex, beforeIndex, afterIndex, emptyBefore, emptyAfter, nElemBefore,
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
    auto sortedLinearIndex_iter = thrust::device_ptr<int64_t>(sortedLinearIndex_beg);
    auto origCounters_iter = thrust::device_ptr<int64_t>(origCounters_beg);
    auto self_info = cuda::detail::getTensorInfo<scalar_t, int64_t>(self);
    TensorAccumFullyIndexedPutOp<scalar_t> aiPutOp(self_info, value.data<scalar_t>(),
        sortedLinearIndex_beg, sortedLinearIndex_end, origCounters_beg);

    thrust::sequence(policy, origCounters_iter, origCounters_iter + idxSize);

    thrust::sort_by_key(policy, sortedLinearIndex_iter, sortedLinearIndex_iter + idxSize,
        origCounters_iter, ThrustLTOp<int64_t>());

    if (!beforeIndex.defined() && !afterIndex.defined()) {
      // Full size index, done:
      thrust::counting_iterator<int64_t> first(0);
      thrust::counting_iterator<int64_t> last(idxSize);
      thrust::for_each(policy, first, last, aiPutOp);
    }
  });

  if (beforeIndex.defined() || afterIndex.defined()) {
    // Sum with broadcasting to compute the full index
    // using unsorted original
    linearIndex = unsqueezeN(linearIndex, emptyBefore, emptyAfter);
    if (emptyBefore > 0) {
      linearIndex = linearIndex + beforeIndex;
    }
    if (emptyAfter > 0) {
      linearIndex = linearIndex + afterIndex;
    }

//    linearIndex = linearIndex.flatten();
//
//
//    std::cout << "linearIndex += before" << std::endl;
//print(linearIndex, 120);
//std::cout << linearIndex.sizes() << std::endl
//<< "strides: " << computeLinearStride(linearIndex)
//<< std::endl
//<< std::endl;



    AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "index_put_cuda_kernel_", [&] {
      cuda::detail::TensorInfo <scalar_t, int64_t> self_info =
          cuda::detail::getTensorInfo<scalar_t, int64_t>(self);
      scalar_t* valuePtr = value.data<scalar_t>();
      int64_t extendedIdxSize = linearIndex.numel();
      int64_t* origCountersPtr = origCounters.data<int64_t>();
      int64_t* extendedLinearIndexPtr = linearIndex.data<int64_t>();
      int64_t baStride = nElemAfter * nElemBefore;

         //    printf("############# %lld \n", baStride); // TODO!!!

//      int blockSize = WARP_SIZE;
      dim3 blockSize(GROUP_SIZE); //, GROUP_SIZE);WARP_SIZE,
      dim3 gridSize(GRID_SIZE); //(extendedIdxSize + GROUP_SIZE - 1) / GROUP_SIZE);
//      dim3 gridSize((extendedIdxSize + GROUP_SIZE - 1) / GROUP_SIZE);
  //        THCCeilDiv(num_indices, (int64_t) 4), THCCeilDiv(stride, (int64_t) 128));

      //int gridSize = GRID_SIZE;// (extendedIdxSize + blockSize - 1) / blockSize;
      void* args[] = {&extendedLinearIndexPtr, &origCountersPtr, &valuePtr, &extendedIdxSize,
                      &baStride, &idxSize, &self_info.data};
      THCudaCheck(
          cudaLaunchKernel((const void*) &backward_indexing_kernel<scalar_t>, gridSize, blockSize,
              args, 0, stream));
      THCudaCheck(cudaStreamSynchronize(stream));
      THCudaCheck(cudaGetLastError());
    });
  }
  return self_;
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
//
//std::cout << "linearIndex += before" << std::endl;
//print(linearIndex, 120);
//std::cout << linearIndex.sizes() << std::endl
//<< "strides: " << computeLinearStride(linearIndex)
//<< std::endl
//<< std::endl;

//std::cout << "self" << std::endl;
//print(self, 120);
//std::cout << self.sizes() << std::endl
//<< "strides: " << computeLinearStride(self)
//<< std::endl
//<< std::endl;
//
//std::cout << "origCounters" << std::endl;
//print(origCounters, 120);
//std::cout << origCounters.sizes() << std::endl
//<< "strides: " << computeLinearStride(origCounters)
//<< std::endl
//<< std::endl;

//      std::cout << "*********** afterIndex" << std::endl;
//      print(afterIndex, 120);
//      std::cout << afterIndex.sizes() << std::endl
//                << "strides: " << computeLinearStride(afterIndex)
//                << std::endl
//                << std::endl;
//
//      std::cout << "linearIndex += after" << std::endl;
//      print(linearIndex, 120);
//      std::cout << linearIndex.sizes() << std::endl
//                << "strides: " << computeLinearStride(linearIndex)
//                << std::endl
//                << std::endl;
//    std::cout << "sortedLinearIndex" << std::endl;
//                        print(sortedLinearIndex, 120);
//    std::cout << sortedLinearIndex.sizes() << std::endl
//                                  << "strides: " << computeLinearStride(sortedLinearIndex)
//                                  << std::endl
//                                  << std::endl;
