#pragma once

// Indexing tensors by by tensors

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {
  struct TensorIterator;
}

namespace at { namespace native {

static void checkIndexTensorTypes(TensorList indices);
static bool hasContiguousSubspace(TensorList tl);
static std::vector<int64_t> computeLinearStride(const Tensor & tensor);
static Tensor unsqueezeN(const Tensor & src, int64_t before, int64_t after);
static std::vector<Tensor> expandTensors(const Tensor & self, TensorList indices);
static std::tuple<Tensor, std::vector<Tensor>> transposeToFront(Tensor self, TensorList indices);

using index_fn = void(*)(TensorIterator &, IntArrayRef indexed_sizes, IntArrayRef indexed_strides);
using index_put_fn = void(*)(TensorIterator &, IntArrayRef indexed_sizes, IntArrayRef indexed_strides, bool accumulate);

DECLARE_DISPATCH(index_fn, index_stub);
DECLARE_DISPATCH(index_put_fn, index_put_stub);

}} // namespace at::native
