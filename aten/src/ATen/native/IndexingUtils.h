#include <ATen/ExpandUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/ATenNativeGeneral.h>

namespace at { namespace native {

[[noreturn]]
AT_NATIVE_API
void invalid_mask(const Tensor & self, int64_t idx, const Tensor & mask,
    int64_t maskIdx);
AT_NATIVE_API
std::string shapes_as_str(TensorList tensors);
AT_NATIVE_API
std::vector<Tensor> expandTensors(const Tensor & self,
    TensorList indices);
AT_NATIVE_API
void checkIndexTensorTypes(TensorList indices);
AT_NATIVE_API
bool hasContiguousSubspace(TensorList tl);
AT_NATIVE_API
std::tuple<Tensor, std::vector<Tensor>> transposeToFront(Tensor self,
    TensorList indices);
AT_NATIVE_API
std::tuple<Tensor, std::vector<Tensor>, std::vector<int64_t>>
transposeToFrontAndInvPerm(Tensor self, TensorList indices);
AT_NATIVE_API
std::vector<int64_t> computeLinearStride(const Tensor & tensor);
AT_NATIVE_API
Tensor unsqueezeN(const Tensor & src, int64_t before, int64_t after);

#ifdef _WIN32
AT_NATIVE_API
#endif
struct AdvancedIndex {
  AdvancedIndex(const Tensor& src, TensorList indices);

  Tensor src;
  std::vector<Tensor> indices;
  DimVector indexed_sizes;
  DimVector indexed_strides;
  int64_t dims_before;
  int64_t dims_after;
};

}}
