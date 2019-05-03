#pragma once
#include <ATen/ATen.h>

#ifdef _WIN32
#if !defined(AT_CORE_STATIC_WINDOWS)
#if defined(ATen_cpu_EXPORTS)   || defined(ATen_cuda_EXPORTS)  || \
    defined(caffe2_EXPORTS)     || defined(caffe2_gpu_EXPORTS) || \
    defined(caffe2_hip_EXPORTS) || defined(CAFFE2_BUILD_MAIN_LIBS)
#  define AT_NATIVE_API __declspec(dllexport)
# else
#  define AT_NATIVE_API __declspec(dllimport)
# endif
#else
# define AT_NATIVE_API
#endif
#elif defined(__GNUC__)
#if defined(ATen_cpu_EXPORTS)   || defined(ATen_cuda_EXPORTS)  || \
    defined(caffe2_EXPORTS)     || defined(caffe2_gpu_EXPORTS) || \
    defined(caffe2_hip_EXPORTS)
#define AT_NATIVE_API __attribute__((__visibility__("default")))
#else
#define AT_NATIVE_API
#endif
#else
# define AT_NATIVE_API
#endif
