#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
SyncedMemory::SyncedMemory()
  : cpu_ptr_(NULL), gpu_ptr_(NULL), head_(UNINITIALIZED), size_(0),
    own_cpu_data_(FALSE), own_gpu_data_(FALSE), cpu_malloc_use_cuda_(FALSE) {
#ifndef CPU_ONLY
#define DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
}

SyncedMemory::SyncedMemory(size_t size)
  : cpu_ptr_(NULL), gpu_ptr_(NULL), head_(UNINITIALIZED), size_(size),
    own_cpu_data_(FALSE), own_gpu_data_(FALSE), cpu_malloc_use_cuda_(FALSE) {
#ifndef CPU_ONLY
#define DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
}

SyncedMemory::~SyncedMemory() {
  check_device(); // check if device_ is the device in use
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }

#ifndef CPU_ONLY
  if (gpu_ptr_ && own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
#endif 
}

void SyncedMemory::check_device() {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  cudaGetDevice(&device);
  CHECK(device == device_);
  if (gpu_ptr_ && own_gpu_data_) {
    // enum cudaMemoryType { cudaMemoryTypeHost = 1, cudaMemoryTypeDevice = 2}
    // struct cudaPointerAttributes {
    //          enum cudaMemoryType memoryType;
    //          int device; 
    //          void *devicePointer;
    //          void *hostPointer;
    //          int isManaged; // indicates if the pointer ptr points to managed
    //                         // memory or not
    //      }
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, gpu_ptr_));
    CHECK(attributes.device == device_)
  }
#endif
#endif
}


inline void SyncedMemory::to_cpu() {
  check_device();
  switch (head_) {
    case UNINITIALIZED:
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      caffe_memset(size_, 0, cpu_ptr_); // void* memset(cpu_ptr_, 0, size_) -> void* memset(void *ptr, int value, size_t num)
      head_ = HEAD_AT_CPU;
      own_cpu_data_ = TRUE;
      break;
    case HEAD_AT_CPU:
      break;
    case HEAD_AT_GPU:
#ifndef CPU_ONLY
      if (cpu_ptr_ == NULL) {
        CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
        own_cpu_data_ = true;
      }
      caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_); // cudaMemcpy(cpu_ptr_, gpu_ptr_, size_, cudaMemcpyDefault) 
                                                   // defined in math_functions.cu
      head_ = SYNCED;
#else 
      NO_GPU; // #define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode."
#endif
      break;
    case SYNCED:
      break;
  }

}

inline void SyncedMemory::to_gpu() {
  check_device();
#ifndef CPU_ONLY
  switch (head_) {
    case UNINITIALIZED:
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
      caffe_gpu_memset(size_, 0, gpu_ptr_); // cudaError_t cudaMemset(gpu_ptr_, 0, size_)
      head_ = HEAD_AT_GPU;
      own_gpu_data = TRUE;
      break;
    case HEAD_AT_CPU:
      if (gpu_ptr_ == NULL) {
        CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
        own_gpu_data = TRUE;
      }
      caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_); // params: size_t N, void *src, void *dst
                                                   // cudaMemcpy(gpu_ptr_, cpu_ptr_, size_, cudaMemcpyDefault)
      head_ = SYNCED;
      break;
    case HEAD_AT_GPU:
      break;
    case SYNCED:
      break;
#else
  NO_GPU;
#endif
  }
}

const void* SyncedMemory::cpu_data() {
  check_device();
  to_cpu();
  return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void *data) {
  check_device();
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU; 
  own_cpu_data = FALSE; // 此处置为false原因:check(data)只能确定指针(地址)是合法有效的，并不能保证data指针指向的内存是有效的
}

const void* SyncedMemory::gpu_data() {
  check_device();
#ifndef CPU_ONLY
  to_gpu();
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

void SyncedMemory::set_gpu_data(void *data) {
  check_device();
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data = FALSE;
#else
  NO_GPU;
#endif

}

void* SyncedMemory::mutable_cpu_data() {
  check_device();
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
  check_device();
#ifndef CPU_ONLY
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

#ifndef CPU_ONLY
void SyncedMemory::async_gpu_push(const cudaStream_t &stream) {
  check_device();
  CHECK(head_ == HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data = TRUE;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
#endif

} // namespace caffe
