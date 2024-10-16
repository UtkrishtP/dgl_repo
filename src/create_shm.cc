#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>
#include <dgl/packed_func_ext.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
using namespace dgl::runtime;

DGL_REGISTER_GLOBAL("createshm._CAPI_DGLCreateShmArray")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    size_t size = args[0];
    int mad = args[1];
    std::string name = args[2];
    void *ptr;
    int flag = O_RDWR | O_CREAT;
    auto fd_ = shm_open(name.c_str(), flag, S_IRUSR | S_IWUSR);
    auto ret = ftruncate(fd_, size);
    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED , fd_, 0);
    if(mad) madvise(ptr, size, MADV_HUGEPAGE);
    memset(ptr, 0, size);
    *rv = ptr;
  });

DGL_REGISTER_GLOBAL("createshm._CAPI_DGLCreateShmOffset")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    size_t size = args[0];
    std::string name = args[1];
    void* ptr_;
    int flag = O_RDWR | O_CREAT;
    auto fd_ = shm_open(name.c_str(), flag, S_IRUSR | S_IWUSR);
    auto ret = ftruncate(fd_, sizeof(size_t) * 2);
    ptr_ = mmap(NULL, sizeof(size_t) * 2, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    *(size_t*)ptr_ = size;
    *((size_t*)ptr_ + 1) = 0;
    *rv = ptr_;
  });

DGL_REGISTER_GLOBAL("createshm._CAPI_DGLGetShmPtr")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string name = args[0];
    size_t size = args[1];
    int offset_or_blocks = args[2];
    void *ptr_;
    if (offset_or_blocks == 0) {
      int flag = O_RDWR;
      auto fd_ = shm_open(name.c_str(), flag, S_IRUSR | S_IWUSR);
      auto ret = ftruncate(fd_, size);
      ptr_ = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    } else {
      int flag = O_RDWR;
      auto fd_ = shm_open(name.c_str(), flag, S_IRUSR | S_IWUSR);
      auto ret = ftruncate(fd_, sizeof(size_t) * 2);
      ptr_ = mmap(NULL, sizeof(size_t) * 2, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
      *((size_t*)ptr_ + 1) = offset_or_blocks;
    }
    *rv = ptr_;
  });

DGL_REGISTER_GLOBAL("createshm._CAPI_DGLResetShm")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    void* ptr_ = args[0];
    *((size_t*)ptr_ + 1) = 0;
    *rv = ptr_;
  });

DGL_REGISTER_GLOBAL("createshm._CAPI_DGLPrintOffset")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    void* ptr_ = args[0];
    std::cout << "offset: " << *((size_t*)ptr_ + 1) << std::endl;
    *rv = ptr_;
  });
