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
    void *ptr;
    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if(mad) madvise(ptr, size, MADV_HUGEPAGE);
    memset(ptr, 0, size);
    *rv = ptr;
  });

DGL_REGISTER_GLOBAL("createshm._CAPI_DGLCreateShmOffset")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    size_t size = args[0];
    void* ptr_;
    ptr_ = mmap(NULL, sizeof(size_t) * 2, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    *(size_t*)ptr_ = size;
    *((size_t*)ptr_ + 1) = 0;
    *rv = ptr_;
  });

DGL_REGISTER_GLOBAL("createshm._CAPI_DGLCreateShmOffset_")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    size_t size = args[0];
    void* ptr__;
    ptr__ = mmap(NULL, sizeof(size_t) * 2, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    *(size_t*)ptr__ = size;
    *((size_t*)ptr__ + 1) = 0;
    *rv = ptr__;
  });

DGL_REGISTER_GLOBAL("createshm._CAPI_DGLResetShm")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    void* ptr_ = args[0];
    *((size_t*)ptr_ + 1) = 0;
    *rv = ptr_;
  });
