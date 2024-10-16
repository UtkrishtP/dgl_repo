from ._ffi.function import _init_api

def create_shmarray(size, mad, name):
  return _CAPI_DGLCreateShmArray(size, mad, name)

def create_shmoffset(size, name):
  return _CAPI_DGLCreateShmOffset(size, name)

def get_shm_ptr(name, size, offset_or_block):
  return _CAPI_DGLGetShmPtr(name, size, offset_or_block)

def reset_shm(offset):
  return _CAPI_DGLResetShm(offset)

def print_offset(offset):
  return _CAPI_DGLPrintOffset(offset)

_init_api("dgl.createshm", __name__)