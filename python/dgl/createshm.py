from ._ffi.function import _init_api

def create_shmarray(size, mad, name, pin):
  return _CAPI_DGLCreateShmArray(size, mad, name, pin)

def create_shmoffset(size, name):
  return _CAPI_DGLCreateShmOffset(size, name)

def get_shm_ptr(name, size, offset_or_block):
  return _CAPI_DGLGetShmPtr(name, size, offset_or_block)

def reset_shm(offset):
  return _CAPI_DGLResetShm(offset)

def print_offset(offset):
  return _CAPI_DGLPrintOffset(offset)

def read_offset(name):
  return _CAPI_DGLReadOffset(name)

_init_api("dgl.createshm", __name__)