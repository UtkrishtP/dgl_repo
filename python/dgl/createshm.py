from ._ffi.function import _init_api

def create_shmarray(size, mad):
  # MyAdd has been registered via `_ini_api` call below
  return _CAPI_DGLCreateShmArray(size, mad)

def create_shmoffset(size):
  return _CAPI_DGLCreateShmOffset(size)

def create_shmoffset_(size):
  return _CAPI_DGLCreateShmOffset_(size)

def reset_shm(offset):
  return _CAPI_DGLResetShm(offset)

_init_api("dgl.createshm", __name__)