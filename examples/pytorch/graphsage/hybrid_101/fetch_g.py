from dgl.convert import hetero_from_shared_memory

def fetch_all():
    # global shapes, dtypes
    g = hetero_from_shared_memory("graph_formats")
    return g