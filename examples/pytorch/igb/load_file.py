import os.path as osp
import numpy as np
num_nodes = 100000000
dir = '/media/yash/dataset/igb_large/'
path = osp.join(dir, 'full', 'processed', 'paper', 'node_label_19.npy')
node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes))
np.save("/media/yash/dataset/igb_large/full_node_labels.npy", node_labels)