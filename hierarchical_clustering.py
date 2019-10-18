from sklearn.cluster import DBSCAN
import numpy as np
import time

if __name__ =='__main__':
    atg_dataset_vgg_feat = np.load("data/atg_dataset_encoder_feat.npz")
    data = atg_dataset_vgg_feat['deep_feat']
    clustering_time = time.time()
    clustering = DBSCAN(eps=3, min_samples=12).fit(data)
    print(clustering.labels_)
    print('clustering took ', time.time()-clustering_time, ' secs')
    print('Found ', np.max(clustering.labels_), ' unique aspect nodes')
    np.savez('data/encoder_clustering_labels.npz', clustering_labels = clustering.labels_)
    print('Done dumping labels')
