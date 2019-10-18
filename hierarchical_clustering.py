from sklearn.cluster import DBSCAN
import numpy as np

if __name__ =='__main__':
    atg_dataset_vgg_feat = np.load("atg_dataset_vgg_feat.npz")
    data = atg_dataset_vgg_feat['deep_feat']
    clustering = DBSCAN(eps=3, min_samples=12).fit(data)
    print(clustering.labels_)
