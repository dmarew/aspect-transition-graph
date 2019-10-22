import numpy as np
import time
from utils import *

if __name__ =='__main__':
    encoder_feats_path = 'data/atg_dataset_encoder_feat.npz'
    clustering_result_path = 'data/encoder_clustering_result.npz'
    cluster_observations_to_aspect_nodes(encoder_feats_path,
                                         clustering_algorithm = 'DBSCAN',
                                         output_path = clustering_result_path)
