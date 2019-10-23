import numpy as np
import time
from utils import *
from PIL import Image

if __name__ =='__main__':
    encoder_feats_path = 'data/real_dataset_encoder_feat.npz'
    clustering_result_path = 'data/real_encoder_clustering_result.npz'
    aspect_nodes_path = 'data/aspect_nodes.npz'
    dataset_path = 'data/real_aspects/'

    autoencoder = nn.Sequential(Encoder(), Decoder())
    autoencoder.load_state_dict(torch.load("weights/autoencoder.pkl"))
    encoder = autoencoder[0]

    cluster_observations_to_aspect_nodes(encoder_feats_path,
                                         clustering_algorithm = 'DBSCAN',
                                         clustering_param = {'eps': 3e1, 'min_samples': 1},
                                         output_path = 'data/real_encoder_clustering_result.npz')
    '''
    cluster_observations_to_aspect_nodes(encoder_feats_path,
                                         clustering_algorithm = 'OPTICS',
                                         clustering_param = {'max_eps':2e8, 'xi': 0.05, 'min_samples': 1, 'min_cluster_size':None},
                                         output_path = 'data/real_encoder_clustering_result.npz')
    '''
    get_aspect_nodes(clustering_result_path, dataset_path, aspect_nodes_path)

    clustering_result = np.load(aspect_nodes_path)
    aspect_node_images = torch.from_numpy(clustering_result['aspect_node_images'].transpose(0, 3 , 1, 2))


    image_dir_path = 'data/real_aspects/Aspect-Raw'

    for i in range(5):
        idx = 800 + np.random.randint(500)
        image_path = image_dir_path + str(idx) + '.jpg'
        print(image_path)
        in_image = Image.open(image_path, 'r')
        plt.figure(0)
        plt.imshow(np.asarray(in_image))
        plt.figure(1)
        belief = get_belief_given_observation(image_path, encoder, aspect_nodes_path)
        plt.bar(np.arange(belief.shape[0]), belief)
        plt.figure(2)
        imshow(torchvision.utils.make_grid(aspect_node_images.data), True)
