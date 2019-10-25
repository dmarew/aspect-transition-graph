import numpy as np
import time
from utils import *
from PIL import Image

def cluster(encoder_feats_path,
            expected_number_of_nodes = 25,
            range_tolerance = 5,
            eps_range = [1e1, 2e1],
            eps_resolution = 0.5,
            clustering_algorithm = 'DBSCAN',
            min_samples = 1,
            output_path = 'data/real_encoder_clustering_result.npz'):

    cluster_time = time.time()

    eps_range_min, eps_range_max = eps_range[0], eps_range[1]
    success = False
    for eps in np.arange(eps_range_min, eps_range_max, eps_resolution):
        print('using eps %.2f'%(eps))
        clustering_param = {'eps': eps, 'min_samples': min_samples}
        n_apect_nodes  = cluster_observations_to_aspect_nodes(encoder_feats_path,
                                                              clustering_algorithm = clustering_algorithm,
                                                              clustering_param = clustering_param,
                                                              output_path = output_path)
        if abs(n_apect_nodes - expected_number_of_nodes) <= range_tolerance:
            success = True
            print('Found Ideal Clustering with eps %.2f it found %d unique aspect nodes'%(eps, n_apect_nodes))
            break
    if not success:
        print('Ideal cluster not found use larger eps range')
    print('Ideal Cluster search took %.2f secs '%(time.time()-cluster_time))
    return success

if __name__ =='__main__':
    encoder_feats_path = 'data/real_dataset_encoder_feat.npz'
    clustering_result_path = 'data/real_encoder_clustering_result.npz'
    aspect_nodes_path = 'data/aspect_nodes.npz'
    dataset_path = 'data/real_aspects/'

    autoencoder = nn.Sequential(Encoder(), Decoder())
    autoencoder.load_state_dict(torch.load("weights/autoencoder.pkl"))
    encoder = autoencoder[0]

    '''
    cluster_observations_to_aspect_nodes(encoder_feats_path,
                                         clustering_algorithm = 'DBSCAN',
                                         clustering_param = {'eps': 1.2e1, 'min_samples': 1},
                                         output_path = 'data/real_encoder_clustering_result.npz')
    clustering_result = np.load(clustering_result_path)
    clustering_features = clustering_result['clustering_features']
    distance_between_feats = distance.cdist(clustering_features , clustering_features, metric='euclidean')
    print(distance_between_feats.min(), distance_between_feats.max(), distance_between_feats.mean(), distance_between_feats.var())


    cluster_observations_to_aspect_nodes(encoder_feats_path,
                                         clustering_algorithm = 'AffinityPropagation',
                                         clustering_param = {'damping':0.9,
                                                             'max_iter':200,
                                                             'convergence_iter':15},
                                         output_path = 'data/real_encoder_clustering_result.npz')


    cluster_observations_to_aspect_nodes(encoder_feats_path,
                                         clustering_algorithm = 'OPTICS',
                                         clustering_param = {'max_eps':2e8, 'xi': 0.05, 'min_samples': 1, 'min_cluster_size':None},
                                         output_path = 'data/real_encoder_clustering_result.npz')


    cluster(encoder_feats_path,
                expected_number_of_nodes = 7,
                range_tolerance = 0,
                eps_range = [1e1, 2e1],
                eps_resolution = 0.25,
                clustering_algorithm = 'DBSCAN',
                min_samples = 1,
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
        [belief_inv, belief_cosine, belief_neg] = get_belief_given_observation(image_path, encoder, aspect_nodes_path)

        plt.subplot(2, 3, 1)
        plt.imshow(np.asarray(in_image))
        plt.title('Input image')
        plt.subplot(2, 3, 2)
        plt.bar(np.arange(belief_inv.shape[0]), belief_inv)
        plt.title('Inverse standard euclidean distance belief')
        plt.xlabel('aspect')
        plt.ylabel('belief')
        plt.subplot(2, 3, 3)
        plt.bar(np.arange(belief_cosine.shape[0]), belief_cosine)
        plt.title('Inverse squared euclidean distance belief')
        plt.xlabel('aspect')
        plt.ylabel('belief')
        plt.subplot(2, 3, 4)
        plt.bar(np.arange(belief_neg.shape[0]), belief_neg)
        plt.title('Negative distance belief')
        plt.xlabel('aspect')
        plt.ylabel('belief')
        plt.subplot(2, 3, 5)
        imshow(torchvision.utils.make_grid(aspect_node_images.data), False)
        plt.title('Aspect nodes')
        plt.show()
