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

#    for i in range(5):
    idx = 1312 #800 + np.random.randint(500)

    #image_path = image_dir_path + str(idx) + '.jpg'
    image_path = '/home/daniel/Desktop/Aspect-Raw1312-crop.jpg'

    print(image_path)
    in_image = Image.open(image_path, 'r')
    plt.figure(0)
    plt.imshow(np.asarray(in_image))
    plt.figure(1)
    belief = get_belief_given_observation(image_path, encoder, aspect_nodes_path)
    plt.bar(np.arange(belief.shape[0]), belief)
    plt.figure(2)
    imshow(torchvision.utils.make_grid(aspect_node_images.data), True)
