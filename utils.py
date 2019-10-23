#numpy
import numpy as np
from scipy.interpolate import interp1d
from PIL import Image
import imageio
import skimage.transform
import matplotlib.pyplot as plt
#torch
import torch
import torchvision.transforms
from torch.autograd import Variable
#system
import os
import time
import multiprocessing
import threading
import queue
import glob

from sklearn.cluster import *
from scipy.spatial import distance
from models import *


def get_belief_given_observation(image_path, encoder, aspect_nodes_path):
    belief_time = time.time()

    aspect_nodes = np.load(aspect_nodes_path)['aspect_nodes']
    num_aspect_nodes = len(aspect_nodes)

    image = Image.open(image_path)
    image = image_to_tensor(image)

    image_tensor = to_var(image)
    encoder_feat = encoder(image_tensor).view(-1).data.numpy()

    belief = np.zeros(num_aspect_nodes)

    for i, aspect_node in enumerate(aspect_nodes):
        belief[i] = 1./(1e-8 + distance.euclidean(encoder_feat, aspect_node))

    total_belief = belief.sum()
    belief /= total_belief
    #belief -= belief.min()
    #belief  = 100*belief

    #belief = torch.nn.functional.softmax(torch.from_numpy(belief), dim=0).numpy()
    #print(belief, belief.max(), belief.sum())
    print('getting belief took %.2f secs'%(time.time()-belief_time))
    return belief

def get_aspect_nodes(clustering_result_path, dataset_path, aspect_nodes_path):

    atg_time = time.time()
    dataset = np.array(glob.glob(dataset_path + '*'))

    clustering_result = np.load(clustering_result_path)

    clustering_labels = clustering_result['clustering_labels']
    clustering_features = clustering_result['clustering_features']

    num_aspect_nodes   = np.max(clustering_labels) + 1
    aspect_nodes = []
    aspect_node_images = []
    for i in range(num_aspect_nodes):

        aspects = clustering_features[clustering_labels==i]
        images_of_aspects  = dataset[clustering_labels==i]

        mean_aspect = aspects.mean(axis=0, keepdims=True)
        distance_to_mean = distance.cdist(aspects ,
                                          mean_aspect,
                                          metric='euclidean').squeeze(1)
        token_a_n_idx = np.where(distance_to_mean==distance_to_mean.min())[0][0]
        token_aspect_node = aspects[token_a_n_idx]
        token_aspect_node_image_path = images_of_aspects[token_a_n_idx]

        aspect_nodes.append(token_aspect_node)
        image = imageio.imread(token_aspect_node_image_path)
        aspect_node_images.append(image)


    aspect_nodes = np.array(aspect_nodes)
    aspect_node_images = np.stack(aspect_node_images)

    np.savez(aspect_nodes_path, aspect_nodes=aspect_nodes, aspect_node_images=aspect_node_images)
    print('extracting aspect_nodes took %.2f secs '%(time.time()-atg_time))

def build_aspect_transition_graph(clustering_result_path):

    atg_time = time.time()

    clustering_result = np.load(clustering_result_path)

    aspect_seq = clustering_result['clustering_labels']
    action_seq = clustering_result['action_seq']

    max_action_space  = np.max(action_seq) + 1
    max_aspect_node   = np.max(aspect_seq) + 1
    num_samples = aspect_seq.shape[0]
    atg = np.zeros((max_aspect_node, max_action_space, max_aspect_node))

    for t in range(1, num_samples):#ignore first sample
        atg[aspect_seq[t-1], action_seq[t], aspect_seq[t]] += 1
    atg = torch.nn.functional.softmax(torch.from_numpy(atg), dim = 2).numpy()
    print('build aspect transition graph took ', time.time() - atg_time, ' secs')
    return atg

def cluster_observations_to_aspect_nodes(encoder_feats_path,
                                         clustering_algorithm = 'DBSCAN',
                                         clustering_param = {'eps': 5e1, 'min_samples': 1},
                                         output_path = 'data/real_encoder_clustering_result.npz'):

    atg_dataset_feat = np.load(encoder_feats_path)
    data = atg_dataset_feat['encoder_feat']
    #print(data[0], data[0].min(), data[0].max())
    clustering_time = time.time()
    num_unique_apect_nodes = 0
    print('Performing %s clustering'%(clustering_algorithm))

    if clustering_algorithm == 'DBSCAN':
        clustering = DBSCAN(eps=clustering_param['eps'], min_samples=clustering_param['min_samples']).fit(data)
        #print(clustering.labels_)
        #print('clustering took ', time.time()-clustering_time, ' secs')
        #print('Found ', np.max(clustering.labels_) + 1, ' unique aspect nodes')
        num_unique_apect_nodes = np.max(clustering.labels_) + 1
        np.savez(output_path, clustering_labels = clustering.labels_, clustering_features=data)


    elif clustering_algorithm == 'OPTICS':
        clustering = OPTICS(min_samples=clustering_param['min_samples'], max_eps= clustering_param['max_eps'], xi=clustering_param['xi']).fit(data)
        #print(clustering.labels_)
        #print('clustering took ', time.time()-clustering_time, ' secs')
        #print('Found ', np.max(clustering.labels_) + 1, ' unique aspect nodes')
        np.savez(output_path, clustering_labels = clustering.labels_, clustering_features=data)
        num_unique_apect_nodes = np.max(clustering.labels_) + 1
    elif clustering_algorithm == 'AffinityPropagation':

        clustering = AffinityPropagation(damping=clustering_param['damping'],
                                         max_iter=clustering_param['max_iter'],
                                         convergence_iter=clustering_param['convergence_iter'],verbose=True).fit(data)
        #print(clustering.labels_)
        print('clustering took ', time.time()-clustering_time, ' secs')
        print('Found ', np.max(clustering.labels_) + 1, ' unique aspect nodes')
        np.savez(output_path, clustering_labels = clustering.labels_, clustering_features=data)
        num_unique_apect_nodes = np.max(clustering.labels_) + 1
    else:
        raise NotImplementedError(clustering_algorithm + " has not been implemented yet!!")
    return num_unique_apect_nodes
def get_encoder_feature_for_dataset(encoder,
                                    num_workers=2,
                                    dataset_path = 'data/real_aspects/',
                                    output_path='data/atg_dataset_encoder_feat.npz'):
    '''
    Creates a trained recognition system by generating training features from all training images.
    [input]clustering_labels_path
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel
    [saved]
    * features: numpy.ndarray of shape (N,K)
    * labels: numpy.ndarray of shape (N)
    '''


    dataset = glob.glob(dataset_path + '*')

    training_time = time.time()

    directory = 'data/tmp/encoder_feats/'
    if not os.path.exists(directory):
        os.makedirs(directory)
        print('creating tmp directory ..')

    pool = multiprocessing.Pool(processes=num_workers)
    args = []

    for i, image_path in enumerate(dataset):
        args.append([i, image_path, encoder, time.time()])

    result_list = pool.map(get_encoder_feature, args)

    feat_dim = np.load(result_list[0][0]).shape[0]
    T    = len(result_list)

    encoder_feat  = np.zeros((T, feat_dim))

    print('Computing deep features ...')

    for result in enumerate(result_list):
        f_name, index = result[1]
        f_data = np.load(f_name)
        encoder_feat[index, :] = f_data

    np.savez(output_path, encoder_feat=encoder_feat)

    print('Done!!')
    print('Feature extraction took '+ str(round(time.time()-training_time))+ ' secs')


def get_encoder_feature(args):
    '''
    Extracts deep features from the prebuilt VGG-16 network.
    This is a function run by a subprocess.
    	[input]
    * i: index of training image
    * image_path: path of image file
    * vgg16: prebuilt VGG-16 network.
    * time_start: time stamp of start time
    	[saved]
    * feat: evaluated deep feature
    '''

    i, image_path, encoder, time_start = args


    # ----- TODO -----


    image = Image.open(image_path)
    image = image_to_tensor(image)

    image = to_var(image)
    encoder_feat = encoder(image).view(-1)

    file_name = os.path.join('./data/tmp/encoder_feats/','encoder_feats_'+str(i)+'.npy')
    np.save(file_name, encoder_feat.data.numpy())

    print('Done Processing image [' + str(i) + ']')

    return [file_name, i]

def get_fake_next_observation(current_object_pose, action, dataset_path='./data/shoe_dataset/9_r', action_space = range(0, 360, 5), stochastic=False, sigma=10):

        next_object_pose = current_object_pose + action

        if stochastic:
            next_object_pose  = int(5*round(np.random.normal(next_object_pose, sigma)/5))

        if (next_object_pose < 0):
            next_object_pose += 360
        next_object_pose %= 360
        next_observation = get_image_from_pose(next_object_pose, dataset_path=dataset_path)
        return next_observation, next_object_pose

def get_image_from_pose(pose, dataset_path='./data/shoe_dataset/9_r'):
    return imageio.imread(dataset_path + str(pose) + '.png')

def get_vgg_feature_for_dataset(vgg16, num_workers=2, dataset_path = './data/atg_dataset.npz'):
    '''
    Creates a trained recognition system by generating training features from all training images.
    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel
    [saved]
    * features: numpy.ndarray of shape (N,K)
    * labels: numpy.ndarray of shape (N)
    '''


    dataset = np.load(dataset_path)
    images_path = dataset ['images_path']
    data = dataset['data']

    training_time = time.time()

    directory = './data/tmp/vgg_feats/'
    if not os.path.exists(directory):
        os.makedirs(directory)
        print('creating tmp directory ..')

    pool = multiprocessing.Pool(processes=num_workers)
    args = []
    i = 0
    for obs_act_obs in data:
        print()
        args.append([i, str(images_path) + str(obs_act_obs[3])+'.png', vgg16, time.time()])
        i +=1

    result_list = pool.map(get_image_feature, args)


    T    = len(result_list)

    deep_feat  = np.zeros((T, 4096))

    print('Computing deep features ...')

    for result in enumerate(result_list):
        f_name, index = result[1]
        f_data = np.load(f_name)
        deep_feat[index, :] = f_data

    np.savez('./data/atg_dataset_vgg_feat.npz',
    	  deep_feat=deep_feat)

    print('Done!!')
    print('Feature extraction took '+ str(round(time.time()-training_time))+ ' secs')

def get_image_feature(args):
	'''
	Extracts deep features from the prebuilt VGG-16 output_pathnetwork.
	This is a function run by a subprocess.
 	[input]
	* i: index of training image
	* image_path: path of image file
	* vgg16: prebuilt VGG-16 network.
	* time_start: time stamp of start time
 	[saved]
	* feat: evaluated deep feature
	'''

	i, image_path, vgg16, time_start = args

	# ----- TODO -----

	image = imageio.imread(image_path)

	if len(image.shape)==2:
		image = np.stack((image,)*3, -1)
	elif(image.shape[2]==4):
		image = image[:, :, :3]

	image = preprocess_image(image)

	image_tensor = torch.autograd.Variable(image)

	vgg_fc7   = torch.nn.Sequential(*list(vgg16.classifier.children())[:-2])
	conv_feat = vgg16.features(image_tensor)
	fc7_feat  = vgg_fc7(conv_feat.view(-1))

	file_name = os.path.join('./data/tmp/vgg_feats/','vgg_feats_'+str(i)+'.npy')
	np.save(file_name, fc7_feat.data.numpy())

	print('Done Processing image [' + str(i) + ']')

	return [file_name, i]

def image_to_tensor(image, image_size=64):

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
    	torchvision.transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)
def preprocess_image(image):
	'''
	Preprocesses the image to load into the prebuilt network.
	[input]
	* image: numpy.ndarray of shape (H,W,3)
	[output]
	* image_processed: torch.Tensor of shape (3,H,W)
	'''

	# ----- TODO -----
	image = skimage.transform.resize(image, (224, 224))
	trans = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	return trans(np.array(image)).unsqueeze(0)

def to_var(x, volatile=False):
	if torch.cuda.is_available():
		x = x.cuda()

	return Variable(x, volatile=volatile)


def imshow(img, display=False):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('autoencoder_output.png')
    if display:
        plt.show()
