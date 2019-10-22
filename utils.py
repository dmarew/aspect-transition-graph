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
from sklearn.cluster import *
from scipy.spatial import distance

def get_belief_given_observation(image, encoder, aspect_nodes):
    pass

def get_aspect_nodes(features, clustering_labels):

    num_aspect_nodes   = np.max(clustering_labels) + 1
    aspect_nodes = []

    for i in range(num_aspect_nodes):
        ith_aspects = features[clustering_labels==i]
        mean_aspect_node = ith_aspects.mean(axis=0, keepdims=True)
        print(ith_aspects.shape, mean_aspect_node.shape)
        distance_to_mean = distance.cdist(ith_aspects ,
                                          mean_aspect_node,
                                          metric='euclidean')
        print('min_distance: ', distance_to_mean.shape, distance_to_mean==distance_to_mean.min())
        token_aspect_node_index = np.where(distance_to_mean==distance_to_mean.min())
        print(i, token_aspect_node_index)
        aspect_nodes.append(token_aspect_node_index)


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
                                         output_path = 'data/encoder_clustering_result.npz'):

    atg_dataset_feat = np.load(encoder_feats_path)
    data = atg_dataset_feat['deep_feat']
    action_seq = atg_dataset_feat['action_seq']
    clustering_time = time.time()
    if clustering_algorithm == 'DBSCAN':
        clustering = DBSCAN(eps=3, min_samples=12).fit(data)
        print(clustering.labels_)
        print('clustering cores ', clustering.components_.shape)
        print('clustering took ', time.time()-clustering_time, ' secs')
        print('Found ', np.max(clustering.labels_) + 1, ' unique aspect nodes')
        np.savez(output_path, clustering_labels = clustering.labels_, action_seq=action_seq)

        get_aspect_nodes(clustering.components_, clustering.labels_)

    else:
        raise NotImplementedError(clustering_algorithm + " has not been implemented yet!!")

def get_encoder_feature_for_dataset(encoder,
                                    num_workers=2,
                                    dataset_path = 'data/atg_dataset.npz',
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


    dataset = np.load(dataset_path)
    images_path = dataset ['images_path']
    data = dataset['data']
    action_seq = data[:, 2]
    training_time = time.time()

    directory = './data/tmp/encoder_feats/'
    if not os.path.exists(directory):
        os.makedirs(directory)
        print('creating tmp directory ..')

    pool = multiprocessing.Pool(processes=num_workers)
    args = []
    i = 0
    for obs_act_obs in data:
        #print()
        args.append([i, str(images_path) + str(obs_act_obs[3])+'.png', encoder, time.time()])
        i +=1

    result_list = pool.map(get_encoder_feature, args)


    T    = len(result_list)

    deep_feat  = np.zeros((T, 6912))

    print('Computing deep features ...')

    for result in enumerate(result_list):
        f_name, index = result[1]
        f_data = np.load(f_name)
        deep_feat[index, :] = f_data

    np.savez(output_path, deep_feat=deep_feat, action_seq=action_seq)

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

    image = imageio.imread(image_path)

    if len(image.shape)==2:
    	image = np.stack((image,)*3, -1)
    elif(image.shape[2]==4):
    	image = image[:, :, :3]

    image = preprocess_image_atg(image)

    image_tensor = to_var(image)
    encoder_feat = encoder(image_tensor).view(-1)

    file_name = os.path.join('./data/tmp/encoder_feats/','encoder_feats_'+str(i)+'.npy')
    np.save(file_name, encoder_feat.data.numpy())

    #print('Done Processing image [' + str(i) + ']')

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

def preprocess_image_atg(image):
	'''
	Preprocesses the image to load into the prebuilt network.
	[input]
	* image: numpy.ndarray of shape (H,W,3)
	[output]
	* image_processed: torch.Tensor of shape (3,H,W)
	'''

	# ----- TODO -----
	image = image/255.0
	trans = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor()
	])
	return trans(np.array(image)).unsqueeze(0).float()
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
if __name__ =='__main__':
    encoder_feats_path = 'data/atg_dataset_encoder_feat.npz'
    clustering_result_path = 'data/encoder_clustering_result.npz'
    print('clustering ...')
    cluster_observations_to_aspect_nodes(encoder_feats_path, clustering_algorithm = 'DBSCAN', output_path = clustering_result_path)
    print('build aspect transition graph ...')
    atg = build_aspect_transition_graph(clustering_result_path)
    plt.bar(np.arange(72), atg[56, 5, :])
    plt.show()
