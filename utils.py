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

def get_encoder_feature_for_dataset(encoder, num_workers=2, dataset_path = './data/atg_dataset.npz'):
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

    directory = './data/tmp/encoder_feats/'
    if not os.path.exists(directory):
        os.makedirs(directory)
        print('creating tmp directory ..')

    pool = multiprocessing.Pool(processes=num_workers)
    args = []
    i = 0
    for obs_act_obs in data:
        print()
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

    np.savez('./data/atg_dataset_encoder_feat.npz',
    	  deep_feat=deep_feat)

    print('Done!!')
    print('Feature extraction took '+ str(round(time.time()-training_time))+ ' secs')


def get_image_feature(args):
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
