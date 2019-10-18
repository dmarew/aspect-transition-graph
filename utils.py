import numpy as np
import cv2
import os
from scipy.interpolate import interp1d

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
    return cv2.imread(dataset_path + str(pose) + '.png')

if __name__ == '__main__':
    action_space = range(-180, 180, 45)
    print(action_space)
    current_object_pose = 0
    for i in range(105):
        action = action_space[np.random.randint(len(action_space))]
        prev_observation_pos = current_object_pose
        next_observation, current_object_pose = get_fake_next_observation(current_object_pose, action, stochastic=True, sigma=10)
        print(prev_observation_pos, action, current_object_pose)
        cv2.imshow('next_observation', next_observation)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
