from utils import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    action_space = [-180, -135, -90, -45, 45, 90, 135, 180]
    images_path = './data/shoe_dataset/9_r'
    current_object_pose = 0
    dataset = {}
    dataset['action_space'] = action_space
    dataset['images_path']  = images_path
    data = []
    for i in range(10000):
        action_index = np.random.randint(len(action_space))
        action = action_space[action_index]
        prev_observation_pos = current_object_pose
        next_observation, current_object_pose = get_fake_next_observation(current_object_pose, action, dataset_path=images_path, stochastic=True, sigma=20)
        print(i, prev_observation_pos, action, action_index, current_object_pose)
        data.append([prev_observation_pos, action, action_index, current_object_pose])
    data = np.array(data)
    np.savez('./data/atg_dataset', data=data, action_space = action_space, images_path=images_path)
    print('writing dataset done!!')

    plt.hist(data[:, 3], 72)
    plt.show()
