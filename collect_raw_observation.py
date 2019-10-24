#!/usr/bin/env python

import sys
import argparse
import os
import datetime

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError

from atg_camera.srv import *
NODE_NAME = 'collect_raw_data'

def collect_raw_observation(rate=0.5, output_path='data/sample', start_index=0, n_of_samples=10):


    rospy.init_node(NODE_NAME)
    rospy.wait_for_service('current_observation')
    get_current_observation = rospy.ServiceProxy('current_observation', CurrentObservation)
    bridge = CvBridge()
    rate = rospy.Rate(rate)
    saving_dir =  os.path.join(output_path)
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)

    index = start_index
    while True:
        response = get_current_observation()
        try:
            cv2_img = bridge.imgmsg_to_cv2(response.img, "bgr8")
        except CvBridgeError(e):
            print(e)
        else:
            cv2.imwrite(os.path.join(saving_dir, 'obs_'+ str(index) + '.jpeg'), cv2_img)
            print('Done dumping %dth observation'%(index))
        index += 1
        rate.sleep()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="collect raw images")
    parser.add_argument("-r", "--rate", type=float, default=0.5, help="sampling rate")
    parser.add_argument("-si", "--start_index", type=int, default=0, help="starting index")
    parser.add_argument("-ns", "--n_of_samples", type=int, default=10, help="number of samples")
    parser.add_argument("-o", "--output_path", type=str, default='../data/', help="Your destination output file.")
    options = parser.parse_args(sys.argv[1:])

    collect_raw_observation(rate=options.rate,
                            output_path=options.output_path,
                            start_index=options.start_index,
                            n_of_samples=options.n_of_samples)
