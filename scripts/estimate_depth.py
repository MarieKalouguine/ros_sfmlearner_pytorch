#!/usr/bin/env python

#import roslib
import rospy
import numpy as np
from ros_numpy import numpify, msgify
from sensor_msgs.msg import Image as ROS_Image, CompressedImage as ROS_CImage, CameraInfo
from geometry_msgs.msg import Pose

import torch
from PIL import Image
from imageio import imread
from helpers import Namespace

from models import DispNetS, PoseExpNet

def get_args():
    
    
    args = Namespace()
    
    # Required arguments
    args.add('pretrained', rospy.get_param('~pretrained'))
    args.add('img_height', rospy.get_param('~img_height'))
    args.add('img_width', rospy.get_param('~img_width'))
    args.add('no_resize', rospy.get_param('~no_resize'))
    
    args.add('dataset_dir', rospy.get_param('~dataset_dir'))
    args.add('output_dir', rospy.get_param('~output_dir'))
    
    args.add('img_exts', rospy.get_param('~img_exts'))
    
    return args

class DepthEstimator(object):

    def __init__(self):
        
        #load arguments
        self.args = get_args()
        
        #load neural network
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.disp_net = DispNetS().to(self.device)
        self.pose_net = PoseExpNet().to(self.device)
        
        weights = torch.load(self.args.pretrained, map_location='cpu')
        self.disp_net.load_state_dict(weights['state_dict'])
        self.disp_net.eval()
        
        self.image_sub = rospy.Subscriber("/zed/rgb/image_raw", ROS_Image, self.camera_callback, queue_size=1, buff_size=52428800)  # A big buffer is needed for a real-time estimation (no queue)
        self.info_sub = rospy.Subscriber("/zed/rgb/camera_info", CameraInfo, self.rgb_info_callback, queue_size=1, buff_size=52428800)
        self.odom_sub = rospy.Subscriber("/husky_velocity_controller/odom", Pose, self.odom_callback, queue_size=1, buff_size=52428800)
        
        self.pub_rgb_info = rospy.Publisher('/sfmlearner/rgb/camera_info', CameraInfo, queue_size=1)
        self.pub_rgb = rospy.Publisher('/sfmlearner/rgb/image_raw', ROS_Image, queue_size=1)
        self.pub_depth_info = rospy.Publisher('/sfmlearner/depth/camera_info', CameraInfo, queue_size=1)
        self.pub_depth = rospy.Publisher('/sfmlearner/depth/image_raw', ROS_Image, queue_size=1)
        self.pub_disp = rospy.Publisher('/sfmlearner/disp/image_raw', ROS_Image, queue_size=1)
        
        self.camera_link = "camera_link"

    def rgb_info_callback(self, data):
        data.header.frame_id = self.camera_link
        self.pub_rgb_info.publish(data)
        self.pub_depth_info.publish(data)
        
    def camera_callback(self, data):
        
        np_image = imread(bytes(data.data))
        output = self.run_inference(np_image)
        output = output.detach().cpu()
        
        np_disp = output[0].numpy()
        ros_disp = msgify(ROS_Image, np_disp, encoding='32FC1')
        
        np_depth = 1/np_disp
        #print(np_depth[np_depth.shape[0]//2][np_depth.shape[1]//2])
        ros_depth = msgify(ROS_Image, np_depth, encoding='32FC1')
        
        ros_image = msgify(ROS_Image, np_image, encoding='rgb8')
        
        ros_depth.header.frame_id = self.camera_link
        ros_disp.header.frame_id = self.camera_link
        ros_image.header.frame_id = self.camera_link
        self.pub_disp.publish(ros_disp)
        self.pub_depth.publish(ros_depth)
        self.pub_rgb.publish(ros_image)
    
    def odom_callback(self, data):
        pose = data.pose.pose
        
    def run_inference(self, img):
        """ Returns the estimation of depth & disparity by using the dispnet. """
        h,w,_ = img.shape
        h1 = self.args.img_height if self.args.img_height else h
        w1 = self.args.img_width if self.args.img_width else w
        if (not self.args.no_resize) and (h != h1 or w != w1):
            img = np.array(Image.fromarray(img).resize((h1, w1)))
        img = np.transpose(img, (2, 0, 1))

        tensor_img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        tensor_img = ((tensor_img/255 - 0.5)/0.5).to(self.device)
        
        disp = self.disp_net(tensor_img)[0]
        pose, _ = self.pose_net(tensor_img)[0]
        print(pose)
        return disp
    

def main():
    
    rospy.init_node('depth_estimation_node', anonymous=True)
    depth_estimator_object = DepthEstimator()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()
