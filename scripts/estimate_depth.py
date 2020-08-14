#!/usr/bin/env python

#import roslib
import rospy
import numpy as np
from ros_numpy import numpify, msgify
from sensor_msgs.msg import Image as ROS_Image, CompressedImage as ROS_CImage, CameraInfo
from nav_msgs.msg import Odometry

import torch
from PIL import Image
from imageio import imread
from helpers import *
import matplotlib.pyplot as plt

from models import DispNetS, PoseExpNet

def get_args():
    
    
    args = Namespace()
    
    # Required arguments
    args.add('pretrained_disp', rospy.get_param('~pretrained_disp'))
    posenet = rospy.get_param('~pretrained_pose')
    odom_topic = rospy.get_param('~odom_topic')
    args.add('use_pose', posenet and odom_topic)
    if args.use_pose:
        args.add('pretrained_posenet', posenet)
        args.add('odom_topic', odom_topic)
    if (not posenet and odom_topic):
        rospy.logerr("You must specify a pretrained pose network!")
        rospy.signal_shutdown()
    if (not odom_topic and posenet):
        rospy.logerr("You must specify an odometry topic!")
        rospy.signal_shutdown()
    image_topic = rospy.get_param('~image_topic')
    if not image_topic:
        rospy.logerr("You must specify the image topic!")
        rospy.signal_shutdown()
    camera_info_topic = rospy.get_param('~camera_info_topic')
    if not camera_info_topic:
        rospy.logerr("You must specify the camera info topic!")
        rospy.signal_shutdown()
    args.add('image_topic', image_topic)
    args.add('camera_info_topic', camera_info_topic)
    args.add('img_height', rospy.get_param('~img_height'))
    args.add('img_width', rospy.get_param('~img_width'))
    args.add('camera_link', rospy.get_param('~camera_link'))
    
    return args

class DepthEstimator(object):

    def __init__(self):
        
        #load arguments
        self.args = get_args()
        
        #load neural network
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.disp_net = DispNetS().to(self.device)
        weights = torch.load(self.args.pretrained_disp, map_location='cpu')
        self.disp_net.load_state_dict(weights['state_dict'])
        self.disp_net.eval()
        
        if self.args.use_pose:
            weights = torch.load(self.args.pretrained_posenet)
            seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
            self.pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(self.device)
            self.seq_length = seq_length
            self.pose_net.load_state_dict(weights['state_dict'], strict=False)
        
        if self.args.image_topic[-11:]=="/compressed":
            self.image_sub = rospy.Subscriber(self.args.image_topic, ROS_CImage, self.camera_callback, queue_size=1, buff_size=52428800)
        else:
            self.image_sub = rospy.Subscriber(self.args.image_topic, ROS_Image, self.camera_callback, queue_size=1, buff_size=52428800)
        self.info_sub = rospy.Subscriber(self.args.camera_info_topic, CameraInfo, self.rgb_info_callback, queue_size=1, buff_size=52428800)
        if self.args.use_pose:
            self.odom_sub = rospy.Subscriber(self.args.odom_topic, Odometry, self.odom_callback, queue_size=1, buff_size=52428800)
        
        self.pub_rgb_info = rospy.Publisher('/sfmlearner/rgb/camera_info', CameraInfo, queue_size=1)
        self.pub_rgb = rospy.Publisher('/sfmlearner/rgb/image_raw', ROS_Image, queue_size=1)
        self.pub_depth_info = rospy.Publisher('/sfmlearner/depth/camera_info', CameraInfo, queue_size=1)
        self.pub_depth = rospy.Publisher('/sfmlearner/depth/image_raw', ROS_Image, queue_size=1)
        self.pub_disp = rospy.Publisher('/sfmlearner/disp/image_raw', ROS_Image, queue_size=1)
        
        self.imu_position_t = []    #ground truth pose at t
        self.imu_position_t_1 = []  #ground truth pose at t-1
        self.imu_position_t_2 = []  #ground truth pose at t-2
        self.net_position_t_2 = []
        self.counter = 0
        self.ratio = 1

    def rgb_info_callback(self, data):
        data.header.frame_id = self.args.camera_link
        self.pub_rgb_info.publish(data)
        self.pub_depth_info.publish(data)
        
    def camera_callback(self, data):
        if self.args.use_pose and len(self.net_position_t_2)>0 and len(self.imu_position_t_2)>0:
            imu_shift = np.linalg.norm(self.imu_position_t - self.imu_position_t_2)
            net_shift = np.linalg.norm(self.net_position_t - self.net_position_t_2)
            if imu_shift and net_shift:
                self.ratio = (self.counter*self.ratio + imu_shift/net_shift)/(self.counter+1)
                self.counter += 1
        
        np_image = imread(bytes(data.data))
        disp, pose = self.run_inference(np_image)   #This is where the magic happens
        if self.args.use_pose:
            net_poses = pose.detach().cpu()
        
            #the following poses are actually estimated for t-1 and t+1, but are used at the next iteration, hence the names
            i = int((self.seq_length-1)/2)
            self.net_position_t_2 = net_poses[0,i-1,:3].numpy() #estimated pose is in 6DoF format (3 for translation and 3 for euler rotation)
            self.net_position_t = net_poses[0,i,:3].numpy()
            
        disp = disp.detach().cpu()
        np_disp = disp[0].numpy() / self.ratio
        ros_disp = msgify(ROS_Image, np_disp, encoding='32FC1')
        
        np_depth = 1/np_disp
        ros_depth = msgify(ROS_Image, np_depth, encoding='32FC1')
        
        ros_image = msgify(ROS_Image, np_image, encoding='rgb8')
        
        ros_depth.header.frame_id = self.args.camera_link
        ros_disp.header.frame_id = self.args.camera_link
        ros_image.header.frame_id = self.args.camera_link
        self.pub_disp.publish(ros_disp)
        self.pub_depth.publish(ros_depth)
        self.pub_rgb.publish(ros_image)
    
    def odom_callback(self, data):
        self.imu_position_t_2 = self.imu_position_t_1
        self.imu_position_t_1 = self.imu_position_t
        pose_msg = data.pose.pose
        self.imu_position_t = matrix_from_Pose_msg(pose_msg).transpose()[3][:3]
        
    def run_inference(self, img):
        """ Returns the estimation of depth & disparity by using the dispnet. """
        h,w,_ = img.shape
        h1 = self.args.img_height if self.args.img_height else h
        w1 = self.args.img_width if self.args.img_width else w
        img = np.array(Image.fromarray(img).resize((h1, w1)))
        img = np.transpose(img, (2, 0, 1))

        tensor_img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        tensor_img = ((tensor_img/255 - 0.5)/0.5).to(self.device)
        disp = self.disp_net(tensor_img)[0]
        pose = []
        if self.args.use_pose:
            pose = self.pose_net(tensor_img, [tensor_img]*(self.seq_length-1))[1]
        return disp, pose
    

def main():
    
    rospy.init_node('depth_estimation_node', anonymous=True)
    depth_estimator_object = DepthEstimator()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()
