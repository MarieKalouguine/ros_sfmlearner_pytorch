from __future__ import division
import numpy as np
from path import Path
from skimage.transform import resize as imresize
from imageio import imread
from tqdm import tqdm

import rosbag
from ros_numpy import numpify
from sensor_msgs.msg import CompressedImage as ROS_Image
from geometry_msgs.msg import Pose, Transform
from rospy.rostime import Time

    
def translation_matrix(translation):
    """Return matrix to translate by direction vector."""
    M = np.identity(4)
    M[:3, 3] = translation[:3]
    return M

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion."""
    q =np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq =np.dot(q, q)
    if nq < 0.01:
        return np.identity(4)
    q *= np.sqrt(2.0 / nq)
    q =np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

def matrix_from_Pose_msg(pose_msg):
    position = [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]
    orientation = [pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w]
    return translation_matrix(position) @ quaternion_matrix(orientation)

def matrix_from_Transform_msg(tf_msg):
    translation = [tf_msg.translation.x, tf_msg.translation.y, tf_msg.translation.z]
    rotation = [tf_msg.rotation.x, tf_msg.rotation.y, tf_msg.rotation.z, tf_msg.rotation.w]
    return translation_matrix(translation) @ quaternion_matrix(rotation)

def pose_from_Pose_and_Transform(pose_msg, tf_msg):
    """
    Return the pose after the transformation of a geometry_msgs/Pose by a geometry_msgs/Tranform.
    Return is a 4x4 matrix.
    """
    return matrix_from_Pose_msg(pose_msg) @ matrix_from_Transform_msg(tf_msg)


class RosbagLoader(object):
    def __init__(self,
                 dataset_dir='/home/interns/learning_data/rosbags',
                 img_height=480,
                 img_width=640,
                 get_depth=True,
                 get_pose=True,
                 depth_size_ratio=1):
        
        self.dataset_files = Path(dataset_dir).files()
        self.image_topic = '/camera/color/image_raw/compressed'
        self.depth_topic = '/camera/depth/image_rect_raw/compressed'
        self.odom_topic = '/husky_velocity_controller/odom'
        self.cam_info_topic = '/camera/color/camera_info'
        self.img_height = img_height
        self.img_width = img_width
        self.min_speed = 0.3
        self.get_depth = get_depth
        self.get_pose = get_pose
        self.depth_size_ratio = depth_size_ratio
        self.collect_train_files()    #loading the file names in self.scenes
        
    def collect_train_files(self):    #Search for the names of the files containing the different scenes
        self.scenes = []
        for scene in self.dataset_files:
            self.scenes.append(scene)

    def collect_scenes(self, drive):    #Create a dictionary for the data from a given scene/drive, including all images from the camera
        bag_object = rosbag.Bag(drive)
        scene_data = {'drive': drive.name[:-4], 'bag': bag_object, 'rel_path': drive.name[:-4], 'pose': [], 'intrinsics': [], 'frame_id': [], 'depth_frame_id': [], 'pose':[]}
        
        known_transform = False
        known_position = False
        known_depth = False
        missing_poses = []
        missing_depths = []
        for msg in scene_data['bag'].read_messages(topics=["/tf", self.image_topic, self.depth_topic, self.odom_topic, self.cam_info_topic]):
            
            if known_position and known_depth:
                if msg.topic == self.image_topic:
                    scene_data['frame_id'].append(msg.timestamp)
                    missing_poses.append(msg.timestamp)
                    missing_depths.append(msg.timestamp)
                  
            if msg.topic == self.depth_topic:
                if known_depth:
                    last_depth_timestamp = depth_timestamp
                depth_timestamp = msg.timestamp
                for t in missing_depths: #all timestamps corresponding to the images read since the last depth reading
                    if (t-last_depth_timestamp) < (depth_timestamp-t):
                        scene_data['depth_frame_id'].append(last_depth_timestamp)
                    else:
                        scene_data['depth_frame_id'].append(depth_timestamp)
                missing_depths = []
                known_depth = True
                    
            if known_transform and msg.topic == self.odom_topic:
                if known_position:
                    last_odom_timestamp = odom_timestamp
                    last_pose = pose
                odom_timestamp = msg.timestamp
                pose_msg = msg.message.pose.pose
                pose = pose_from_Pose_and_Transform(pose_msg, tf_msg)
                for t in missing_poses: #all timestamps corresponding to the images read since the last odometry reading
                    ratio = (t - last_odom_timestamp)/(odom_timestamp - last_odom_timestamp)
                    computed_pose = ratio*last_pose + (1-ratio)*pose
                    scene_data['pose'].append(computed_pose)
                missing_poses = []
                known_position = True
                
            if msg.topic=="/tf":
                for tf in msg.message.transforms:
                    if  tf.header.frame_id=="base_link" and tf.child_frame_id=="camera_link":
                        tf_msg = tf.transform
                        known_transform = True
            
            if msg.topic == self.cam_info_topic:
                scene_data['intrinsics'] = msg.message.K
        
        if known_position:
            for t in missing_poses: #if there are still images read since the last odometry message of the bag
                ratio = (t - odom_timestamp)/(odom_timestamp - last_odom_timestamp)
                computed_pose = ratio*last_pose + (1-ratio)*pose
                scene_data['pose'].append(computed_pose)
            
        if len(scene_data['frame_id'])==0:  #Check that the scene is not empty
            return []

        return [scene_data]

    def get_scene_imgs(self, scene_data):   #Collect the images of a given scene, taking speed into account
        
        selected_frames = [0]
        last_pose = scene_data['pose'][0]
        for i, pose in enumerate(scene_data['pose'][1:]):
            transform = np.linalg.inv(last_pose) @ pose  #pose transform matrix from the last selected frame
            speed_mag = np.linalg.norm(transform - np.eye(4))
            if speed_mag > self.min_speed:  #If the last frame was taken far enough
                selected_frames.append(i+1) #register this frame as a valid one
                last_pose = pose
        
        nb_frames = len(selected_frames)
        k = 0
        i = 0
        read_image = False
        read_depth = False
        sample  = {}
        for msg in scene_data['bag'].read_messages(topics=[self.image_topic, self.depth_topic]):
            
            if read_image and read_depth:
                sample['id'] = str(scene_data['frame_id'][i])
                sample['img'] = np_img    #get the rgb image
                if self.get_depth:
                    sample['depth'] = np_depth    #get ground truth depth map
                if self.get_pose:
                    sample['pose'] = scene_data['pose'][i][:3]    #get ground truth pose (we remove the last line of the matrix, which is always [0, 0, 0, 1] and carries no information)
                yield sample
                read_image = False
                read_depth = False
                k+=1
                if k >= nb_frames:
                    break
                i = selected_frames[k]
            
            if msg.topic == self.image_topic and msg.timestamp==scene_data['frame_id'][i]:
                msg.message.__class__ = ROS_Image
                np_img = imread(bytes(msg.message.data))
                np_img = imresize(np_img, (self.img_height, self.img_width))  #resize the image to the size provided by the user
                np_img = (np_img * 255).astype(np.uint8)
                read_image = True
                
            if msg.topic == self.depth_topic and msg.timestamp==scene_data['depth_frame_id'][i]:
                msg.message.__class__ = ROS_Image
                np_depth = imread(bytes(msg.message.data))
                np_depth = imresize(np_depth, (self.img_height, self.img_width))  #resize the image to the size provided by the user
                read_depth = True
