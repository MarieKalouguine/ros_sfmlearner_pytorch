import argparse
import numpy as np
from pebble import ProcessPool
from tqdm import tqdm
from path import Path
from imageio import imwrite


parser = argparse.ArgumentParser()
parser.add_argument("dataset_dir", metavar='DIR',
                    help='path to original dataset')
parser.add_argument("--dataset-format", type=str, default='rosbag', choices=["kitti", "cityscapes", "rosbag"])
parser.add_argument("--with-depth", action='store_true',
                    help="If available (e.g. with KITTI), will store depth ground truth along with images, for validation")
parser.add_argument("--with-pose", action='store_true',
                    help="If available (e.g. with KITTI), will store pose ground truth along with images, for validation")
parser.add_argument("--no-train-gt", action='store_true',
                    help="If selected, will delete ground truth depth to save space")
parser.add_argument("--dump-root", type=str, default='dump', help="Where to dump the formatted data")
parser.add_argument("--height", type=int, default=0, help="image height")
parser.add_argument("--width", type=int, default=0, help="image width")
parser.add_argument("--depth-size-ratio", type=int, default=1, help="will divide depth size by that ratio")
parser.add_argument("--num-threads", type=int, default=4, help="number of threads to use")

#The following arguments are only useful if the dataset is availible as rosbags
parser.add_argument("--image-topic", type=str, default='/camera/color/image_raw/compressed',
                    help="ROS topic containing the images from the camera")
parser.add_argument("--cam-info-topic", type=str, default='/camera/color/camera_info',
                    help="ROS topic containing the camera intrinsics (the camera must be calibrated)")
parser.add_argument("--depth-topic", type=str, default='/camera/depth/image_rect_raw/compressed',
                    help="ROS topic containing the ground truth depth images")
parser.add_argument("--odom-topic", type=str, default='/husky_velocity_controller/odom',
                    help="ROS topic containing the ground truth odometry")
parser.add_argument("--use-tf", action='store_true',
                    help="Use tf readings in order to get the camera's odometry")
parser.add_argument("--camera-link", type=str, default='camera_link',
                    help="The link of the camera in the tf structure")

args = parser.parse_args()

def dump_example(args, drive):
    scene_list = data_loader.collect_scenes(drive)
    for scene_data in scene_list:
        dump_dir = args.dump_root/scene_data['rel_path']
        dump_dir.makedirs_p()
        intrinsics = scene_data['intrinsics']

        dump_cam_file = dump_dir/'cam.txt'

        np.savetxt(dump_cam_file, intrinsics)
        poses_file = dump_dir/'poses.txt'
        poses = []
        for sample in data_loader.get_scene_imgs(scene_data):
            img, frame_nb = sample["img"], sample["id"]
            dump_img_file = dump_dir/'{}.jpg'.format(frame_nb)
            dump_img_file.parent.makedirs_p()
            imwrite(dump_img_file, img)

            if "pose" in sample.keys():
                poses.append(sample["pose"].tolist())
            if "depth" in sample.keys():
                dump_depth_file = dump_dir/'{}.npy'.format(frame_nb)
                np.save(dump_depth_file, sample["depth"])
        if len(poses) != 0:
            np.savetxt(poses_file, np.array(poses).reshape(-1, 12), fmt='%.6e')

        if len(dump_dir.files('*.jpg')) < 3:
            dump_dir.rmtree()


def main():
    args.dump_root = Path(args.dump_root)   #Path where to dump the data
    args.dump_root.mkdir_p()    #Create the path recursively if it does not exist yet

    #Import the data loader object depending on the dataset chosen by the user :
    global data_loader

    if args.dataset_format == 'kitti':
        from kitti_raw_loader import KittiRawLoader
        data_loader = KittiRawLoader(args.dataset_dir,
                                     static_frames_file=args.static_frames,
                                     img_height=args.height,
                                     img_width=args.width,
                                     get_depth=args.with_depth,
                                     get_pose=args.with_pose,

                                     depth_size_ratio=args.depth_size_ratio)
    if args.dataset_format == 'cityscapes':
        from cityscapes_loader import cityscapes_loader
        data_loader = cityscapes_loader(args.dataset_dir,
                                        img_height=args.height,
                                        img_width=args.width)
    if args.dataset_format == 'rosbag':
        from rosbag_loader import RosbagLoader
        data_loader = RosbagLoader(args.dataset_dir,
                                    get_depth=args.with_depth,
                                    get_pose=args.with_pose,
                                    image_topic=args.image_topic,
                                    cam_info_topic=args.cam_info_topic,
                                    depth_topic=args.depth_topic,
                                    odom_topic=args.odom_topic,
                                    img_height=args.height,
                                    img_width=args.width,
                                    use_tf=args.use_tf,
                                    camera_link=args.camera_link,
                                    depth_size_ratio=args.depth_size_ratio)

    n_scenes = len(data_loader.scenes)  #A scene is a bunch of frames that correspond to a continuous movement.
    print('Found {} potential scenes'.format(n_scenes))
    print('Retrieving frames')
    if args.num_threads == 1:   #If no multithreading
        for scene in tqdm(data_loader.scenes):  #Tqdm (taqaddum) shows a progress bar associated to the for loop
            dump_example(args, scene)   #Write the images into folders
    else:
        with ProcessPool(max_workers=args.num_threads) as pool:
            tasks = pool.map(dump_example, [args]*n_scenes, data_loader.scenes)
            try:
                for _ in tqdm(tasks.result(), total=n_scenes):
                    pass
            except KeyboardInterrupt as e:
                tasks.cancel()
                raise e

    print('Generating train val lists')
    np.random.seed(8964)
    # to avoid data snooping, we will make two cameras of the same scene to fall in the same set, train or val
    subdirs = args.dump_root.dirs()
    canonic_prefixes = set([subdir.basename()[:-2] for subdir in subdirs])
    with open(args.dump_root / 'train.txt', 'w') as tf:
        with open(args.dump_root / 'val.txt', 'w') as vf:
            for pr in tqdm(canonic_prefixes):
                corresponding_dirs = args.dump_root.dirs('{}*'.format(pr))
                if np.random.random() < 0.1:
                    for s in corresponding_dirs:
                        vf.write('{}\n'.format(s.name))
                else:
                    for s in corresponding_dirs:
                        tf.write('{}\n'.format(s.name))
                        if args.with_depth and args.no_train_gt:
                            for gt_file in s.files('*.npy'):
                                gt_file.remove_p()


if __name__ == '__main__':
    main()
