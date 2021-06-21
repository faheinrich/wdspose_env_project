from __future__ import division

import argparse
import os

import cv2
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from databases.joint_sets import MuPoTSJoints, CocoExJoints
from training.torch_tools import torch_predict
from util.experiments import load_model, load_transforms
from util.misc import load, save, assert_shape
from util.mx_tools import depth_from_coords
from util.pose import combine_pose_and_trans, extend_hrnet_raw

import matplotlib.pyplot as plt
from util import viz

import torch


def recommended_size(img_shape):
    """
    Calculates the recommended size for a MegaDepth prediction.
    The width will be 512 pixels long, the height the nearest multiple of 32
    """
    new_width = 512
    new_height = img_shape[0] / img_shape[1] * 512
    new_height = round(new_height / 32) * 32
    return new_width, int(new_height)


class ImageFolderDataset:
    def __init__(self, img_folder, metadata, poses_path, depth_folder):
        self.transform = None
        self.images = sorted(os.listdir(img_folder))

        # Load camera parameters
        with open(metadata, 'r') as f:
            data = f.readlines()
            data = [x.split(',') for x in data]
            data = [[y.strip() for y in x] for x in data]
            camera_params = {x[0]: [float(y) for y in x[1:]] for x in data[1:]}

        # Prepare data
        poses2d = []
        fx = []
        fy = []
        cx = []
        cy = []
        img_names = []
        jointwise_depth = []

        pred2d = load(poses_path)
        # print(pred2d)
        for image in self.images:
            poses = [np.array(x['keypoints']).reshape((17, 3)) for x in pred2d[image]]
            poses = np.stack(poses, axis=0)  # (nPoses, 17, 3)
            poses = extend_hrnet_raw(poses)  # (nPoses, 19, 3)

            img = cv2.imread(os.path.join(img_folder, image))
            width, height = recommended_size(img.shape)

            depth = load(os.path.join(depth_folder, image + '.npy'))
            depth = depth_from_coords(depth, poses.reshape((1, -1, 3))[:, :, :2], width, height)  # (nFrames(=1), nPoses*19)
            depth = depth.reshape((-1, 19))  # (nPoses, 19)
            jointwise_depth.append(depth)

            poses2d.append(poses)
            for i, field in enumerate([fx, fy, cx, cy]):
                field.extend([camera_params[image][i]] * len(poses))
            img_names.extend([image] * len(poses))

        self.poses2d = np.concatenate(poses2d).astype('float32')

        self.poses3d = np.ones_like(self.poses2d)[:, :17]
        self.fx = np.array(fx, dtype='float32')
        self.fy = np.array(fy, dtype='float32')
        self.cx = np.array(cx, dtype='float32')
        self.cy = np.array(cy, dtype='float32')
        self.img_names = np.array(img_names)
        self.pred_cdepths = np.concatenate(jointwise_depth).astype('float32')

        self.pose2d_jointset = CocoExJoints()
        self.pose3d_jointset = MuPoTSJoints()

    def __len__(self):
        return len(self.poses2d)

    def __getitem__(self, idx):
        sample = {'pose2d': self.poses2d[idx], 'pose3d': self.poses3d[idx], 'pred_cdepth': self.pred_cdepths[idx],
                  'index': idx}

        if self.transform:
            sample = self.transform(sample)
        # print(sample)
        # print(sample.shape)
        # print("+++++++++++++++++++++++++")

        return sample




def process_like_dataset(image, poses_path, depth_image, transform, camera_params):

    
    # Prepare data
    # fx = []
    # fy = []
    # cx = []
    # cy = []
    img_names = []

    pred2d = load(poses_path)

    img_name = "test.jpg"

    poses = [np.array(x['keypoints']).reshape((17, 3)) for x in pred2d[img_name]]
    poses = np.stack(poses, axis=0)  # (nPoses, 17, 3)
    poses = extend_hrnet_raw(poses)  # (nPoses, 19, 3)

    img = image
    width, height = recommended_size(img.shape)

    depth = depth_image
    depth = depth_from_coords(depth, poses.reshape((1, -1, 3))[:, :, :2], width, height)  # (nFrames(=1), nPoses*19)
    depth = depth.reshape((-1, 19))  # (nPoses, 19)
    jointwise_depth = [depth]
    pred_cdepths = np.concatenate(jointwise_depth).astype('float32')

    poses2d = [poses]
    
    # for i, field in enumerate([fx, fy, cx, cy]):
    #     field.extend([camera_params[image][i]] * len(poses))
    img_names.extend([image] * len(poses))


    poses2d = np.concatenate(poses2d).astype('float32')
    poses3d = np.ones_like(poses2d)[:, :17]

    samples = []
    for i in range(len(poses2d)):
        sample = {'pose2d': poses2d[i], 'pose3d': poses3d[i], 'pred_cdepth': pred_cdepths[i], 'index': i}
        sample = transform(sample)
        # print(sample)
        # print(sample.shape)
        samples.append(sample)
        
    

    return samples


def show_result(img, poses):
    assert_shape(poses, (None, MuPoTSJoints.NUM_JOINTS, 3))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    plt.figure(figsize=(9, 4.5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    ax = viz.subplot(1, 2, 2)
    viz.show3Dpose(poses, MuPoTSJoints(), ax, invert_vertical=True)

    
    plt.savefig("workspace/test.jpg")
    
def setup_camera_params(metadata):
    # Load camera parameters
    with open(metadata, 'r') as f:
        data = f.readlines()
        data = [x.split(',') for x in data]
        data = [[y.strip() for y in x] for x in data]
        camera_params = {x[0]: [float(y) for y in x[1:]] for x in data[1:]}
        return camera_params

def setup_model():
    config, model = load_model('unnormalized')
    model.eval()
    return config, model

def setup_data(config, img_folder, metadata, poses_path, depth_folder):
    test_set = ImageFolderDataset(img_folder, metadata, poses_path, depth_folder)

    transforms = load_transforms('unnormalized', config, test_set) + [lambda x: x['pose2d']]
    test_set.transform = Compose(transforms)

    test_loader = DataLoader(test_set)

    return test_loader, test_set, transforms

def predict_single_image(model, image, poses_path, depth, transforms, out_path, camera_params):
    samples = process_like_dataset(image, poses_path, depth, Compose(transforms), camera_params)
    # print(len(samples))
    stack = torch.stack(samples)
    # print(stack.shape)
    model_input = stack.to('cuda')
    pred = model(model_input)
    # print(pred)
    # print(pred.shape)

    pred = pred.cpu()

    # pred = torch_predict(model, samples, batch_size=len(samples))

    mean3d = torch.from_numpy(transforms[1].normalizer.mean)
    std3d = torch.from_numpy(transforms[1].normalizer.std)
    pred = combine_pose_and_trans(pred, std3d, mean3d, MuPoTSJoints(), 'hip')

    
    result = {"test.jpg" : pred}
    
    save(out_path, result)

    image_path = "workspace/input_imgs/test.jpg"

    show_result(image, pred)

def do_your_thing(model, test_loader, test_set, transforms, img_folder, out_path):
    pred = torch_predict(model, test_loader)

    # print(pred)
    # print(pred.shape)

    mean3d = transforms[1].normalizer.mean
    std3d = transforms[1].normalizer.std
    pred = combine_pose_and_trans(pred, std3d, mean3d, MuPoTSJoints(), 'hip')

    # print(pred.shape)

    result = {}
    for image in test_set.images:
        inds = test_set.img_names == image
        result[image] = pred[inds]
        # print("res[img]", result[image])
        # print("pred[inds]", pred[inds])
        # print(pred[inds].shape)

    save(out_path, result)

    for img in test_set.images:
        # print(img)
        # image = test_set.images[0]
        image_path = os.path.join(img_folder, img)
        image = cv2.imread(image_path)
        show_result(image, result[img])

def main(model, test_loader, transforms, img_folder, metadata, poses_path, depth_folder, out_path, visualize):
    # config, model = load_model('unnormalized')

    # test_set = ImageFolderDataset(img_folder, metadata, poses_path, depth_folder)

    # transforms = load_transforms('unnormalized', config, test_set) + [lambda x: x['pose2d']]
    # test_set.transform = Compose(transforms)

    # test_loader = DataLoader(test_set)
    pred = torch_predict(model, test_loader)

    mean3d = transforms[1].normalizer.mean
    std3d = transforms[1].normalizer.std
    pred = combine_pose_and_trans(pred, std3d, mean3d, MuPoTSJoints(), 'hip')

    result = {}
    for image in test_set.images:
        inds = test_set.img_names == image
        result[image] = pred[inds]

    save(out_path, result)

    if visualize:
        for img in test_set.images:
            # image = test_set.images[0]
            image_path = os.path.join(img_folder, img)
            show_result(image_path, result[img])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img_folder', help="Folder containing the images")
    parser.add_argument('metadata', help="Folder containing the images")
    parser.add_argument('poses_path', help="Path to the output of HR-Net")
    parser.add_argument('depth_folder', help="The depth estimation")
    parser.add_argument('out_path', help="Output path")
    parser.add_argument('-v', '--visualize', help='Visualizes one of the images', action='store_true')
    args = parser.parse_args()

    assert args.out_path.endswith('.pkl')
    main(args.img_folder, args.metadata, args.poses_path, args.depth_folder, args.out_path, args.visualize)

