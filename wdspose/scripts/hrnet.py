"""
Example script to run HR-net on a subset of a video. You'll need person detections from another source (Mask-RCNN is a good choice).
First clone the following git project:
https://github.com/leoxiaobin/deep-high-resolution-net.pytorch

It needs a pytorch1 environment. To run go to object-passing base dir and call the following:

HRNET_PATH=<path-to-hrnet lib dir> python scripts/hrnet_predict.py --cfg ~/pkgs/deep-hrnet/experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os

HRNET_PATH = os.environ['HRNET_PATH']
sys.path.append(os.path.join(HRNET_PATH, 'lib'))


from scripts import hrnet_dataset

# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

import argparse
import time
import os
import cv2

import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import numpy as np
import models
from config import cfg
from config import update_config
from core.function import AverageMeter
from utils.utils import create_logger
from core.inference import get_final_preds
from utils.transforms import flip_back

from utils.transforms import get_affine_transform

from util.misc import load, ensuredir


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--imgs',
                        help='image folder',
                        required=True,
                        type=str)

    parser.add_argument('--bbox',
                        help='image folder',
                        required=True,
                        type=str)

    parser.add_argument('--out',
                        help='image folder',
                        required=True,
                        type=str)
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly - can't remove these as they are expected by update_config
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def setup_keypoint_db(cfg, image_file, image, boxes):

    img_name = image_file

    image_thre = cfg.TEST.IMAGE_THRE  # bounding boxes lower than this value are not predicted, just thrown away
    soft_nms = cfg.TEST.SOFT_NMS
    oks_thre = cfg.TEST.OKS_THRE
    in_vis_thre = cfg.TEST.IN_VIS_THRE

    # Unpack image size parameters
    image_width = cfg.MODEL.IMAGE_SIZE[0]
    image_height = cfg.MODEL.IMAGE_SIZE[1]
    image_size = np.array(cfg.MODEL.IMAGE_SIZE)
    aspect_ratio = image_width * 1.0 / image_height
    pixel_std = 200

    

    def _xywh2cs(x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    # prep_db
    def _lurb2cs(box):  # TODO check!!!!!1111
        x, y = box[:2]
        w = box[2] - box[0]
        h = box[3] - box[1]
        return _xywh2cs(x, y, w, h)

    kpt_db = []
    for box in boxes:
        score = box[4]
        if score < image_thre:
            continue

        center, scale = _lurb2cs(box[:4])
        kpt_db.append({
            'image': img_name,
            'center': center,
            'scale': scale,
            'score': score,
            'origbox': box[:4]
        })

    return kpt_db, image_size


def process_like_in_loader(image, db_rec, image_size, transform):

    image_file = db_rec['image']

    frame = image

    
    c = db_rec['center']
    s = db_rec['scale']
    score = db_rec['score'] if 'score' in db_rec else 1
    r = 0

    trans = get_affine_transform(c, s, r, image_size)
    input = cv2.warpAffine(frame, trans, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)

    if transform:
        input = transform(input)
    meta = {
        'image': image_file,
        'origbox': db_rec['origbox'],
        'center': c,
        'scale': s,
        'rotation': r,
        'score': score
    }

   

    return input, meta



def predict_single_image(model, config, image, bbox, normalize, detection_thresh):
   
    image_file = "test.jpg"

    # switch to evaluate mode
    model.eval()

    keypoints, image_size = setup_keypoint_db(config, image_file, image, bbox)


    num_samples = len(keypoints)


    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_names = []
    orig_boxes = []
    
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, kpt in enumerate(keypoints):
            input, meta = process_like_in_loader(image, kpt, image_size, normalize)

            input = input[None, ...]

            outputs = model(input)


            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                            flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            num_images = input.size(0)
            # print(num_images)

            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()

            c = meta['center'][None,...]#.numpy()
            s = meta['scale'][None, ...]#.numpy()
            score = meta['score']#.numpy()

            preds, maxvals = get_final_preds(config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals

            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx:idx + num_images, 5] = score

            names = meta['image']
            image_names.extend(names)
            orig_boxes.extend(meta['origbox'])

            idx += num_images

        image_names = []
        for i in range(num_samples):
            image_names.append("test.jpg")
        return all_preds, all_boxes, image_names, orig_boxes


def predict(config, val_loader, val_dataset, model):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_names = []
    orig_boxes = []


    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, meta) in enumerate(val_loader):
            print(meta)
            print("\nhier", input.shape, "\n")
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            num_images = input.size(0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()
            print("c", c)
            print("s", s)
            preds, maxvals = get_final_preds(config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals

            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx:idx + num_images, 5] = score

            names = meta['image']
            image_names.extend(names)
            orig_boxes.extend(meta['origbox'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    i, len(val_loader), batch_time=batch_time)
                print(msg)

        return all_preds, all_boxes, image_names, orig_boxes


def predict_imgs(model, img_folder, bbox_folder, output_file, normalize, valid_dataset, valid_loader):
  

    # start = time.time()
    preds, boxes, image_names, orig_boxes = predict(cfg, valid_loader, valid_dataset, model)
    # end = time.time()
    # print("Time in prediction: " + str(end - start))
    print("preds", preds)
    ensuredir(os.path.dirname(output_file))
    valid_dataset.rescore_and_save_result(output_file, preds, boxes, image_names, orig_boxes)


def setup(img_folder, detection_thresh, bbox_folder):
    args = parse_args()
    update_config(cfg, args)
    cfg.defrost()
    cfg.TEST.MODEL_FILE = HRNET_PATH + '/lib/models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth'
    cfg.TEST.USE_GT_BBOX = False
    cfg.GPUS = (0,)
    cfg.freeze()

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'valid')
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)
    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = transforms.Compose([transforms.ToTensor(), normalize])

    detections = {}
    for file in os.listdir(bbox_folder):
        dets = load(os.path.join(bbox_folder, file))
        assert dets.shape[1] == 5
        img_name = file[:-4]  # remove extension
        detections[img_name] = dets

    valid_dataset = hrnet_dataset.ImgFolderDataset(cfg, img_folder,
                                                   detections,
                                                   normalize, detection_thresh)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        pin_memory=True
    )

    return model, args, normalize, cfg, valid_dataset, valid_loader

def main():
    args = parse_args()
    update_config(cfg, args)
    cfg.defrost()
    cfg.TEST.MODEL_FILE = HRNET_PATH + '/lib/models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth'
    cfg.TEST.USE_GT_BBOX = False
    cfg.GPUS = (0,)
    cfg.freeze()

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'valid')
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)
    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = transforms.Compose([transforms.ToTensor(), normalize])

    predict_imgs(model, args.imgs, args.bbox, args.out, normalize, 0.85)

if __name__ == '__main__':
    main()

