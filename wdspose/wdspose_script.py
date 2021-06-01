# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import sys

import cv2

os.environ['DETECTRON_PATH'] = '../Detectron'
os.environ['HRNET_PATH'] = '../deep-high-resolution-net.pytorch'

import json


import torch
import glob
import shutil


import scripts.maskrcnn
import scripts.hrnet
import scripts.megadepth
import scripts.predict

import matplotlib.pyplot as plt

import time
import numpy as np

import detectron.core.test_engine as infer_engine
import detectron.utils.c2 as c2_utils

from util.misc import save


def main():

    with torch.no_grad():
        GPU_ID = 0

        # img_folder = "examples/imgs"
        # metadata = "examples/metadata.csv"

        img_folder = "workspace/input_imgs"
        metadata = "workspace/metadata.csv"

        keypoints = "workspace/keypoints.json"
        model_yaml = "../deep-high-resolution-net.pytorch/experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml"

        depth_folder = "workspace/depth"
        bboxes = "workspace/bboxes"

        hrnet_threshold = 0.85

        depth_model = scripts.megadepth.load_model().eval()

        maskrcnn_model = scripts.maskrcnn.load_model()

        sys.argv.append("--imgs")
        sys.argv.append(img_folder)
        sys.argv.append("--bbox")
        sys.argv.append(bboxes)
        sys.argv.append("--out")
        sys.argv.append(keypoints)
        sys.argv.append("--cfg")
        sys.argv.append(model_yaml)


 
        # src = "workspace/dummy.json"
        # dst =  "workspace/keypoints.json"
        # shutil.copyfile(src, dst)
        
        hrnet_model, hrnet_args, hrnet_normalize, hrnet_config, hrnet_dataset, hrnet_loader = scripts.hrnet.setup(img_folder, hrnet_threshold, bboxes)
        # hrnet_model, hrnet_args, hrnet_normalize, hrnet_config = scripts.hrnet.setup(img_folder, hrnet_threshold, bboxes)

        camera_params = scripts.predict.setup_camera_params(metadata)
        wdspose_config, wdspose_model = scripts.predict.setup_model()

        # test_loader, test_set, wdspose_transforms = scripts.predict.setup_data(wdspose_config, img_folder, metadata, keypoints, depth_folder)
        test_loader, test_set, wdspose_transforms = scripts.predict.setup_data(wdspose_config, img_folder, metadata, keypoints, depth_folder)



        cap = cv2.VideoCapture(0)            


        print("start")
        for i in range(10000):
            # time.sleep(5)
            ges_time = time.time()


            # cam_time = time.time()
            # capture webcam image
            ret, frame = cap.read()
            if not ret:
                print("webcam failed")
                exit()
          
            # cv2.imwrite("workspace/input_imgs/test.jpg", frame)
            # cv2.imwrite("workspace/input_imgs/test.jpg", frame)
            # cap.release()
            # print("cam time:", time.time() - cam_time)


            # depthtime = time.time()
            # print("depth")
            # calculate depth image
            

            # for img_f in glob.glob(img_folder + "/*"):
            # img = cv2.imread(img_f)

            d_width, d_height = scripts.megadepth.recommended_size(frame.shape)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (d_width, d_height))
            img = np.expand_dims(img, 0)
            depth_image = scripts.megadepth.predict_depth(img, depth_model, batch_size=1)
            # np.save(os.path.join("workspace/depth", '%s.npy' % img_f.split("/")[-1]), depth_image)
            # scripts.megadepth.predict(depth_model, img_folder, depth_folder)
            # print("depth_time", time.time() - depthtime)
            
            #torch.cuda.empty_cache()

    
            # boxtime = time.time()
            # print("bboxes")
            # calculate bounding boxes
            
            # for img_f in glob.glob(img_folder + "/*"):
            with c2_utils.NamedCudaScope(GPU_ID):
                cls_boxes, _, _ = infer_engine.im_detect_all(maskrcnn_model, frame, None)
            
            # save(os.path.join("workspace/bboxes", "%s.pkl" % img_f.split("/")[-1]), cls_boxes[1])
                # torch.cuda.empty_cache()
            # print("box_time", time.time() - boxtime)
            # scripts.maskrcnn.predict(maskrcnn_model, img_folder, bboxes)
            
          

            #torch.cuda.empty_cache()
            # hrnettime = time.time()
            # print("hrnet")
     

            bbox = cls_boxes[1]
            all_preds, all_boxes, image_names, orig_boxes = scripts.hrnet.predict_single_image(hrnet_model, hrnet_config, frame, bbox, hrnet_normalize, 0.85)
            hrnet_dataset.rescore_and_save_result(keypoints, all_preds, all_boxes, image_names, orig_boxes)
            
            # exit()
            # scripts.hrnet.predict_imgs(hrnet_model, img_folder, bboxes, keypoints, hrnet_normalize, hrnet_dataset, hrnet_loader)
            # keypoints = scripts.hrnet.predict_imgs(hrnet_model, hrnet_args.imgs, hrnet_args.bbox , hrnet_args.out, hrnet_normalize, hrnet_dataset, hrnet_loader)
            # print("2dpose", time.time() - hrnettime)
            
            
            
            # wdsposetime = time.time()
            # print("start wdspose")
         
            #torch.cuda.empty_cache()

            scripts.predict.predict_single_image(wdspose_model, frame, keypoints, depth_image, wdspose_transforms, "workspace/results.pkl", camera_params)

            # exit()
            # test_loader, test_set, wdspose_transforms = scripts.predict.setup_data(wdspose_config, img_folder, metadata, keypoints, depth_folder)
            # scripts.predict.do_your_thing(wdspose_model, test_loader, test_set, wdspose_transforms, img_folder, "workspace/results.pkl")
            # print("wdspose", time.time() - wdsposetime)

            # exit()
            # torch.cuda.empty_cache()
            
            print("gesamt:", time.time() - ges_time, "freq:", 1/(time.time() - ges_time))
            # exit()

if __name__ == "__main__":
    main()

