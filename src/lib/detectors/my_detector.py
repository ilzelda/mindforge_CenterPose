from .object_pose import ObjectPoseDetector
from lib.utils.debugger import Debugger
import time
import numpy as np
import torch
import cv2
import math

class MyDebugger(Debugger):
    def __init__(self, *args, **kwargs):
        super(MyDebugger, self).__init__(*args, **kwargs)
        # self.edges = [
        #     [1, 2], [2, 3], [3, 4], [4, 1],  # 윗면
        #     [5, 6], [6, 7], [7, 8], [8, 5],  # 아랫면
        #     [1, 5], [2, 6], [3, 7], [4, 8]   # 수직 모서리
        # ]

        # objectron 기준
        self.edges = [[2, 4], [2, 6], [6, 8], [4, 8],
                      [1, 2], [3, 4], [5, 6], [7, 8],
                      [1, 3], [1, 5], [3, 7], [5, 7]]


    def add_coco_hp(self, points, img_id='default', pred_flag='pred', PAPER_DISPLAY=False):
        points = np.array(points, dtype=np.int32).reshape(self.num_joints, 2)

        # Draw corner points
        for j in range(self.num_joints):
            # cv2.circle(self.imgs[img_id],
            #             (points[j, 0], points[j, 1]), 2, self.colors_hp[j], -1)
            cv2.putText(self.imgs[img_id], f"{j+1}", (points[j, 0], points[j, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw edges
        axis = [2, 1, 6]
        for j, e in enumerate(self.edges):
            
            temp = [e[0] - 1, e[1] - 1]
            if pred_flag == 'pred':
                edge_color = (255, 0, 0)  # bgr
                label = 'pred'
            elif pred_flag == 'gt':
                edge_color = (0, 255, 0)
                label = 'gt'
            elif pred_flag == 'pnp':
                edge_color = (0, 0, 255)  # bgr
                label = 'pnp'
            elif pred_flag == 'extra':
                edge_color = (0, 165, 255)
                label = 'extra'

            if j == axis[0]: # y
                edge_color = (128, 0, 128) # 보라색
            elif j == axis[1]: # x
                edge_color = (0, 255, 0) # 초록색
            elif j == axis[2]: # z
                edge_color = (0, 128, 255) # 주황색

            if points[temp[1], 0] <= -10000 or points[temp[1], 1] <= -10000 or points[temp[0], 0] <= -10000 or \
                    points[temp[0], 1] <= -10000:
                continue
            else:
                start_point = (int(points[temp[0], 0]), int(points[temp[0], 1]))
                end_point = (int(points[temp[1], 0]), int(points[temp[1], 1]))
                cv2.line(self.imgs[img_id], start_point, end_point, edge_color, 2)
                
                
        if pred_flag == 'pred':
            edge_color = (0, 0, 255)
        elif pred_flag == 'gt':
            edge_color = (255, 255, 255)
        elif pred_flag == 'pnp':
            edge_color = (0, 0, 0)


class MyDetector(ObjectPoseDetector):
    def __init__(self, opt):
        super(MyDetector, self).__init__(opt)

    def run(self, image_or_path_or_tensor, filename=None, meta_inp={}, preprocessed_flag=False):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, track_time, pnp_time, tot_time = 0, 0, 0, 0
        debugger = MyDebugger(dataset=self.opt.dataset, ipynb=(self.opt.debug == 3),
                            theme=self.opt.debugger_theme)
        start_time = time.time()

        # pre_processed = False
        pre_processed = preprocessed_flag

        # File input
        if isinstance(image_or_path_or_tensor, np.ndarray):
            # For eval or for CenterPose as data generator
            if len(image_or_path_or_tensor.shape) < 4:
                image = np.expand_dims(image_or_path_or_tensor, axis=0)
            else:
                image = image_or_path_or_tensor

            # We usually use image_or_path_or_tensor to represent filename
            if filename is not None:
                image_or_path_or_tensor = filename

        # String input
        elif type(image_or_path_or_tensor) == type(''):
            # For demo
            image = cv2.imread(image_or_path_or_tensor)
            image = np.expand_dims(np.array(image), axis=0)
        else:
            # Not used yet
            try:
                image = image_or_path_or_tensor['image'][0].numpy()
                pre_processed_images = image_or_path_or_tensor
                pre_processed = True
            except:
                return None
            
        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []
        
        # self.scales : [1.0]
        for scale in self.scales:
            scale_start_time = time.time()
            if not pre_processed:
                images, meta = self.pre_process(image, scale, meta_inp)
            else:

                # Used for data generation
                # 1 * 3 * 512 * 512
                images = np.expand_dims(image, axis=0)
                # images = image.reshape(1, 3, meta_inp['inp_height'], meta_inp['inp_width'])
                images = torch.from_numpy(images)
                meta = meta_inp

            images = images.to(self.opt.device)
            # initializing tracker
            pre_hms, pre_hm_hp, pre_inds = None, None, None

            torch.cuda.synchronize()
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time

            # run the network
            # output: the output feature maps, only used for visualizing
            # dets: output tensors after extracting peaks
            output, dets, forward_time = self.process(
                images, self.pre_images, pre_hms, pre_hm_hp, pre_inds, return_time=True)
            
            dets['kps'] *= self.opt.input_res / self.opt.output_res # *= 512 / 128
            # for k in dets:
            #     dets[k] = dets[k].detach().cpu().numpy()
            
            # debugger = Debugger(
            #     dataset=self.opt.dataset, ipynb=(self.opt.debug == 3), theme=self.opt.debugger_theme)
            
            std_tensor = torch.tensor(self.opt.std, device=images.device).view(1, 3, 1, 1)
            mean_tensor = torch.tensor(self.opt.mean, device=images.device).view(1, 3, 1, 1)

            # PyTorch 텐서로 계산 수행
            img = ((images * std_tensor + mean_tensor) * 255).clamp(0, 255).byte()

            # CPU로 이동 후 NumPy 배열로 변환
            img = img.cpu().numpy().transpose(0, 2, 3, 1)
            # print(f'dets: {dets}')

            for i in range(img.shape[0]):
                pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
                # gt = debugger.gen_colormap(batch['hm'][i][choice_list[i]].detach().cpu().numpy())
                debugger.add_blend_img(img[i], pred, f'out_hm_pred_{i}')
                # debugger.add_blend_img(img, gt, 'out_hm_gt')

                # Predictions
                debugger.add_img(img[i], img_id=f'out_img_pred_{i}')
                # print(f"dets['scores'][i]: {len(dets['scores'][i])}")
                for k in range(len(dets['scores'][i])):
                    # print(f"score: {dets['scores'][i][k][0]}/{self.opt.center_thresh}")
                    if dets['scores'][i][k][0] > self.opt.center_thresh:
                        debugger.add_coco_hp(dets['kps'][i][k], img_id=f'out_img_pred_{i}')

                # print(f'output["hm_hp"]: {output["hm_hp"].shape}') # [1 * 8 * 128 * 128]
                pred = debugger.gen_colormap_hp(output['hm_hp'][i].detach().cpu().numpy())
                debugger.add_blend_img(img[i], pred, f'out_hmhp_pred_{i}')

                pred = debugger.gen_colormap_hp(output['hps'][i].detach().cpu().numpy())
                debugger.add_blend_img(img[i], pred, f'out_hp_pred_{i}')

                pred = debugger.gen_colormap_hp(output['hp_offset'][i].detach().cpu().numpy())
                debugger.add_blend_img(img[i], pred, f'out_hp_offset_pred_{i}')
    
        return dets, debugger.imgs
