# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2

from lib.opts import opts
from lib.detectors.detector_factory import detector_factory
from lib.detectors.my_detector import MyDetector

import glob
import numpy as np
from tqdm import tqdm
import torch
import subprocess

def get_available_memory():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
    else:
        return 0

def set_device(required_memory):
    if torch.cuda.is_available():
        available_memory = get_available_memory()
        if available_memory >= required_memory:
            print(f"GPU 메모리 충분: {available_memory / 1e9:.2f} GB 사용 가능")
            return torch.device("cuda")
        else:
            print(f"GPU 메모리 부족: {available_memory / 1e9:.2f} GB 사용 가능, {required_memory / 1e9:.2f} GB 필요")
    
    print("CPU를 사용합니다.")
    return torch.device("cpu")

# 예시: 4GB의 메모리가 필요하다고 가정
# required_memory = 4 * 1024 * 1024 * 1024  # 4GB in bytes
required_memory = 0  # 4GB in bytes

device = set_device(required_memory)

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'pnp', 'track']


def demo(opt, meta):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    # Detector = detector_factory[opt.task]
    print("opt.arch: ", opt.arch)
    detector = MyDetector(opt)
    print("model loaded")
    if opt.use_pnp == True and 'camera_matrix' not in meta.keys():
        raise RuntimeError('Error found. Please give the camera matrix when using pnp algorithm!')

    if opt.demo == 'webcam' or \
            opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        
        cam = cv2.VideoCapture(1 if opt.demo == 'webcam' else opt.demo)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cam.get(cv2.CAP_PROP_FPS)
        print("start")
        if opt.demo == 'webcam':
            print(f"input 크기: {width}x{height}, fps: {fps}")
            
            frame_idx = 0
            while cam.isOpened():
                ret, frame = cam.read()
                if not ret:
                    break
                _, image_dict = detector.run(frame, meta_inp=meta)
                out_img = image_dict['out_img_pred']
                
                # 창 크기 조절
                window_name = 'Webcam demo'
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 크기 조절 가능한 창 생성
                cv2.imshow(window_name, out_img)

                # press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame_idx += 1
        else:
            total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
            print(f"원본 비디오 크기: {width}x{height}, fps: {fps}, total_frames: {total_frames}")
            
            if not os.path.exists(opt.output_dir):
                os.makedirs(opt.output_dir)

            detector.pause = False

            # Check if camera opened successfully
            if (cam.isOpened() == False):
                print("Error opening video stream or file")

            idx = 0
            pbar = tqdm(total=total_frames)
            while(cam.isOpened()):        
                batch = []        
                while(True):                
                    ret, img = cam.read()
                    if not ret : break
                    
                    batch.append(img)
                    # if len(batch) == opt.batch_size:
                    if len(batch) == 32:
                        break
                    
                batch = np.array(batch)
                _, image_dict = detector.run(batch, meta_inp=meta)
                
                if image_dict is not None:
                    for i in range(batch.shape[0]):
                        out_img = image_dict[f'out_img_pred_{i}']
                        # print(out_img.shape)
                        # 필요한 경우 크기 조정
                        # if out_img.shape[:2] != (height, width):
                        #     out_img = cv2.resize(out_img, (width, height))
                        
                        # uint8로 변환
                        out_img = np.clip(out_img, 0, 255).astype(np.uint8)
                    
                        # 비디오로 저장
                        # success = output_video.write(out_img)
                        # if not success:
                        #     print(f"프레임 {idx} 저장 실패")
                        
                        # exit()

                        # 이미지로 저장
                        filename = f"frame_{idx*32+i:04d}.png"
                        output_path = os.path.join(opt.output_dir, filename)
                        cv2.imwrite(output_path, out_img)
                        # print(f"프레임 {idx} 저장됨: {output_path}")
                else:
                    print(f"image_dict is None for frame {idx}")
                
                idx += 1
                pbar.update(batch.shape[0])
            print(f"처리 완료: {opt.output_dir}에 {idx}개의 프레임이 저장되었습니다.")
        
        cam.release()
        # output_video.release()
        # print(f"비디오 출력 완료: {output_filename}")
        # print(f"총 {idx}개의 프레임이 처리되었습니다.")

        # ffmpeg_cmd = f"ffmpeg -framerate {fps} -i {opt.output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_filename}"
        # subprocess.run(ffmpeg_cmd, shell=True, check=True)
        # print(f"비디오 생성 완료: {output_filename}")
        # 프레임 이미지 삭제
        # for i in range(idx):
        #     frame_path = os.path.join(opt.output_dir, f"frame_{i:04d}.png")
        #     os.remove(frame_path)
    else:

        # # Option 1: only for XX_test with a lot of sub-folders
        # image_names = []
        # for ext in image_ext:
        #     file_name=glob.glob(os.path.join(opt.demo,f'**/*.{ext}'))
        #     if file_name is not []:
        #         image_names+=file_name

        # Option 2: if we have images just under a folder, uncomment this instead
        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]

        detector.pause = False
        
        print("현재 작업 디렉토리:", os.getcwd())
        
        for idx, image_name in enumerate(image_names):
            print(f'Processing image {image_name}')

            # Todo: External GT input is not enabled in demo yet
            _, img_dict = detector.run(image_name, meta_inp=meta)
            # print(f"img_dict: {img_dict.keys()}")
            # print(f"type: {type(img_dict['out_img_pred'])}")
            # print(f"shape: {img_dict['out_img_pred'].shape}")
            # print(f"data type: {img_dict['out_img_pred'].dtype}")
            # exit(1)        
            for key, value in img_dict.items():
                # print(f"{key}: {value.shape}")
                # value = cv2.resize(value, (1920, 1080))
                cv2.imwrite(f'{opt.output_dir}/{image_name.split("/")[-1].split(".")[0]}_{key}.png', value)
            # debugger(ret)

        
            # time_str = ''
            # for stat in time_stats:
            #     time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            # print(f'Frame {idx}|' + time_str)

            


if __name__ == '__main__':

    # Default params with commandline input
    parser = opts().parser
    parser.add_argument('--output_dir', default="./demo_output/new/", help='출력 디렉토리')
    opt = parser.parse_args()
    opt.output_dir = f"./demo_output/{opt.output_dir}/"
    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
        print(f"출력 디렉토리 생성됨: {opt.output_dir}")

    # Local machine configuration example for CenterPose
    # opt.c = 'cup' # Only meaningful when enables show_axes option
    # opt.demo = "../images/CenterPose/cup/00007.png"
    # opt.arch = 'dlav1_34'
    # opt.load_model = f"../models/CenterPose/cup_mug_v1_140.pth"
    # opt.debug = 2
    # opt.show_axes = True

    # Local machine configuration example for CenterPoseTrack
    # opt.c = 'cup' # Only meaningful when enables show_axes option
    # opt.demo = '../images/CenterPoseTrack/shoe_batch-25_10.mp4'
    # opt.tracking_task = True
    # opt.arch = 'dla_34'
    # opt.load_model = f"../models/CenterPoseTrack/shoe_15.pth"
    # opt.debug = 2
    # opt.show_axes = True

    # Default setting
    opt.nms = True
    opt.obj_scale = True
    # opt.obj_scale = False

    # Tracking stuff
    if opt.tracking_task == True:
        print('Running tracking')
        opt.pre_img = True
        opt.pre_hm = True
        opt.tracking = True
        opt.pre_hm_hp = True
        opt.tracking_hp = True
        opt.track_thresh = 0.1

        opt.obj_scale_uncertainty = True
        opt.hps_uncertainty = True
        opt.kalman = True
        opt.scale_pool = True

        opt.vis_thresh = max(opt.track_thresh, opt.vis_thresh)
        opt.pre_thresh = max(opt.track_thresh, opt.pre_thresh)
        opt.new_thresh = max(opt.track_thresh, opt.new_thresh)

        # # For tracking moving objects, better to set up a small threshold
        # opt.max_age = 2

        print('Using tracking threshold for out threshold!', opt.track_thresh)

    # PnP related
    meta = {}
    if opt.cam_intrinsic is None:
        meta['camera_matrix'] = np.array(
            [[663.0287679036459, 0, 300.2775065104167], [0, 663.0287679036459, 395.00066121419275], [0, 0, 1]])
        opt.cam_intrinsic = meta['camera_matrix']
    else:
        meta['camera_matrix'] = np.array(opt.cam_intrinsic).reshape(3, 3)

    # Update default configurations
    opt = opts().parse(opt)

    # Update dataset info/training params
    opt = opts().init(opt)
    
    opt.c = 'custom_box'
    opt.arch = 'dlav1_34'

    demo(opt, meta)
