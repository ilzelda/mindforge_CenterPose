# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import threading
import importlib
import numpy as np
from tqdm import tqdm
import json
import math
import shutil

from lib.opts import opts
from lib.detectors.detector_factory import detector_factory
from lib.detectors.my_detector import MyDetector

from aruco_marker import CharucoDetector
from util_dim import resize_into_original_image, calc_box_size_pixel

import lib.detectors.my_detector as my_detector_module
import aruco_marker as aruco_marker_module
import util_dim as util_dim_module


def round_custom(num):
    # 정수 부분과 일의 자리 분리
    integer_part = int(num // 10) * 10  # 일의 자리 제거
    remainder = num % 10  # 일의 자리 값 추출

    # 조건에 따라 처리
    if remainder >= 0 and remainder < 3:
        return integer_part  # 그대로 버림
    elif remainder >= 3 and remainder < 8:
        return integer_part + 5  # 5로 맞춤
    elif remainder >= 8:
        return integer_part + 10  # 올림

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'pnp', 'track']


def demo(opt, meta):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    detector = MyDetector(opt)
    print("model loaded : ", "GPU" if opt.gpus[0] >= 0 else "CPU")
    if opt.use_pnp == True and 'camera_matrix' not in meta.keys():
        raise RuntimeError('Error found. Please give the camera matrix when using pnp algorithm!')

    if opt.demo == 'webcam' or \
            opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        
        if opt.demo == 'webcam' :
            print("opening webcam...")
            cam = cv2.VideoCapture(1)
        else:
            cam = cv2.VideoCapture(opt.demo)
            pbar = tqdm(total=int(cam.get(cv2.CAP_PROP_FRAME_COUNT)))

        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cam.get(cv2.CAP_PROP_FPS)
        print(f"input 크기: {width}x{height}, fps: {fps}")
        print("start")
        
        test_mode = False
        save_mode = False
        save_count = 0

        test_count = 0
        marker_detector = CharucoDetector(debug=True)

        frame_idx = 0
        x_sum = 0
        z_sum = 0
        y_sum = 0
        sum_count = 0
        while cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                # cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # continue
                break

            if save_mode:
                out.write(frame)

            # 스레드 생성
            detector_thread = threading.Thread(target=lambda: (
                setattr(threading.current_thread(), 'result', detector.run(frame, meta_inp=meta))
            ))
            marker_thread = threading.Thread(target=lambda: (
                # setattr(threading.current_thread(), 'result', marker_detector.detect_charuco(frame))
                setattr(threading.current_thread(), 'result', marker_detector.detect_aruco(frame))
            ))

            # 스레드 시작
            detector_thread.start()
            marker_thread.start()

            # 스레드 종료 대기
            detector_thread.join()
            marker_thread.join()

            # 결과 가져오기
            dets, image_dict = detector_thread.result
            dets['kps'] = dets['kps'].reshape(100, -1)[0].reshape(8, 2)
            dets['kps'] = resize_into_original_image(dets['kps'])
            # for i in range(len(dets['kps'])):
            #     cv2.circle(frame, (int(dets['kps'][i][0]), int(dets['kps'][i][1])), 1, (0, 0, 255), 2)
            # cv2.imshow("original_image", frame)

            dets['obj_scale'] = dets['obj_scale'].reshape(100, -1)[0]
            out_img = image_dict['out_img_pred_0']
            
            marker_detected = marker_thread.result is not None
            box_detected = (np.array(dets['scores'])[0,:,0] > opt.center_thresh).any()
            # cv2.putText(out_img, f"marker detected:{marker_detected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255) if marker_detected else (0, 0, 255), 1)
            # cv2.putText(out_img , f"box detected:{box_detected}", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255) if box_detected else (0, 0, 255), 1)
            # cv2.putText(out_img, f"test_mode:{test_mode}", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255) if test_mode else (0, 0, 255), 1)

            transform_matrix = None
            if marker_detected :
                marker_corners, marker_ids = marker_thread.result
                try:    
                    transform_matrix = marker_detector.find_homography(marker_corners, marker_detector.dst_points_aruco[marker_ids])
                    # transform_matrix = 
                except Exception as e:
                    print(f"homography error: {e}")
                # if transform_matrix is not None:
                #     warped_frame = cv2.warpPerspective(frame.copy(), transform_matrix, (2048, 3058))
                #     for i in range(len(marker_ids)):
                #         for j in range(4):  
                #             cv2.circle(warped_frame, (int(marker_detector.dst_points_aruco[marker_ids[i]][0][j][0]), \
                #                                         int(marker_detector.dst_points_aruco[marker_ids[i]][0][j][1])), \
                #                         1, (0, 0, 255), 2)
                #             if j == 0:
                #                 cv2.putText(warped_frame, f"{marker_ids[i]}", (int(marker_detector.dst_points_aruco[marker_ids[i]][0][j][0]), \
                #                                                                 int(marker_detector.dst_points_aruco[marker_ids[i]][0][j][1])), \
                #                                                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                #     cv2.putText(warped_frame, f"detected marker: {len(marker_ids)}", (10, 3058//4 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    # cv2.imshow("warped_frame", warped_frame)
            if box_detected and transform_matrix is not None:
                kps_reshaped = dets['kps'].reshape(-1, 1, 2)
                kps_BEV = cv2.perspectiveTransform(kps_reshaped, transform_matrix)
                kps_BEV = kps_BEV.reshape(-1, 2)
                # for i in range(len(kps_BEV)):
                #     cv2.circle(warped_frame, (int(kps_BEV[i][0]), int(kps_BEV[i][1])), 1, (0, 255, 255), 2)
                    # cv2.imshow("warped_frame", warped_frame)

                box_size_pixel = calc_box_size_pixel(dets['kps'])
                warped_box_size_pixel = calc_box_size_pixel(kps_BEV)
               
                # x_sum += warped_box_size_pixel['x']
                # z_sum += warped_box_size_pixel['z']
                # sum_count += 1
                # x_length = round_custom(x_sum / sum_count / marker_detector.MARKER_LENGTH * 3)
                # z_length = round_custom(z_sum / sum_count / marker_detector.MARKER_LENGTH * 3)

                x_length = warped_box_size_pixel['x'] / marker_detector.MARKER_LENGTH * 3
                z_length = warped_box_size_pixel['z'] / marker_detector.MARKER_LENGTH * 3
                
                y_length = x_length / dets['obj_scale'][0]
                
                # y_sum += x_length / dets['obj_scale'][0]
                # try:
                #     y_length = round_custom(y_sum / sum_count)
                # except Exception as e:
                #     print(f"y_length error: {e}, y_sum: {y_sum}, sum_count: {sum_count}")
                
                    
                # 계산된 dimension 표시
                # ori = 3
                colors = [(0, 255, 0), (128, 0, 128), (0, 128, 255)]
                for i, (key, value) in enumerate([('X', x_length), ('Y', y_length), ('Z', z_length)]):
                    text = f"{key}: {value:.2f}cm"
                    # 텍스트 크기 측정
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    # 텍스트 배경 그리기
                    cv2.rectangle(out_img, (10, 50 + 20 * i), (10 + text_width, 65 + 20 * i), (255, 255, 255), -1)
                    # 텍스트 그리기
                    cv2.putText(out_img, text, (10, 60 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)
                
                print(f"obj_scale      - X: {dets['obj_scale'][0]:05.2f},   Y: {dets['obj_scale'][1]:05.2f},   Z: {dets['obj_scale'][2]:05.2f}",
                        end='\n', flush=False)
                print(f"box pixel size - X: {box_size_pixel['x']:05.2f}px, Y: {box_size_pixel['y']:05.2f}px, Z: {box_size_pixel['z']:05.2f}px",
                    end='\n', flush=False) 
                print(f"warped size    - X: {warped_box_size_pixel['x']:05.2f}px, Y: {warped_box_size_pixel['y']:05.2f}px, Z: {warped_box_size_pixel['z']:05.2f}px",
                        end='\n\033[F\033[F\033[F', flush=True)
                
            if opt.demo == 'webcam':
                # 창 크기 조절
                window_name = 'out_img'
                cv2.putText(out_img, f"Frame: {frame_idx}", (out_img.shape[1] - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 크기 조절 가능한 창 생성
                cv2.imshow(window_name, out_img)
                
                key = cv2.waitKey(1)
                # press 'q' to quit
                if key & 0xFF == ord('q'):
                    break

                if key & 0xFF == ord('r'):
                    importlib.reload(util_dim_module)
                    importlib.reload(my_detector_module)
                    detector = my_detector_module.MyDetector(opt)
                    importlib.reload(aruco_marker_module)
                    marker_detector = CharucoDetector(debug=True)
                    print("reload")
                
                if key & 0xFF == ord('s'):
                    if not save_mode:
                        save_mode = True
                        print("save mode on")
                        save_count += 1
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(f'output_{save_count}.mp4', fourcc, fps, (int(cam.get(3)), int(cam.get(4))))

                    else:
                        save_mode = False
                        print("save mode off")

                if key & 0xFF == ord('t'):
                    if test_mode:
                        test_mode = False
                        test_count = 0
                    else:
                        data = {
                            'obj_scale': {
                                'X': [],
                                'Y': [], 
                                'Z': []
                            },
                            'length': {
                                'X': [],
                                'Y': [],
                                'Z': []
                            }
                        }
                        test_mode = True

                if test_mode and frame_idx % (int(fps)//2) == 0:
                    if test_count < 10:
                        test_count += 1
                        data['obj_scale']["X"].append(float(dets['obj_scale'][0]))
                        data['obj_scale']["Y"].append(float(dets['obj_scale'][1]))
                        data['obj_scale']["Z"].append(float(dets['obj_scale'][2]))
                        data['length']["X"].append(float(x_length))
                        data['length']["Y"].append(float(y_length))
                        data['length']["Z"].append(float(z_length))
                        print(f"log saved #{test_count}")
                    else: 
                            # 객체 크기 정보를 파일에 기록
                        with open(f'dimension_log.json', 'w') as f:
                            json.dump(data, f, indent=2)
                            print(f"log saved into dimension_log.json")
                        test_mode = False
                        test_count = 0
            else :
                # pbar.write(f"obj_scale      - X: {dets['obj_scale'][0]:05.2f},   Y: {dets['obj_scale'][1]:05.2f},   Z: {dets['obj_scale'][2]:05.2f}")
                # pbar.write(f"box pixel size - X: {box_size_pixel['x']:05.2f}px, Y: {box_size_pixel['y']:05.2f}px, Z: {box_size_pixel['z']:05.2f}px")
                # pbar.write(f"warped size    - X: {warped_box_size_pixel['x']:05.2f}px, Y: {warped_box_size_pixel['y']:05.2f}px, Z: {warped_box_size_pixel['z']:05.2f}px")
                pbar.set_postfix({
                    "obj_scale_X": f"{dets['obj_scale'][0]:05.2f}",
                    "obj_scale_Y": f"{dets['obj_scale'][1]:05.2f}",
                    "obj_scale_Z": f"{dets['obj_scale'][2]:05.2f}",
                    "box_size_X": f"{box_size_pixel['x']:05.2f}px",
                    "box_size_Y": f"{box_size_pixel['y']:05.2f}px",
                    "box_size_Z": f"{box_size_pixel['z']:05.2f}px",
                    "warped_X": f"{warped_box_size_pixel['x']:05.2f}px",
                    "warped_Y": f"{warped_box_size_pixel['y']:05.2f}px",
                    "warped_Z": f"{warped_box_size_pixel['z']:05.2f}px"
                })

                # 이미지로 저장
                # 프레임 번호 표시
                cv2.putText(out_img, f"Frame: {frame_idx}", (out_img.shape[1] - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
                filename = f"frame_{frame_idx:04d}.png"
                output_path = os.path.join(opt.output_dir, filename)
                cv2.imwrite(output_path, out_img)
                pbar.update(1)
                if opt.labelling \
                    and ((frame_idx % (cam.get(cv2.CAP_PROP_FRAME_COUNT) // 20) == 0) \
                            # or (frame_idx % (cam.get(cv2.CAP_PROP_FRAME_COUNT) // 30) == 20)
                            ):
                    cv2.imwrite(f"{opt.output_dir}/_output/frame_{frame_idx:04d}.jpg", frame)
                    with open(f"{opt.output_dir}/_output/frame_{frame_idx:04d}.json", 'w') as f:
                        kps = {}
                        for i, n in enumerate([1,5,7,3,2,6,8,4]):
                            kps[f"x{i+1}"] = float(dets['kps'][n-1][0])
                            kps[f"y{i+1}"] = float(dets['kps'][n-1][1])
                            kps[f"z{i+1}"] = 0 if i < 4 else 1
                        info = {"metaData":"","inspRejectYn":"N","labelingInfo":[{"3DBox":{"location":[kps],"label":"box","type":"3DBox"}}]}
                        json.dump(info, f, indent=2)

            frame_idx += 1
        
                        
        if opt.demo != 'webcam':
            print(f"처리 완료: {opt.output_dir}에 {frame_idx}개의 프레임이 저장되었습니다.")
        
        cam.release()
        # output_video.release()
        # print(f"비디오 출력 완료: {output_filename}")
        # print(f"총 {idx}개의 프레임이 처리되었습니다.")

        # ffmpeg_cmd = f"ffmpeg -framerate {fps} -i {opt.output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_filename}"
        # subprocess.run(ffmpeg_cmd, shell=True, check=True)
        # print(f"비디오 생성 완료: {output_filename}")
        # 프레임 이미�� 삭제
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
    parser.add_argument('--labelling', action='store_true', help='labeling mode', default=False)
    opt = parser.parse_args()
    
    if opt.demo != 'webcam':
        opt.output_dir = f"./demo_output/{opt.load_model.split('/')[-2]}/{opt.load_model.split('/')[-1].split('.')[0]}/{opt.demo.split('/')[-1].split('.')[0]}"
        # 출력 디렉토리가 없으면 생성
        if not os.path.exists(opt.output_dir):
            os.makedirs(opt.output_dir)
            print(f"출력 디렉토리 생성됨: {opt.output_dir}")
        else:
            print(f"출력 디렉토리 : {opt.output_dir}")

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
    
    opt.gpus = [0]
    opt.c = 'custom_box'
    opt.arch = 'dlav1_34'
    opt.use_pnp = False
    print('rep_mode : ',opt.rep_mode)
    print("labelling mode: ", opt.labelling)  
    if opt.labelling:  
        if not os.path.exists(os.path.join(opt.output_dir, "_output")):
            os.makedirs(os.path.join(opt.output_dir, "_output"))
            print("_output 디렉토리 생성됨")

        else :
            delete_count = 0
            for filename in os.listdir(os.path.join(opt.output_dir, "_output")):
                file_path = os.path.join(opt.output_dir, "_output", filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        delete_count += 1  
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
            print(f"이전 레이블링 정보 삭제됨({delete_count})")

    demo(opt, meta)
