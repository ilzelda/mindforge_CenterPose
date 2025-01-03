# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.utils.data
from lib.opts import opts
from lib.models.model import create_model, load_model, save_model
from lib.models.data_parallel import DataParallel
from lib.logger import Logger
from lib.datasets.dataset_factory import collate_fn_filtered
from lib.trains.train_factory import train_factory
import time
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from lib.datasets.dataset_combined import ObjectPoseDataset


def main(opt):
    torch.cuda.empty_cache()
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    torch.cuda.set_per_process_memory_fraction(0.8)

    Dataset = ObjectPoseDataset


    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

    logger = Logger(opt)

    # opt.gpus_str = "0, 1"
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    
    # opt.gpus = [0, 1] or [-1]
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    # opt.device = torch.device('cuda:1')



    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
    
    scheduler = CosineAnnealingLR(
                                    optimizer,
                                    T_max=opt.num_epochs,  # 하나의 사이클이 끝나는 epoch 수 (일반적으로 전체 epoch)
                                    eta_min=1e-7           # LR가 도달할 최소값(필요 시 조절)
                                )
    
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print('Setting up data...')
    val_dataset = Dataset(opt, 'val')
    if opt.tracking_task == True:
        val_dataset_subset = torch.utils.data.Subset(val_dataset, range(0, len(val_dataset), 15))
    else:
        val_dataset_subset = val_dataset

    val_loader = torch.utils.data.DataLoader(
        val_dataset_subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn_filtered
    )

    if opt.test:
        _, preds, _ = trainer.val(0, val_loader)

    # print("train datset test")
    # example_train = Dataset(opt, 'train')
    # # print(example_train[0])
    # for i in range(len(example_train)):
    #     try:
    #         item = example_train[i]
    #         if item is None:
    #             print(f"Item {i} is None")
    #     except Exception as e:
    #         print(f"Error loading item {i}: {e}")

    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn_filtered
    )

    print('Starting training...')
    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _, log_imgs = trainer.train(epoch, train_loader)
        logger.write('epoch: {} | '.format(epoch))  # txt logging
        # else : logger.write(f"Epoch [{epoch+1}/{opt.num_epochs}] - LR: {scheduler.get_last_lr()[0]} | ")
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)  # tensorboard logging
            logger.write('train_{} {:8f} | '.format(k, v))  # txt logging
        logger.img_summary('train', log_imgs, epoch)
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, f'{opt.c}_{mark}.pth'),
                       epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds, log_imgs = trainer.val(epoch, val_loader)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('val_{} {:8f} | '.format(k, v))
            logger.img_summary('val', log_imgs, epoch)
            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, f'{opt.c}_best.pth'),
                           epoch, model)
            # else:
            save_model(os.path.join(opt.save_dir, f'{opt.c}_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n\n')

        # if opt.custom:
        #     scheduler.step()
        # else:
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, f'{opt.c}_{epoch}.pth'),
                    epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    # Default params with commandline input
    opt = opts()
    opt = opt.parser.parse_args()

    # Local configuration
    # opt.c = 'custom_box'
    # opt.c = 'parcel'
    # opt.c = 'case3_filtered'
    # opt.c = 'case4_filtered'
    # opt.c = 'case5_filtered'
    # opt.c = 'case6_filtered'
    # opt.c = 'case7_filtered'
    # opt.c = 'case8_filtered'
    opt.c = 'case9_filtered'

    opt.arch='dlav1_34'
    opt.obj_scale = True
    opt.obj_scale_weight = 1
    opt.mug = False

    # Training param
    opt.exp_id = f'objectron_{opt.c}_{opt.arch}'
    opt.num_epochs = 200
    opt.val_intervals = 5
    opt.lr_step = '90,120'
    opt.batch_size = 4
    opt.lr = 6e-5
    # opt.lr = 6e-7

    opt.gpus = '1'
    opt.num_workers = 4
    opt.print_iter = 5
    opt.debug = 5
    opt.save_all = True

    # # To continue
    # opt.resume = True
    # opt.load_model = ""

    # Copy from parse function from opts.py
    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')] # '0, 1' -> [0, 1]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1] # [1, 2] -> [0, 1] or [-1]
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
    opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

    opt.fix_res = not opt.keep_res
    print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
    opt.reg_offset = not opt.not_reg_offset
    opt.reg_bbox = not opt.not_reg_bbox
    opt.hm_hp = not opt.not_hm_hp
    opt.reg_hp_offset = (not opt.not_reg_hp_offset) and opt.hm_hp

    if opt.head_conv == -1:  # init default head_conv
        opt.head_conv = 256 if 'dla' in opt.arch else 64
    opt.pad = 127 if 'hourglass' in opt.arch else 31
    opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

    if opt.trainval:
        opt.val_intervals = 100000000

    if opt.master_batch_size == -1:
        opt.master_batch_size = opt.batch_size // len(opt.gpus)
    rest_batch_size = (opt.batch_size - opt.master_batch_size)
    opt.chunk_sizes = [opt.master_batch_size]
    for i in range(len(opt.gpus) - 1):
        slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
        if i < rest_batch_size % (len(opt.gpus) - 1):
            slave_chunk_size += 1
        opt.chunk_sizes.append(slave_chunk_size)
    print('training chunk_sizes:', opt.chunk_sizes)

    opt.root_dir = os.path.join(os.path.dirname(__file__), '..') # CenterPose/
    print("opt.root_dir: ", opt.root_dir)
    opt.data_dir = os.path.join(opt.root_dir, 'data') # CenterPose/data
    print("opt.data_dir: ", opt.data_dir)
    opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task) # CenterPose/exp/opt.task

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    # opt.save_dir = os.path.join(opt.exp_dir, f'{opt.exp_id}_{time_str}')
    opt.save_dir = os.path.join('/mnt/CenterPose_exp/object_pose', f'{opt.exp_id}_{time_str}')

    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    print('The output will be saved to ', opt.save_dir)

    # custom options
    opt.custom = True
    opt.num_workers = 0
    opt.batch_size = 16
    # opt.obj_scale_weight = 0
    # opt.obj_scale = False
    
    if opt.c == 'custom_box':
        opt.data_dir = '/home/CenterPose/data/custom_box'
    elif opt.c == 'parcel':
        opt.data_dir = '/home/CenterPose/data/parcel'
    elif 'case' in opt.c:
        opt.data_dir = f'/home/CenterPose/data/{opt.c}'

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    main(opt)
