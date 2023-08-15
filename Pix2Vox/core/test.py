# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import json
import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data
import pandas as pd
import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

import torchvision
from tensorboardX import SummaryWriter

from datetime import datetime as dt

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger

import matplotlib.pyplot as plt


def test_net(cfg,
             epoch_idx=-1,
             output_dir=None,
             test_data_loader=None,
             test_writer=None,
             encoder=None,
             decoder=None,
             refiner=None,
             merger=None,
             samples_to_save=None,
             filename=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}


    log_dir = output_dir % 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    val_writer = SummaryWriter(os.path.join(log_dir, 'test'))

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
                                                       batch_size=1,
                                                       num_workers=1,
                                                       pin_memory=True,
                                                       shuffle=False)


    # Set up networks
    if decoder is None or encoder is None:
        encoder = Encoder(cfg)
        decoder = Decoder(cfg)
        refiner = Refiner(cfg)
        merger = Merger(cfg)

        if torch.cuda.is_available():
            encoder = torch.nn.DataParallel(encoder).cuda()
            decoder = torch.nn.DataParallel(decoder).cuda()
            refiner = torch.nn.DataParallel(refiner).cuda()
            merger = torch.nn.DataParallel(merger).cuda()

        print(cfg.CONST.WEIGHTS)

        print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS, map_location='cpu')
        epoch_idx = checkpoint['epoch_idx']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

        if cfg.NETWORK.USE_REFINER:
            refiner.load_state_dict(checkpoint['refiner_state_dict'])
        if cfg.NETWORK.USE_MERGER:
            merger.load_state_dict(checkpoint['merger_state_dict'])

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()
    encoder_losses = utils.network_utils.AverageMeter()
    refiner_losses = utils.network_utils.AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()

    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]

        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            ground_truth_volume = utils.network_utils.var_or_cuda(ground_truth_volume)

            # Test the encoder, decoder, refiner and merger
            image_features = encoder(rendering_images)
            raw_features, generated_volume = decoder(image_features)

            # ADDED
            grid = torchvision.utils.make_grid(rendering_images.squeeze())
            val_writer.add_image(f'images_{sample_name}', grid, 0)

            if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
                generated_volume = merger(raw_features, generated_volume)
            else:
                generated_volume = torch.mean(generated_volume, dim=1)
            encoder_loss = bce_loss(generated_volume, ground_truth_volume) * 10

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volume = refiner(generated_volume)
                refiner_loss = bce_loss(generated_volume, ground_truth_volume) * 10
            else:
                refiner_loss = encoder_loss

            # Append loss and accuracy to average metrics
            encoder_losses.update(encoder_loss.item())
            refiner_losses.update(refiner_loss.item())

            # IoU per sample
            sample_iou = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                sample_iou.append((intersection / union).item())

            # IoU per taxonomy
            if taxonomy_id not in test_iou:
                test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
            test_iou[taxonomy_id]['n_samples'] += 1
            test_iou[taxonomy_id]['iou'].append(sample_iou)

            # Append generated volumes to TensorBoard
            if output_dir and samples_to_save and sample_name in samples_to_save:
                
                img_dir = output_dir % 'images'

                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)

                # Volume Visualization
                gv = generated_volume.cpu().numpy()
                voxel_img = utils.binvox_visualization.get_volume_views(gv, title=f"Sample {sample_name} IoU {sample_iou[0]:.3f}")
                img_file = os.path.join(img_dir, filename % (sample_name, sample_idx))
                voxel_img.savefig(img_file, bbox_inches='tight')

                img_file = os.path.join(img_dir, f"GT_{sample_name}_{sample_idx}_voxel.png")

                if not os.path.exists(img_file):
                    gtv = ground_truth_volume.cpu().numpy()
                    voxel_gt_img = utils.binvox_visualization.get_volume_views(gtv, title=f"")
                    
                    voxel_gt_img.savefig(img_file, bbox_inches='tight')

                plt.close('all')

            # Print sample loss and IoU
            print('[INFO] %s Test[%d/%d] Taxonomy = %s Sample = %s EDLoss = %.4f RLoss = %.4f IoU = %s' %
                  (dt.now(), sample_idx + 1, n_samples, taxonomy_id, sample_name, encoder_loss.item(),
                   refiner_loss.item(), ['%.4f' % si for si in sample_iou]))

    # Output testing results
    summary = pd.DataFrame({'IoU': np.array(test_iou[taxonomy_id]['iou'])[:,0]}).describe()
    print(summary)

    mean_iou = []
    for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])

    mean_iou = np.sum(mean_iou, axis=0) / n_samples

    # Print header
    print('\n============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    print('Baseline', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    # Print body
    for taxonomy_id in test_iou:
        print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
        print('%d' % test_iou[taxonomy_id]['n_samples'], end='\t')
        if 'baseline' in taxonomies[taxonomy_id]:
            print('%.4f' % taxonomies[taxonomy_id]['baseline']['%d-view' % cfg.CONST.N_VIEWS_RENDERING], end='\t\t')
        else:
            print('N/a', end='\t\t')

        for ti in test_iou[taxonomy_id]['iou']:
            print('%.4f' % ti, end='\t')
        print()

    # Print mean IoU for each threshold
    print(f'\nOverall {mean_iou.tolist()}\n')
    # print('%.4f' % , end='\t')

    # Add testing results to TensorBoard
    max_iou = np.max(mean_iou)

    pandas_rows = []
    pandas_rows.append(('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx))
    pandas_rows.append(('Refiner/EpochLoss', refiner_losses.avg, epoch_idx))
    pandas_rows.append(('Refiner/IoU', max_iou, epoch_idx)) 
    
    if test_writer is not None:
        test_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx)
        test_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx)
        test_writer.add_scalar('Refiner/IoU', max_iou, epoch_idx)

    return mean_iou, pandas_rows
