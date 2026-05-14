import os
import sys
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import default_collate
import torchvision
from torchvision.transforms import Compose
import numpy as np

import utils
import network
from vis import *
from dataset import FWIDataset
import transforms as T
import pytorch_ssim
from physics import AcousticWaveSolver2D, denormalize_velocity, normalize_data, denormalize_data


def compute_physics_consistency(model, solver, dataloader, device, ctx, k):
    model.eval()
    solver.eval()

    vmin_d = T.log_transform(ctx['data_min'], k=k)
    vmax_d = T.log_transform(ctx['data_max'], k=k)
    phys_mse_list = []
    phys_mae_list = []

    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            output = model(data)

            v_phys = denormalize_velocity(output, ctx['label_min'], ctx['label_max'])

            pred_data = solver(v_phys, use_checkpoint=False)
            pred_data_norm = normalize_data(pred_data, vmin_d, vmax_d, k)

            T = min(data.shape[2], pred_data_norm.shape[2])
            phys_mse = torch.mean((pred_data_norm[:, :, :T, :] - data[:, :, :T, :]) ** 2)
            phys_mae = torch.mean(torch.abs(pred_data_norm[:, :, :T, :] - data[:, :, :T, :]))

            phys_mse_list.append(phys_mse.item())
            phys_mae_list.append(phys_mae.item())

    return np.mean(phys_mse_list), np.mean(phys_mae_list)


def evaluate(model, solver, criterions, dataloader, device, k, ctx,
             vis_path, vis_batch, vis_sample):
    model.eval()

    label_list, label_pred_list = [], []
    label_tensor, label_pred_tensor = [], []

    with torch.no_grad():
        batch_idx = 0
        for data, label in dataloader:
            data = data.type(torch.FloatTensor).to(device, non_blocking=True)
            label = label.type(torch.FloatTensor).to(device, non_blocking=True)

            label_np = T.tonumpy_denormalize(
                label, ctx['label_min'], ctx['label_max'], exp=False
            )
            label_list.append(label_np)
            label_tensor.append(label)

            pred = model(data)

            label_pred_np = T.tonumpy_denormalize(
                pred, ctx['label_min'], ctx['label_max'], exp=False
            )
            label_pred_list.append(label_pred_np)
            label_pred_tensor.append(pred)

            if vis_path and batch_idx < vis_batch:
                for i in range(vis_sample):
                    plot_velocity(
                        label_pred_np[i, 0], label_np[i, 0],
                        f'{vis_path}/V_{batch_idx}_{i}.png'
                    )
            batch_idx += 1

    label = np.concatenate(label_list)
    label_pred = np.concatenate(label_pred_list)
    label_t = torch.cat(label_tensor)
    pred_t = torch.cat(label_pred_tensor)

    l1 = nn.L1Loss()
    l2 = nn.MSELoss()
    mae_val = l1(label_t, pred_t).item()
    mse_val = l2(label_t, pred_t).item()

    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    ssim_val = ssim_loss(label_t / 2 + 0.5, pred_t / 2 + 0.5).item()

    print(f'MAE: {mae_val:.6f}')
    print(f'MSE: {mse_val:.6f}')
    print(f'SSIM: {ssim_val:.6f}')

    for name, criterion_fn in criterions.items():
        print(f'  Velocity {name}: {criterion_fn(label, label_pred):.4f}')

    if solver is not None:
        phys_mse, phys_mae = compute_physics_consistency(
            model, solver, dataloader, device, ctx, k
        )
        print(f'Physics MSE: {phys_mse:.6f}')
        print(f'Physics MAE: {phys_mae:.6f}')
        return mae_val, mse_val, ssim_val, phys_mse, phys_mae

    return mae_val, mse_val, ssim_val, 0.0, 0.0


def main(args):
    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    utils.mkdir(args.output_path)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    with open('dataset_config.json') as f:
        try:
            ctx = json.load(f)[args.dataset]
        except KeyError:
            print('Unsupported dataset.')
            sys.exit()

    if args.file_size is not None:
        ctx['file_size'] = args.file_size

    print("Loading data")
    log_data_min = T.log_transform(ctx['data_min'], k=args.k)
    log_data_max = T.log_transform(ctx['data_max'], k=args.k)

    transform_valid_data = Compose([
        T.LogTransform(k=args.k),
        T.MinMaxNormalize(log_data_min, log_data_max),
    ])
    transform_valid_label = Compose([
        T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
    ])

    if args.val_anno[-3:] == 'txt':
        dataset_valid = FWIDataset(
            args.val_anno,
            sample_ratio=args.sample_temporal,
            file_size=ctx['file_size'],
            transform_data=transform_valid_data,
            transform_label=transform_valid_label
        )
    else:
        dataset_valid = torch.load(args.val_anno)

    print("Creating data loaders")
    valid_sampler = SequentialSampler(dataset_valid)
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=args.batch_size,
        sampler=valid_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=default_collate
    )

    print("Creating model")
    if args.model not in network.model_dict:
        print('Unsupported model.')
        sys.exit()

    model = network.model_dict[args.model](
        upsample_mode=args.up_mode,
        sample_spatial=args.sample_spatial,
        sample_temporal=args.sample_temporal,
        norm=args.norm
    ).to(device)

    if args.measure_physics:
        print("Creating physics solver for consistency evaluation")
        solver = AcousticWaveSolver2D(
            dx=ctx.get('dx', 10),
            dt=ctx.get('dt', 0.001),
            nz=ctx.get('n_grid', 70),
            nx=ctx.get('n_grid', 70),
            n_boundary=args.n_boundary,
            n_sources=ctx.get('ns', 5),
            n_receivers=ctx.get('ng', 70),
            source_depth=int(ctx.get('sz', 10) / ctx.get('dx', 10)),
            receiver_depth=int(ctx.get('gz', 10) / ctx.get('dx', 10)),
            source_freq=ctx.get('f', 15),
            save_every=1
        ).to(device)
    else:
        solver = None

    criterions = {
        'MAE': lambda x, y: np.mean(np.abs(x - y)),
        'MSE': lambda x, y: np.mean((x - y) ** 2)
    }

    if args.resume:
        print(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(network.replace_legacy(checkpoint['model']))
        print(f'Loaded model checkpoint at Epoch {checkpoint["epoch"]} / Step {checkpoint["step"]}.')

    if args.vis:
        vis_folder = f'visualization_{args.vis_suffix}' if args.vis_suffix else 'visualization'
        vis_path = os.path.join(args.output_path, vis_folder)
        utils.mkdir(vis_path)
    else:
        vis_path = None

    print("Start testing")
    start_time = time.time()
    evaluate(model, solver, criterions, dataloader_valid, device,
             args.k, ctx, vis_path, args.vis_batch, args.vis_sample)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PGINN Testing with Physics Metrics')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-ds', '--dataset', default='flatfault-b', type=str, help='dataset name')
    parser.add_argument('-fs', '--file-size', default=None, type=int, help='number of samples in each npy file')

    parser.add_argument('-ap', '--anno-path', default='split_files', help='annotation files location')
    parser.add_argument('-v', '--val-anno', default='flatfault_b_val_invnet.txt', help='name of val anno')
    parser.add_argument('-o', '--output-path', default='PGINN_models', help='path to parent folder')
    parser.add_argument('-n', '--save-name', default='pginn_ffb', help='folder name for this experiment')
    parser.add_argument('-s', '--suffix', type=str, default=None, help='subfolder name')

    parser.add_argument('-m', '--model', type=str, default='InversionNet', help='inverse model name')
    parser.add_argument('-no', '--norm', default='bn', help='normalization layer type')
    parser.add_argument('-um', '--up-mode', default=None, help='upsampling layer mode')
    parser.add_argument('-ss', '--sample-spatial', type=float, default=1.0, help='spatial sampling ratio')
    parser.add_argument('-st', '--sample-temporal', type=int, default=1, help='temporal sampling ratio')

    parser.add_argument('-b', '--batch-size', default=50, type=int)
    parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers')
    parser.add_argument('--k', default=1, type=float, help='k in log transformation')
    parser.add_argument('-r', '--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--vis', action="store_true", help='visualization option')
    parser.add_argument('-vsu', '--vis-suffix', default=None, type=str, help='visualization suffix')
    parser.add_argument('-vb', '--vis-batch', type=int, default=0, help='number of batch to visualize')
    parser.add_argument('-vsa', '--vis-sample', type=int, default=0, help='samples per batch to visualize')

    parser.add_argument('-mp', '--measure-physics', action='store_true',
                        help='compute physics consistency metrics')
    parser.add_argument('-nbnd', '--n-boundary', type=int, default=30,
                        help='number of boundary grid points')
    parser.add_argument('-seg', '--segment-size', type=int, default=50,
                        help='checkpoint segment size')

    args = parser.parse_args()

    args.output_path = os.path.join(args.output_path, args.save_name, args.suffix or '')
    args.val_anno = os.path.join(args.anno_path, args.val_anno)
    args.resume = os.path.join(args.output_path, args.resume)

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
