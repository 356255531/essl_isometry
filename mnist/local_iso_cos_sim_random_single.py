import os
import math
import time
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from utils import fix_seed


class MnistRotDataset(Dataset):
    def __init__(self, file, transform=None):
        self.transform = transform

        data = np.loadtxt(file, delimiter=' ')

        self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)


class GEncoder(nn.Module):
    def __init__(self):
        super(GEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 maxpool
        self.fc1 = nn.Linear(5 * 5 * 10, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 24x24x10
        x = self.pool(x)  # 12x12x10
        x = F.relu(self.conv2(x))  # 8x8x10
        x = self.pool(x)  # 4x4x10
        x = x.view(-1, 5 * 5 * 10)  # flattening
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class ContrastiveLearningTransform:
    def __init__(self, num_base_rotation, base_rot_range, num_nn, local_rot_range):
        self.num_base_rotation = num_base_rotation
        self.base_rot_range = base_rot_range
        self.num_nn = num_nn
        self.local_rot_range = local_rot_range

    def __call__(self, x):
        x = T.Pad(padding=(0, 0, 1, 1), fill=0)(x)
        x = T.Resize(87)(x)
        base_rot_angles = [random.random() * self.base_rot_range for _ in range(self.num_base_rotation)]
        rotate_angles = [
            [base_rot_angle + random.random() * self.local_rot_range for _ in range(self.num_nn)]
            for base_rot_angle in base_rot_angles
        ]
        rotate_angles = sum(rotate_angles, [])
        xs = [
            TF.rotate(x, angle, interpolation=TF.InterpolationMode.BILINEAR)
            for angle in rotate_angles
        ]
        xs = [T.Resize(32)(x) for x in xs]
        xs = [T.ToTensor()(x) for x in xs]
        cc, hh, ww = xs[0].shape
        xs = torch.stack(xs, dim=0).reshape(self.num_base_rotation, self.num_nn, cc, hh, ww)

        rotate_angles = torch.tensor(rotate_angles).reshape(self.num_base_rotation, self.num_nn)
        return xs, rotate_angles


def loss_func(z, angles):
    bb, num_base_rot, num_nn, repr_dim = z.shape # 1, 8, 2, repr_dim

    from matplotlib import pyplot as plt
    for i in range(num_base_rot):
        fig = plt.figure()
        coords = z[0, i, :, :].detach().cpu().numpy()
        plt.scatter(coords[:, 0], coords[:, 1], c='b')
        coords_gt = angles[0, i, :].detach().cpu().numpy() * np.pi / 180
        coords_gt = np.stack([np.cos(coords_gt), np.sin(coords_gt)], axis=1)
        plt.scatter(coords_gt[:, 0], coords_gt[:, 1], c='r')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig("{}.png".format(i))

    from matplotlib import pyplot as plt
    for i in range(num_base_rot):
        fig = plt.figure()
        coords = z[0, :, i, :].detach().cpu().numpy()
        plt.scatter(coords[:, 0], coords[:, 1], c='b')
        coords_gt = angles[0, :, i].detach().cpu().numpy() * np.pi / 180
        coords_gt = np.stack([np.cos(coords_gt), np.sin(coords_gt)], axis=1)
        plt.scatter(coords_gt[:, 0], coords_gt[:, 1], c='r')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig("{}'.png".format(i))
    import pdb
    pdb.set_trace()

    cos_sim = torch.bmm(
        z.reshape(bb * num_base_rot, num_nn, repr_dim),
        z.transpose(-1, -2).reshape(bb * num_base_rot, repr_dim, num_nn),
    ).reshape(bb, num_base_rot, num_nn, num_nn)
    W = torch.repeat_interleave(angles.unsqueeze(2), angles.shape[2], dim=2)
    cos_sim_gt = torch.cos((W - W.transpose(-1, -2)) * torch.pi / 180)
    iso_loss = (cos_sim - cos_sim_gt).mul(2).exp().mean().log()

    uniform_loss = 0
    for i in range(num_nn):
        zz = z[:, :, i, :]
        cos_sim = torch.bmm(zz, zz.transpose(-1, -2))
        uniform_loss += cos_sim.mul(2).exp().mean().log()
    uniform_loss /= num_nn

    return 0.2 * uniform_loss + 0.8 * iso_loss


def adjust_learning_rate(epochs, warmup_epochs, base_lr, optimizer, loader, step):
    max_steps = epochs * len(loader)
    warmup_steps = warmup_epochs * len(loader)
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = 0
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def ssl_loop(args):
    if args.checkpoint_path:
        print('checkpoint provided => moving to evaluation')
        main_branch = GEncoder().to(args.device)
        saved_dict = torch.load(os.path.join(args.checkpoint_path))['state_dict']
        main_branch.load_state_dict(saved_dict)
        # file_to_update = open(os.path.join(args.path_dir, 'train_and_eval.log'), 'a')
        # file_to_update.write(f'evaluating {args.checkpoint_path}\n')
        # return main_branch.encoder, file_to_update

    # logging
    os.makedirs(args.path_dir, exist_ok=True)
    file_to_update = open(os.path.join(args.path_dir, 'train_and_eval.log'), 'a')

    # dataset
    dataset = MnistRotDataset(
        file=args.data_path + "mnist_all_rotation_normalized_float_train_valid.amat",
        transform=ContrastiveLearningTransform(args.num_base_rot, args.base_rot_range, args.num_nn,
                                               args.local_rot_range)
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=args.bsz,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    # models
    # main_branch = GEncoder().to(args.device)

    # optimization
    optimizer = torch.optim.SGD(
        main_branch.parameters(),
        momentum=0.9,
        lr=args.lr * args.bsz / 256,
        weight_decay=args.wd
    )

    # logging
    start = time.time()
    os.makedirs(args.path_dir, exist_ok=True)
    torch.save(dict(epoch=0, state_dict=main_branch.state_dict()), os.path.join(args.path_dir, '0.pth'))
    scaler = GradScaler()

    # training
    for e in range(1, args.epochs + 1):
        # declaring train
        main_branch.train()

        inputs = dataset[0]
        # adjust
        lr = adjust_learning_rate(epochs=args.epochs,
                                  warmup_epochs=args.warmup_epochs,
                                  base_lr=args.lr * args.bsz / 256,
                                  optimizer=optimizer,
                                  loader=train_loader,
                                  step=e)
        # zero grad
        main_branch.zero_grad()

        def forward_step():
            xs = inputs[0][0].to(args.device).unsqueeze(0)
            rotated_angles = inputs[0][1].to(args.device).unsqueeze(0)
            bb, num_base_rot, num_nn, cc, hh, ww = xs.shape
            zs = main_branch(xs.reshape(bb * num_base_rot * num_nn, cc, hh, ww)).reshape(bb, num_base_rot, num_nn, -1)

            return loss_func(zs, rotated_angles)

        # optimization step
        if args.fp16:
            with autocast():
                loss = forward_step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = forward_step()
            loss.backward()
            optimizer.step()

        line_to_print = (
            f'epoch: {e} | '
            f'loss: {loss.item():.7f} | lr: {lr:.6f} | '
            f'time_elapsed: {time.time() - start:.3f}'
        )
        if file_to_update:
            file_to_update.write(line_to_print + '\n')
            file_to_update.flush()
        print(line_to_print)

        if e % args.save_every == 0:
            torch.save(dict(epoch=e, state_dict=main_branch.state_dict()),
                       os.path.join(args.path_dir, f'{e}.pth'))

    return main_branch.encoder, file_to_update


def main(args):
    fix_seed(args.seed)
    ssl_loop(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--data_path', default='./data/mnist_rotation_new/', type=str)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--lr', default=0.06, type=float)
    parser.add_argument('--bsz', default=512, type=int)
    parser.add_argument('--wd', default=0.0005, type=float)
    parser.add_argument('--save_every', default=50, type=int)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--path_dir', default='experiment/LocalIsoCosSimRandomSingle/', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lmbd', default=0.0, type=float)
    parser.add_argument('--num_base_rot', default=8, type=int)
    parser.add_argument('--base_rot_range', default=360, type=int)
    parser.add_argument('--num_nn', default=8, type=int)
    parser.add_argument('--local_rot_range', default=30, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--checkpoint_path', default="experiment/LocalIsoCosSimRandomSingle/300.pth", type=str)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    main(args)
