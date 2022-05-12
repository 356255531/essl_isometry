import os
import math
import time
import argparse
import random
import numpy as np
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        return torch.stack(xs, dim=0).reshape(self.num_base_rotation, self.num_nn, cc, hh, ww), torch.tensor(rotate_angles).reshape(self.num_base_rotation, self.num_nn)


def cos_sim_loss(z, angles):
    bb, num_base_rot, num_nn, repr_dim = z.shape

    param = torch.repeat_interleave(z.unsqueeze(1), z.shape[1], dim=1)
    param_T = param.transpose(-1, -2).transpose(1, 2)
    param = torch.bmm(
        param.reshape(bb * num_base_rot * num_base_rot, num_nn, repr_dim),
        param_T.reshape(bb * num_base_rot * num_base_rot, repr_dim, num_nn)
    ).view(bb, num_base_rot, num_base_rot, num_nn, num_nn)
    logits = torch.linalg.matrix_norm(param, 'fro')
    labels = torch.arange(0, num_base_rot, dtype=torch.long).to(logits.device)
    labels = torch.repeat_interleave(labels.unsqueeze(0), bb, dim=0)
    equi_contrastive_loss = F.cross_entropy(logits, labels)

    cos_sim = torch.bmm(
        z.reshape(bb * num_base_rot, num_nn, repr_dim),
        z.transpose(-1, -2).reshape(bb * num_base_rot, repr_dim, num_nn),
    ).reshape(bb, num_base_rot, num_nn, num_nn)
    W = torch.repeat_interleave(angles.unsqueeze(2), angles.shape[2], dim=2)
    cos_sim_gt = torch.cos((W - W.transpose(-1, -2)) * torch.pi / 180)
    equi_iso_loss = F.mse_loss(cos_sim, cos_sim_gt)

    return equi_contrastive_loss, equi_iso_loss


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
        file_to_update = open(os.path.join(args.path_dir, 'train_and_eval.log'), 'a')
        file_to_update.write(f'evaluating {args.checkpoint_path}\n')
        return main_branch.encoder, file_to_update

    # logging
    os.makedirs(args.path_dir, exist_ok=True)
    file_to_update = open(os.path.join(args.path_dir, 'train_and_eval.log'), 'w')

    # dataset
    train_loader = torch.utils.data.DataLoader(
        dataset=MnistRotDataset(
            file=args.data_path + "mnist_all_rotation_normalized_float_train_valid.amat",
            transform=ContrastiveLearningTransform(args.num_base_rot, args.base_rot_range, args.num_nn, args.local_rot_range)
        ),
        shuffle=True,
        batch_size=args.bsz,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    # models
    main_branch = GEncoder().to(args.device)

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

    # training
    for e in range(1, args.epochs + 1):
        # declaring train
        main_branch.train()

        losses = []
        equi_contrastive_losses = []
        equi_iso_losses = []
        # epoch
        pbar = tqdm.tqdm(enumerate(train_loader, start=(e - 1) * len(train_loader)), total=len(train_loader))
        for it, (inputs, y) in pbar:
            # adjust
            lr = adjust_learning_rate(epochs=args.epochs,
                                      warmup_epochs=args.warmup_epochs,
                                      base_lr=args.lr * args.bsz / 256,
                                      optimizer=optimizer,
                                      loader=train_loader,
                                      step=it)
            # zero grad
            main_branch.zero_grad()

            def forward_step():
                xs = inputs[0].to(args.device)
                rotated_angles = inputs[1].to(args.device)
                bb, num_base_rot, num_nn, cc, hh, ww = xs.shape
                zs = main_branch(xs.reshape(bb * num_base_rot * num_nn, cc, hh, ww)).reshape(bb, num_base_rot, num_nn, -1)

                return cos_sim_loss(zs, rotated_angles)

            # optimization step
            equi_contrastive_loss, equi_iso_loss = forward_step()
            loss = (1 - args.lmbd) * equi_contrastive_loss + args.lmbd * equi_iso_loss
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().numpy())
            equi_contrastive_losses.append(equi_contrastive_loss.detach().cpu().numpy())
            equi_iso_losses.append(equi_iso_loss.detach().cpu().numpy())
            pbar.set_description(
                "epoch: {} | loss: {:.7f} | equi contrastive loss: {:.7f} | "
                "equi iso loss: {:.7f}".format(
                    e, np.mean(losses), np.mean(equi_contrastive_losses), np.mean(equi_iso_losses)))

        line_to_print = (
            f'epoch: {e} | '
            f'loss: {np.mean(losses):.3f} | '
            f'equi contrastive loss: {np.mean(equi_contrastive_losses):.3f} | '
            f'equi iso loss: {np.mean(equi_iso_losses):.3f} | '            
            f'lr: {lr:.6f} | '
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
    parser.add_argument('--path_dir', default='experiment/LocalIsoCosSimContrastive/', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lmbd', default=0.1, type=float)
    parser.add_argument('--num_base_rot', default=8, type=int)
    parser.add_argument('--base_rot_range', default=180, type=int)
    parser.add_argument('--num_nn', default=8, type=int)
    parser.add_argument('--local_rot_range', default=10, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--checkpoint_path', default=None, type=str)
    args = parser.parse_args()

    main(args)
