import os
import math
import time
import copy
import argparse
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from resnet import resnet18
from utils import knn_monitor, fix_seed

normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
single_transform = T.Compose([T.ToTensor(), normalize])


class ContrastiveLearningTransform:
    def __init__(self):
        transforms = [
            T.RandomRotation(degrees=360, interpolation=TF.InterpolationMode.BILINEAR),
            T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2)
        ]
        transforms_rotation = [
            T.RandomRotation(degrees=360, interpolation=TF.InterpolationMode.BILINEAR),
            T.RandomResizedCrop(size=16, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2)
        ]

        self.transform = T.Compose(transforms)
        self.transform_rotation = T.Compose(transforms_rotation)

    def __call__(self, x):
        output = [
            single_transform(self.transform(x)),
            single_transform(self.transform(x)),
            single_transform(self.transform_rotation(x))
        ]
        return output


def rotate_images(images):
    rotated_angles = [float(random.random() * 20 - 10) for _ in range(7)]
    rotated_images = [TF.rotate(images, angle, interpolation=TF.InterpolationMode.BILINEAR) for angle in rotated_angles]
    rotated_images = [images] + rotated_images
    rotated_angles = [0] + rotated_angles

    rotated_images = torch.cat([torch.unsqueeze(rotated_img, dim=1) for rotated_img in rotated_images], dim=1)
    rotated_angles = torch.tensor(rotated_angles).cuda()
    return rotated_images, rotated_angles


def iso_loss_func(z_rotated, rotated_angles):
    param = F.normalize(z_rotated, p=2, dim=2)
    param_ortho = torch.stack([-param[:, :, 1], param[:, :, 0]], dim=-1)
    param = torch.stack([param, param_ortho], dim=-1)
    param = torch.repeat_interleave(param.unsqueeze(1), param.shape[1], dim=1)
    param_T = torch.transpose(torch.transpose(param, -1, -2), 1, 2)
    param = torch.matmul(param_T, param)

    W = rotated_angles.repeat(rotated_angles.shape[0], 1) * torch.pi / 180
    W = W - W.T
    sinW = torch.sin(W)
    cosW = torch.cos(W)
    W = torch.stack([torch.stack([cosW, -sinW], dim=-1),  torch.stack([sinW, cosW], dim=-1)], dim=-1)
    W = W.unsqueeze(0)
    W = torch.repeat_interleave(W, param.shape[0], dim=0)
    loss = F.mse_loss(param, W)
    return loss


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


def negative_cosine_similarity_loss(p, z):
    return - F.cosine_similarity(p, z.detach(), dim=-1).mean()


def info_nce_loss(z1, z2, temperature=0.5):
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)

    logits = z1 @ z2.T
    logits /= temperature
    n = z2.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=False),
                                 nn.BatchNorm1d(hidden_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, out_dim, bias=False),
                                 nn.BatchNorm1d(out_dim, affine=False))

    def forward(self, x):
        return self.net(x)


class Branch(nn.Module):
    def __init__(self, args, encoder=None):
        super().__init__()
        dim_proj = [int(x) for x in args.dim_proj.split(',')]
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = resnet18()
        self.projector = ProjectionMLP(512, dim_proj[0], dim_proj[1])
        self.net = nn.Sequential(
            self.encoder,
            self.projector
        )
        self.projector2 = ProjectionMLP(512, 256, 2)

    def forward(self, x):
        return self.net(x)


def knn_loop(encoder, train_loader, test_loader):
    accuracy = knn_monitor(net=encoder.cuda(),
                           memory_data_loader=train_loader,
                           test_data_loader=test_loader,
                           device='cuda',
                           k=200,
                           hide_progress=True)
    return accuracy


def ssl_loop(args, encoder=None):
    if args.checkpoint_path:
        print('checkpoint provided => moving to evaluation')
        main_branch = Branch(args, encoder=encoder).cuda()
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
        dataset=torchvision.datasets.CIFAR10(
            '../data', train=True, transform=ContrastiveLearningTransform(), download=True
        ),
        shuffle=True,
        batch_size=args.bsz,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    memory_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10(
            '../data', train=True, transform=single_transform, download=True
        ),
        shuffle=False,
        batch_size=args.bsz,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10(
            '../data', train=False, transform=single_transform, download=True,
        ),
        shuffle=False,
        batch_size=args.bsz,
        pin_memory=True,
        num_workers=args.num_workers
    )

    # models

    main_branch = Branch(args, encoder=encoder).cuda()

    # optimization
    optimizer = torch.optim.SGD(
        main_branch.parameters(),
        momentum=0.9,
        lr=args.lr * args.bsz / 256,
        weight_decay=args.wd
    )

    # macros
    backbone = main_branch.encoder
    projector = main_branch.projector

    # logging
    start = time.time()
    os.makedirs(args.path_dir, exist_ok=True)
    torch.save(dict(epoch=0, state_dict=main_branch.state_dict()), os.path.join(args.path_dir, '0.pth'))
    scaler = GradScaler()

    # training
    for e in range(1, args.epochs + 1):
        # declaring train
        main_branch.train()

        # epoch
        for it, (inputs, y) in enumerate(train_loader, start=(e - 1) * len(train_loader)):
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
                x1 = inputs[0].cuda()
                x2 = inputs[1].cuda()
                b1 = backbone(x1)
                b2 = backbone(x2)
                z1 = projector(b1)
                z2 = projector(b2)

                # forward pass
                if args.loss == 'simclr':
                    loss = info_nce_loss(z1, z2) / 2 + info_nce_loss(z2, z1) / 2
                else:
                    raise

                if args.lmbd > 0:
                    x = inputs[2].cuda()
                    rotated_images, rotated_angles = rotate_images(x)
                    bb, g, c, h, w = rotated_images.shape
                    b = backbone(rotated_images.reshape(bb * g, c, h, w))
                    z_rotated = main_branch.projector2(b).reshape(bb, g, -1)
                    iso_loss = iso_loss_func(z_rotated, rotated_angles)
                    loss += args.lmbd * iso_loss
                return loss

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

        if args.fp16:
            with autocast():
                knn_acc = knn_loop(backbone, memory_loader, test_loader)
        else:
            knn_acc = knn_loop(backbone, memory_loader, test_loader)

        line_to_print = (
            f'epoch: {e} | knn_acc: {knn_acc:.3f} | '
            f'loss: {loss.item():.3f} | lr: {lr:.6f} | '
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


def eval_loop(encoder, file_to_update, ind=None):
    # dataset
    train_transform = T.Compose([
        T.RandomResizedCrop(32, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ])
    test_transform = T.Compose([
        T.Resize(36, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(32),
        T.ToTensor(),
        normalize
    ])

    train_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10('../data', train=True, transform=train_transform, download=True),
        shuffle=True,
        batch_size=256,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10('../data', train=False, transform=test_transform, download=True),
        shuffle=False,
        batch_size=256,
        pin_memory=True,
        num_workers=args.num_workers
    )

    classifier = nn.Linear(512, 10).cuda()
    # optimization
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        momentum=0.9,
        lr=30,
        weight_decay=0
    )
    scaler = GradScaler()

    # training
    for e in range(1, 101):
        # declaring train
        classifier.train()
        encoder.eval()
        # epoch
        for it, (inputs, y) in enumerate(train_loader, start=(e - 1) * len(train_loader)):
            # adjust
            adjust_learning_rate(epochs=100,
                                 warmup_epochs=0,
                                 base_lr=30,
                                 optimizer=optimizer,
                                 loader=train_loader,
                                 step=it)
            # zero grad
            classifier.zero_grad()

            def forward_step():
                with torch.no_grad():
                    b = encoder(inputs.cuda())
                logits = classifier(b)
                loss = F.cross_entropy(logits, y.cuda())
                return loss

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

        if e % 10 == 0:
            accs = []
            classifier.eval()
            for idx, (images, labels) in enumerate(test_loader):
                with torch.no_grad():
                    if args.fp16:
                        with autocast():
                            b = encoder(images.cuda())
                            preds = classifier(b).argmax(dim=1)
                    else:
                        b = encoder(images.cuda())
                        preds = classifier(b).argmax(dim=1)
                    hits = (preds == labels.cuda()).sum().item()
                    accs.append(hits / b.shape[0])
            accuracy = np.mean(accs) * 100
            # final report of the accuracy
            line_to_print = (
                f'seed: {ind} | accuracy (%) @ epoch {e}: {accuracy:.2f}'
            )
            file_to_update.write(line_to_print + '\n')
            file_to_update.flush()
            print(line_to_print)

    return accuracy


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    fix_seed(args.seed)
    encoder, file_to_update = ssl_loop(args)
    accs = []
    for i in range(5):
        accs.append(eval_loop(copy.deepcopy(encoder), file_to_update, i))
    line_to_print = f'aggregated linear probe: {np.mean(accs):.3f} +- {np.std(accs):.3f}'
    file_to_update.write(line_to_print + '\n')
    file_to_update.flush()
    print(line_to_print)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_proj', default='2048,2048', type=str)
    parser.add_argument('--dim_pred', default=512, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--lr', default=0.06, type=float)
    parser.add_argument('--bsz', default=512, type=int)
    parser.add_argument('--wd', default=0.0005, type=float)
    parser.add_argument('--loss', default='simclr', type=str, choices=['simclr', 'simsiam'])
    parser.add_argument('--save_every', default=50, type=int)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--path_dir', default='../experiment', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lmbd', default=0.4, type=float)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--fp16', action='store_false')
    args = parser.parse_args()

    main(args)
