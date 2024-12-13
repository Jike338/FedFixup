import argparse
import os
import pdb
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision

from utils import *#DatasetSplit
import lenet
import resnet
from network import ResNet18
from network import WN_self

#
# (choose from 'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'googlenet', 'inception_v3', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf', 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_16gf', 'regnet_y_1_6gf', 'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'wide_resnet101_2', 'wide_resnet50_2')

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Style Training')
parser.add_argument('--data', default="/research/nfs_chao_209/james/dataset/imagenet", metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', default="imagenet", help='dataset name')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
parser.add_argument('-j', '--workers', default=24, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=128, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')                    
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--norm_layer', default=None, type=str)
parser.add_argument('--num_data', default=49000, type =int)  
parser.add_argument('--num_users', default=1, type=int, help='number of users')
parser.add_argument('--method', default="dir", type=str)
parser.add_argument('--alpha', default=100, type=int)
best_acc1 = 0


def main():
    args = parser.parse_args()

    for k, v in vars(args).items():
        print(k,v)
        
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def pop_weights(new_weights):
    try:
        new_weights.pop("module.fc.weight")
        new_weights.pop("module.fc.bias")
        new_weights.pop("fc.weight")
        new_weights.pop("fc.bias")
        new_weights.pop("linear.weight")
        new_weights.pop("linear.bias")
        new_weights.pop("classifier.linear.weight")
        new_weights.pop("classifier.linear.bias")
    except:
        pass 

def check_keys(old_weights, new_weights):
    cnt = 0
    for k in new_weights.keys():
        if k in old_weights.keys():
            cnt += 1
    print("%d out of %d keys matched"%(cnt, len(old_weights)))
    assert cnt > 0

def create_model(args):
    if "cifar" in args.dataset:
        args.num_classes = 100 if "cifar100" in args.dataset else 10
    
    # create model
    print("=> creating model '{}'".format(args.arch))
    if "lenet" in args.arch:

        if "cifar" in args.dataset:
            if args.norm_layer == "GroupNorm":
                model = lenet.lenet_gn(num_classes=args.num_classes)
            elif args.norm_layer=="BatchNorm":
                model = lenet.lenet_bn(num_classes=args.num_classes)
        elif "tiny" in args.dataset:
            if args.norm_layer == "GroupNorm":
                model = lenet.lenet_gn_tiny(num_classes=args.num_classes)
            elif args.norm_layer=="BatchNorm":
                model = lenet.lenet_bn_tiny(num_classes=args.num_classes)

    elif "resnet" in args.arch:
        original_norm_layer = str(args.norm_layer)
        if args.norm_layer=="GroupNorm":
            args.norm_layer = GroupNorm2d
        elif args.norm_layer =="BatchNorm":
            args.norm_layer = nn.BatchNorm2d
        elif args.norm_layer == "InstanceNorm":
            args.norm_layer = nn.InstanceNorm2d
        
        if "resnet18" in args.arch:
            if "WeightNormalization" in original_norm_layer:
                model = ResNet18(num_classes=args.num_classes)
            else:
                # model = models.__dict__[args.arch](norm_layer=args.norm_layer, num_classes=args.num_classes)
                model = resnet.resnet18(norm_layer=args.norm_layer, num_classes=args.num_classes)
        elif "resnet20" in args.arch:
                model = resnet.resnet20(norm_layer=args.norm_layer, num_classes=args.num_classes)
        elif "resnet50" in args.arch:
            model = resnet.resnet50(norm_layer=args.norm_layer, num_classes=args.num_classes)

        if "cifar" in args.dataset and "resnet20" not in args.arch:
            if "WeightNormalization" in original_norm_layer:
                model.conv1 = WN_self(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
            else:
                model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                    bias=False)

    if "inat" in args.dataset:
        feat_dim = 2048 if args.arch == "resnet50" else 512
        if "mobilenet" in args.arch:
            model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, args.num_classes),
            )
        else:
            model.fc = nn.Linear(feat_dim, args.num_classes)
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            #model = torch.nn.DataParallel(model).cuda()
            model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
                                
    return model, criterion, optimizer 

def load_client_data(args):
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if "cifar" in args.dataset:
        train_transform = transforms.Compose(
            [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize])
        valid_transform = transforms.Compose(
            [
            transforms.ToTensor(),
            normalize])
        if "cifar10" == args.dataset:
            train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
            val_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
            test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=valid_transform)
        elif "cifar100" == args.dataset:
            train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=train_transform)
            val_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=train_transform)
            test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=valid_transform)         
        assert test_dataset.class_to_idx == train_dataset.class_to_idx
    else:
        if args.dataset=="tiny":    
            train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(64),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
                
            valid_transform = transforms.Compose([
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    normalize,
                ])   
        
        else:
            train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
                
            valid_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])  
            
        train_dataset = datasets.ImageFolder(
            traindir,
            train_transform
            )
            
        val_dataset = datasets.ImageFolder(
            traindir, 
            valid_transform
            )
        test_dataset = datasets.ImageFolder(valdir, valid_transform)

        assert test_dataset.class_to_idx == train_dataset.class_to_idx
    
    if args.dataset=="imagenet":
        args.num_classes = 1000
        dict_users, server_id, cnts_dict = imagenet_noniid(train_dataset, args.num_users, tiny=False, num_data=args.num_data, method=args.method, alpha=args.alpha)
    elif args.dataset=="tiny":
        args.num_classes = 200
        dict_users, server_id, cnts_dict = imagenet_noniid(train_dataset, args.num_users, num_data=args.num_data, method=args.method, alpha=args.alpha)
    elif "inat" in args.dataset:
        args.num_classes = 1203
        dict_users, server_id, cnts_dict = inat_geo_noniid(args.dataset, train_dataset, num_val_data=args.num_classes*5)
    elif "cifar" in args.dataset:
        if args.dataset == "cifar100":
            args.num_classes = 100 
            dict_users, server_id, cnts_dict = cifar100_noniid(train_dataset, args.num_users, num_data=args.num_data, method=args.method, alpha=args.alpha)
        elif args.dataset == "cifar10":
            args.num_classes = 10
            dict_users, server_id, cnts_dict = cifar_noniid(train_dataset, args.num_users, num_data=args.num_data, method=args.method, alpha=args.alpha)



    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
            DatasetSplit(train_dataset, dict_users[0]),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        DatasetSplit(val_dataset, server_id), 
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)    
        
    return train_loader, val_loader, test_loader, cnts_dict, train_sampler

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    args.logdir = "log_%s_%s"%(args.dataset, args.arch)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    # model = models.__dict__[args.arch]()
    model, _, _ = create_model(args)
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        if args.pretrained == "imagenet":
            print("=> loading ImageNet checkpoint")
            pretrained_model = models.__dict__[args.arch](pretrained=True)
            new_weights = pretrained_model.state_dict()
            pop_weights(new_weights)
            
            model.load_state_dict(new_weights, strict=False)
            args.logdir += "_imgnet"
            
        else:
            print("=> loading checkpoint '{}'".format(args.pretrained))
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(args.pretrained)
            
            if "places" in args.pretrained:
                checkpoint = checkpoint["state_dict"]  
            pop_weights(checkpoint)    
            
            new_weights = {}
            for k, v in checkpoint.items():
                if "module." in k: 
                    k = k.replace("module.", "")
                new_weights[k] = v
            
            model.load_state_dict(new_weights, strict=False) #, strict=False
            
            name = [n.strip(".") for n in args.pretrained.split('/') if ("imagenet" in n or "fractal" in n or "place" in n)]
            init = args.pretrained.split('/')[-2] + "_".join(name)
            
            args.logdir += init

        check_keys(model.state_dict(), new_weights)

    if args.dataset == "inat":
        num_classes = 1203 
        model.fc = nn.Linear(512, num_classes)
    elif args.dataset=="tiny":
        num_classes = 200
        model.fc = nn.Linear(512, num_classes)
        
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    
    # # Data loading code
    # traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')
    
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # if args.dataset=="tiny":    
    #     train_transform = transforms.Compose([
    #             transforms.RandomResizedCrop(64),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             normalize,
    #         ])
            
    #     valid_transform = transforms.Compose([
    #             transforms.Resize(64),
    #             transforms.ToTensor(),
    #             normalize,
    #         ])   
    
    # else:
    #     train_transform = transforms.Compose([
    #             transforms.RandomResizedCrop(224),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             normalize,
    #         ])
            
    #     valid_transform = transforms.Compose([
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             normalize,
    #         ])  
        
    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     train_transform
    #     )
        
    # valid_dataset = datasets.ImageFolder(
    #     valdir, 
    #     valid_transform
    #     )

    
    # assert valid_dataset.class_to_idx == train_dataset.class_to_idx
    


    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    #     num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # val_loader = torch.utils.data.DataLoader(
    #     valid_dataset,
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)


    train_loader, val_loader, test_loader, cnts_dict, train_sampler = load_client_data(args)
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    
    
    print("Start training ...")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.logdir, filename='checkpoint_ep%d.pth.tar'%epoch if epoch%10==0 else 'checkpoint.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

    return top1.avg


def save_checkpoint(state, is_best, logdir, filename='checkpoint.pth.tar'):
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    filename = os.path.join(logdir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(logdir, 'model_best.pth.tar'))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# def adjust_learning_rate(optimizer, epoch, args):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch > 128:
        lr = args.lr * 0.001
    elif epoch > 96:
        lr = args.lr * 0.01
    elif epoch > 64:
        lr = args.lr * 0.1
    else:
        lr = args.lr 

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()