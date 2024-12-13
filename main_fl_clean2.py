import argparse
import os
import copy
import random
import shutil
import time
import math
import json
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
import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter

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


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default="/research/nfs_chao_209/james/dataset/imagenet", metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', default="imagenet", help='dataset name')                    
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
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
parser.add_argument('--pretrained', default=None, type=str, dest='pretrained', 
                    help='use pre-trained model')
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

# FL setting
parser.add_argument('--num_users', default=10, type=int, help='number of users')
parser.add_argument('--num_val_data', default=10000, type=int, help='number of validation set')
parser.add_argument('--alpha', default=1.0, type=float, help='alpha for Dir(alpha) non-IID')
parser.add_argument('--frac', default=1.0, type=float, help='alpha for Dir(alpha) non-IID')
parser.add_argument('--local_epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--rounds', default=100, type=int, metavar='N',
                    help='number of total epochs to run') 
parser.add_argument('--start_round', default=0, type=int, metavar='N',
                    help='number of total epochs to run')       
parser.add_argument('--freeze_bn', default=0, type=int,
                    help='freeze bn during training')
parser.add_argument('--save_dir', default='/research/nfs_chao_209/james/log_224', type=str,
                    help='distributed backend')
                    
parser.add_argument('--lr_decay', default="step", help='step, cosine, exp, none')                      
parser.add_argument('--local_training', default='1stage', type=str,
                    help='local training: 1stage, bnfc, fc, bn') 
parser.add_argument('--num_classes', default=10, type =int)  
parser.add_argument('--num_data', default=10, type =int)  
parser.add_argument('--steps', default=0, type =int)  
parser.add_argument('--log_every_round', default=200, type =int)  
parser.add_argument('--freeze_bn_at_round', default=0, type =int)  
parser.add_argument('--clip_grad', default=0, type =float)  
parser.add_argument('--norm_layer', default=None, type=str)
parser.add_argument('--method', default="dir", type=str)
parser.add_argument('--id', default="test_run/tensorboard/test", type=str)
parser.add_argument('--use_true_stats', action='store_true')
parser.add_argument('--probe_activs', action='store_true')
parser.add_argument('--update_bn_after_freeze', action='store_true')
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--fix_warmup', action='store_true')
parser.add_argument("--freeze_bn_params", action='store_true')


best_acc1 = 0
# torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    
    args = parser.parse_args()

    for k, v in vars(args).items():
        print(k,v)
        
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        #torch.manual_seed(args.seed)
        #cudnn.deterministic = True

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

def change_fc_and_conv_and_send_model(model, args):
    feat_dim = 2048 if args.arch == "resnet50" else 512
    model.fc = nn.Linear(feat_dim, args.num_classes)
    if args.dataset == "cifar10":
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                    bias=False)
    if torch.cuda.is_available():
        model.cuda()

def show_stats(net):
    for m in net.modules():
        if isinstance(m, nn.GroupNorm):
            print(m.affine)
        elif isinstance(m, nn.BatchNorm2d):
            print(m.running_mean)

def freeze_bn(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def remove_fc(net, args):
    if "resnet" in args.arch:
        modules=list(net.children())[:-2]
        net=nn.Sequential(*modules)
        for p in net.parameters():
            p.requires_grad = False
    elif "lenet" in args.arch:
        modules=list(net.children())[:-1]
        net=nn.Sequential(*modules)
        for p in net.parameters():
            p.requires_grad = False    
    return net

def l2_distance(alist, blist):
    sum_of = 0
    for x, y in zip(alist, blist):
        ans = (x - y)**2
        sum_of += ans
    return (sum_of)**(1/2)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.logdir = "log_fl_%s_a=%.1f_user=%d_frac=%.1f_arch=%s_freezeBN=%d_%s_split=%s_bs=%d_lr=%.3f_id=%s"%(args.dataset, args.alpha, args.num_users, args.frac, args.arch, args.freeze_bn_at_round, args.norm_layer, args.method, args.batch_size, args.lr, args.id)
    # summary_writer_logdir = "runs/log_fl_%s_a=%.1f_user=%d_frac=%.1f_arch=%s_freezeBN=%d_%s_split=%s_bs=%d_lr=%.3f_id=%s"%(args.dataset, args.alpha, args.num_users, args.frac, args.arch, args.freeze_bn_at_round, args.norm_layer, args.method, args.batch_size, args.lr, args.id)
    summary_writer_logdir = args.id
    writer = SummaryWriter(log_dir=summary_writer_logdir)

    if args.dataset == "imagenet": 
        args.logdir += "u%d_alpha%.1f"%(args.num_users, args.alpha)
    if args.freeze_bn_at_round == 1:
        args.logdir += "_freezeBN"
    if args.local_training != '1stage':
        args.logdir += f"_{args.local_training}"        
        
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    cudnn.benchmark = True
    # Data loading code
    client_train_loaders, val_loader, test_loader, cnts_dict, train_loader_all = load_client_data(args)    

    # Create initial model
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
        if "tiny" in args.pretrained or "cifar" in args.pretrained:
            print("=> loading checkpoint '{}'".format(args.pretrained))
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(args.pretrained)
            
            # if "places" in args.pretrained:
            checkpoint = checkpoint["state_dict"]  
            # pop_weights(checkpoint)

            new_weights = {}
            for k, v in checkpoint.items():
                #if "module." not in k: 
                #    k = "module."+k
                new_weights[k.replace("module.", "")] = v
            
            model.load_state_dict(new_weights, strict=False) #, strict=False
            
            name = [n.strip(".") for n in args.pretrained.split('/') if ("imagenet" in n or "fractal" in n or "place" in n)]
            init = args.pretrained.split('/')[-2] + "_".join(name)
            
            args.logdir += init       
        else:
            print("=> loading checkpoint '{}'".format(args.pretrained))
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(args.pretrained)
            
            # if "places" in args.pretrained:
            checkpoint = checkpoint["state_dict"]  
            pop_weights(checkpoint)

            new_weights = {}
            for k, v in checkpoint.items():
                #if "module." not in k: 
                #    k = "module."+k
                new_weights[k.replace("module.", "")] = v
            
            model.load_state_dict(new_weights, strict=False) #, strict=False
            
            name = [n.strip(".") for n in args.pretrained.split('/') if ("imagenet" in n or "fractal" in n or "place" in n)]
            init = args.pretrained.split('/')[-2] + "_".join(name)
            
            args.logdir += init

        check_keys(model.state_dict(), new_weights)

    elif args.resume:
        checkpoint = torch.load(args.resume)["state_dict"]
        model.load_state_dict(checkpoint)
        args.logdir += "_resume_"+args.resume.split("/")[-1]
    
    if args.evaluate:
        validate(test_loader, model, criterion, args)
        return

    print(model)

    w_glob_avg = model.state_dict() 
    accs = []   
    activations = {} #{r: true, false}
    activations_var = {}
    activations_rank = {}
    activation_single = {}
    activation_quantile = {}

    client_train_iters = {}
    for key, val in client_train_loaders.items():
        client_train_iters[key] = iter(client_train_loaders[key])

    for r in range(args.start_round, args.rounds, 1):
        print("Round", r)
        w_locals = []
        participating_size = []
        
        client_ids_round = np.random.choice(list(client_train_loaders.keys()), int(len(client_train_loaders)*args.frac), replace=False)
        local_acc = []
        local_knn_acc = []
        w_glob_avg_old_bn_before_train = copy.deepcopy(w_glob_avg)

        if args.probe_activs:
            if r % args.log_every_round == 0 or r==args.rounds-1:
                total_activs_mean_false = None
                total_activs_mean_true = None
                total_activs_var_false = None
                total_activs_var_true = None
                total_activs_rank_false = None
                total_activs_rank_true = None
                total_quantile_mean_false = []
                total_quantile_mean_true = []
                total_loss_false = []
                total_loss_true = []
                total = 0
                part_size = {}
                part_size_dict = {}
                for client_id in client_ids_round:
                    part_size[client_id] = np.sum(cnts_dict[client_id])
                    total += np.sum(cnts_dict[client_id])
                for client_id in client_ids_round:
                    part_size_dict[client_id] = part_size[client_id]/total

                for client_id in client_ids_round:
                    print("Start inferencing client", client_id)
                    train_loader = client_train_loaders[client_id]
                    model_false, criterion_false, _ = create_model(args)
                    model_true, criterion_true, _ = create_model(args)

                    model_false.load_state_dict(copy.deepcopy(w_glob_avg))
                    model_true.load_state_dict(copy.deepcopy(w_glob_avg))

                    # model_false = remove_fc(model_false, args)
                    # model_true = remove_fc(model_true, args)

                    activs_mean_false, activs_var_false, activs_rank_false, quantile_mean_false, loss_false = inference(train_loader, criterion_false, model_false, args)
                    total_activs_mean_false = activs_mean_false if total_activs_mean_false is None else torch.cat((total_activs_mean_false, activs_mean_false))
                    total_activs_var_false =  activs_var_false if total_activs_var_false is None else torch.cat((total_activs_var_false, activs_var_false))
                    total_activs_rank_false = activs_rank_false if total_activs_rank_false is None else total_activs_rank_false+activs_rank_false
                    total_quantile_mean_false.append(quantile_mean_false)
                    total_loss_false.append(loss_false)

                    update_bn(train_loader, model_true)
                    activs_mean_true, activs_var_true, activs_rank_true, quantile_mean_true, loss_true  = inference(train_loader, criterion_true,model_true, args)
                    total_activs_mean_true = activs_mean_true  if total_activs_mean_true is None else torch.cat((total_activs_mean_true,activs_mean_true))
                    total_activs_var_true = activs_var_true  if total_activs_var_true is None else torch.cat((total_activs_var_true, activs_var_true))
                    total_activs_rank_true = activs_rank_true if total_activs_rank_true is None else total_activs_rank_true+activs_rank_true
                    total_quantile_mean_true.append(quantile_mean_true)
                    total_loss_true.append(loss_true)

                    if client_id == 0:
                        single_activs_false = inference_single(train_loader, model_false, args)
                        single_activs_true = inference_single(train_loader, model_true, args)

                writer.add_scalar("activs_l2/mean", l2_distance(total_activs_mean_true, total_activs_mean_false), r)
                writer.add_scalar("activs_l2/var", l2_distance(total_activs_var_true, total_activs_var_false), r)
                writer.add_scalar("activs_l2/rank", l2_distance([total_activs_rank_true], [total_activs_rank_false]), r)
                writer.add_scalar("activs_mean/true", torch.mean(total_activs_mean_true).item(), r)
                writer.add_scalar("activs_mean/false", torch.mean(total_activs_mean_false).item(), r)
                writer.add_scalar("activs_mean/true_false", torch.mean(total_activs_mean_true).item()-torch.mean(total_activs_mean_false).item(), r)       
                writer.add_scalar("activs_var/true", torch.mean(total_activs_var_true).item(), r)
                writer.add_scalar("activs_var/false", torch.mean(total_activs_var_false).item(), r)  
                writer.add_scalar("activs_var/true_false", torch.mean(total_activs_var_true).item() - torch.mean(total_activs_var_false).item(), r)
                writer.add_scalar("activs_rank/true", total_activs_rank_true, r)
                writer.add_scalar("activs_rank/false", total_activs_rank_false, r)   
                writer.add_scalar("activs_rank/true_false", total_activs_rank_true-total_activs_rank_false, r)             
                writer.add_scalar("activs_quantile/true", np.mean(total_quantile_mean_true), r)
                writer.add_scalar("activs_quantile/false", np.mean(total_quantile_mean_false), r)
                writer.add_scalar("activs_quantile/true_false", np.mean(total_quantile_mean_true) - np.mean(total_quantile_mean_false), r)
                writer.add_scalar("train_loss/true", np.mean(total_loss_true), r)
                writer.add_scalar("train_loss/false", np.mean(total_loss_false), r)
                writer.add_scalar("train_loss/true_false", np.mean(total_loss_true) - np.mean(total_loss_false), r)

                print("Writing activations...\n")

                mean_file, var_file, rank_file, single_file = create_activs_probe_path(args)

                if 0 in client_ids_round:
                    activation_single[r] = [[float(i) for i in list(single_activs_true)], [float(i) for i in list(single_activs_false)]]
                    with open(single_file, 'w') as convert_file:
                        convert_file.write(json.dumps(activation_single))

                activations[r] = [[float(i) for i in list(total_activs_mean_true)], [float(i) for i in list(total_activs_mean_false)]]
                with open(mean_file, 'w') as convert_file:
                    convert_file.write(json.dumps(activations))

                activations_var[r] = [[float(i) for i in list(total_activs_var_true)], [float(i) for i in list(total_activs_var_false)]]
                with open(var_file, 'w') as convert_file:
                    convert_file.write(json.dumps(activations_var))

                activations_rank[r] = [total_activs_rank_true, total_activs_rank_false]
                with open(rank_file, 'w') as convert_file:
                    convert_file.write(json.dumps(activations_rank))

        total_local_loss = []

        model, criterion, optimizer = create_model(args)
        model.load_state_dict(copy.deepcopy(w_glob_avg))
        if args.freeze_bn_at_round > 0 and r >=args.freeze_bn_at_round:
            print("updating BN w entire train set")
            update_bn(train_loader_all, model, args)
        w_glob_avg = model.state_dict() 

        for client_id in client_ids_round:
            print("Start training client", client_id)
            train_loader = client_train_iters[client_id]
            minibatch_curr = []

            # Samples the batch
            for i in range(args.steps):
                try:
                    minibatch = next(train_loader)
                except StopIteration:
                    client_train_iters[client_id] = iter(client_train_loaders[client_id])
                    train_loader = client_train_iters[client_id]
                    minibatch = next(train_loader)
                minibatch_curr.append(minibatch)

            if args.fix_warmup and r >= args.rounds/3:
                args.norm_layer = "BatchNorm"
                model, criterion, optimizer = create_model(args)
                model.load_state_dict(copy.deepcopy(w_glob_avg), strict=False)
                check_keys(model.state_dict(), w_glob_avg)
            else:
                model, criterion, optimizer = create_model(args)
                model.load_state_dict(copy.deepcopy(w_glob_avg))

            if args.use_true_stats:
                if args.freeze_bn_at_round == 0:
                    update_bn(train_loader, model)
                if args.freeze_bn_at_round > 0 and r < args.freeze_bn_at_round:
                    update_bn(train_loader, model)

            if args.steps > 0:
                adjust_learning_rate_with_step(optimizer, r, args)
                train_with_step_minibatch(minibatch_curr, model, criterion, optimizer, 0, args, r)
            else:
                adjust_learning_rate(optimizer, r, args)
                for epoch in range(args.local_epochs):
                    train(train_loader, model, criterion, optimizer, epoch, args, r)


            if args.freeze_bn_at_round > 0 and r >=args.freeze_bn_at_round and args.update_bn_after_freeze:
                update_bn(train_loader, model)
                
            if args.probe_activs:
                if r % args.log_every_round == 0 or r==args.rounds-1:
                    _, _, _, _, local_loss = inference(train_loader, criterion, model, args)
                    total_local_loss.append(local_loss)

            model.cpu()
            w_locals.append(copy.deepcopy(model.state_dict()))
            participating_size.append(np.sum(cnts_dict[client_id]))
            
            
            # if r%100==0 and args.save_model:
            #     save_checkpoint({
            #         'epoch': r + 1,
            #         'arch': args.arch,
            #         'state_dict': model.state_dict(),
            #         'best_acc1': best_acc1,
            #         'optimizer' : optimizer.state_dict(),
            #     }, False, args.id.split("/")[0]+"/models", filename='%s_checkpoint_r%d.pth.tar'%(args.id.split("/")[2],r))           
            

        

        w_glob_avg = FedAvg(w_locals, size_arr=participating_size)

        if r % args.log_every_round !=0 and r != args.rounds -1:
            continue

        if args.probe_activs:
            writer.add_scalar("train_loss/local", np.mean(total_local_loss), r)

        model, criterion, _ = create_model(args)
        model.cpu()
        model.load_state_dict(copy.deepcopy(w_glob_avg))
        
        # evaluate on validation set
        model.cuda()
        
        if r%50==0 and args.save_model:
            save_checkpoint({
                'epoch': r + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, False, args.id.split("/")[0]+"/models", filename='%s_checkpoint_r%d.pth.tar'%(args.id.split("/")[2],r))
            
        print("\n**********")
        print("Round FedAvg", r)
        acc1 = validate(test_loader, model, criterion, args)   
        accs.append(float("{:.2f}".format(acc1.item())))
        writer.add_scalar("FedAvg/test", acc1.item(), r)
        
        # print("Round Local Avg.", r)
        # print("Linear", "{:.2f}".format(np.mean(local_acc)))

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        print("Best", "{:.2f}".format(best_acc1.item()))
        writer.add_scalar("FedAvg/best", best_acc1.item(), r)

        #evaluete on validation set with updated BN after freeze
        if args.freeze_bn_at_round > 0 and r >=args.freeze_bn_at_round and args.update_bn_after_freeze:
            w_glob_avg_old_bn_after_train = copy.deepcopy(w_glob_avg)
            w_glob_avg_old_bn_after_train.update({k:v for k,v in w_glob_avg_old_bn_before_train.items() if ("running_mean" in k or "running_var" in k)})
            model, criterion, _ = create_model(args)
            model.cpu()
            model.load_state_dict(copy.deepcopy(w_glob_avg_old_bn_after_train))
            model.cuda()
            print("\n**********")
            print("Round FedAvg BN Updated After Freeze", r)
            acc1_ = validate(test_loader, model, criterion, args)   
            writer.add_scalar("FedAvg/BN_updated_after_freeze", acc1_.item(), r)

        writer.flush()
        print(accs)
        print("**********\n")
        
        
    print("Final")
    print("Best", "{:.2f}".format(best_acc1.item()))
    print(accs)

    # if args.probe_activs:
    #     print("Writing activations...")
    #     with open(mean_file, 'w') as convert_file:
    #         convert_file.write(json.dumps(activations))
    #     with open(var_file, 'w') as convert_file:
    #         convert_file.write(json.dumps(activations_var))
    #     with open(rank_file, 'w') as convert_file:
    #         convert_file.write(json.dumps(activations_var))

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


    client_train_loader_dict = {}
    train_loader_all = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

    for user_id, ids in dict_users.items():
        train_loader = torch.utils.data.DataLoader(
            DatasetSplit(train_dataset, ids),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        client_train_loader_dict[user_id] = train_loader   

    val_loader = torch.utils.data.DataLoader(
        DatasetSplit(val_dataset, server_id), 
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)    
        
    return client_train_loader_dict, val_loader, test_loader, cnts_dict, train_loader_all


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


            
def train_warmup(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        args,
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    
    disable_model_layers(model)
    
    # Freeze all parameters but warm-up parameters
    if "bn" in args.local_training:
        enable_bn_layers(model)
        
    if "fc" in args.local_training:
        enable_fc(model)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda()
        target = target.cuda()

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
            
    enable_model_layers(model)        

def train_with_step_minibatch(train_loader, model, criterion, optimizer, epoch, args, r):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        args,
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Step: [{}]".format(epoch))

    end = time.time()

    # switch to train mode
    model.train()

    if args.freeze_bn_at_round > 0 and r >=args.freeze_bn_at_round:
        if args.freeze_bn_params:
            disable_bn_layers(model)
            print("BN params Freezed")
        else:
            # switch to train mode
            model.train()
            freeze_bn(model)
            print("BN Stats Freezed")

    data_time.update(time.time() - end)

    idx = 0 
    for minibatch in train_loader:
        images, target = minibatch
        
        

        images = images.cuda()
        target = target.cuda()

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
        if args.freeze_bn_at_round > 0 and r >=args.freeze_bn_at_round and args.clip_grad>0:
            torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=args.clip_grad, norm_type=2.0)
        optimizer.step()

        # measure elapsed time
        idx += 1

    batch_time.update(time.time() - end)
    end = time.time()

    progress.display(idx)
    
def train_with_step(train_loader, model, criterion, optimizer, epoch, args, r):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        args,
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    if args.freeze_bn_at_round > 0 and r >=args.freeze_bn_at_round:
        freeze_bn(model)
        print("BN Freezed")

    if args.use_true_stats:
        freeze_bn(model)
        print("BN Freezed")      
    '''
    if args.freeze_bn > 0 and args.round > args.freeze_bn:
        print("BN Freezed")
        freeze_bn(model)
    '''
    end = time.time()
    total_step = 0
    train_loader_iter = iter(train_loader)
    while total_step < args.steps:
        try:
            # Samples the batch
            images, target = next(train_loader_iter)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            train_loader_iter = iter(train_loader)
            images, target = next(train_loader_iter)

        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda()
        target = target.cuda()

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

        if total_step % args.print_freq == 0:
            progress.display(total_step)
        total_step+=1

def train(train_loader, model, criterion, optimizer, epoch, args, r):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        args,
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    if args.freeze_bn_at_round > 0 and r >=args.freeze_bn_at_round:
        freeze_bn(model)
        print("BN Freezed")

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda()
        target = target.cuda()

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


def validate_rewind_features(val_loader, local_features, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        args,
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    features = torch.from_numpy(local_features).cuda()
    
    cnt = 0
    with torch.no_grad():
        end = time.time()
        for i, (_, target) in enumerate(val_loader):
            target = target.cuda()
            feature_batch = features[cnt:cnt+len(target)]
            cnt += len(target)
            
            # compute output
            output = model.fc(feature_batch)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), output.size(0))
            top1.update(acc1[0], output.size(0))
            top5.update(acc5[0], output.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

    return top1.avg
    
def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        args,
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

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

def extract_features(model, loader, args):
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(loader):
            images = images.cuda()

            # compute output
            output = model(images)
            
            if i==0: 
                features = output.cpu().numpy()
                labels = target.numpy()
                
            else:
                features = np.append(features, output.cpu().numpy(), axis=0)
                labels = np.append(labels, target.numpy())
                
    return np.squeeze(features), labels

    
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
    def __init__(self, args, num_batches, meters, prefix=""):
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


def FedAvg(w, global_w=None, size_arr=None):
    w_avg = {}
    for k in w[0].keys():
      if "num_batches_tracked" in k: continue
      w_avg[k] = torch.zeros(w[0][k].size())
      
    # Prepare p 
    if size_arr is not None:
      total_num = np.sum(size_arr)
      size_arr = np.array([float(p)/total_num for p in size_arr])*len(size_arr)
    else:
      size_arr = np.array([1.0]*len(size_arr))

    for k in w_avg.keys():
      if "num_batches_tracked" in k: continue 
      for i in range(0, len(w)):
        w_avg[k] += size_arr[i]*w[i][k] 
      w_avg[k] = torch.div(w_avg[k], len(w))
          
    return w_avg
    
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.lr_decay == "step":
        lr = args.lr * (0.1 ** (epoch // 30))
    elif args.lr_decay == "exp":
        lr = args.lr * (0.95 ** epoch)
    elif args.lr_decay == "cosine":
        decay_steps = 10
        step = min(epoch, decay_steps)
        alpha = 0.01
        cosine_decay = 0.5 * (1 + math.cos(math.pi* step / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        lr = args.lr * decayed

    elif args.lr_decay == "none":
        lr = args.lr 
    print("Current learning rate is ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def adjust_learning_rate_with_step(optimizer, epoch, args):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     rounds_per_epoch = args.num_data / (args.steps * args.num_users * args.batch_size * args.frac)


#     if epoch > 640*rounds_per_epoch:
#         lr = args.lr * 0.001
#     elif epoch > 512*rounds_per_epoch:
#         lr = args.lr * 0.01
#     elif epoch > 384*rounds_per_epoch:  
#         lr = args.lr * 0.1
#     else:
#         lr = args.lr   
#     print("Current learning rate is ", lr)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


        
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