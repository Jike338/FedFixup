import pdb
import os
import glob

import numpy as np
from PIL import Image
from enum import Enum
import random
import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from fedlab.utils.dataset.partition import CIFAR10Partitioner
import torch.nn.functional as F
from math import pi

import torch
import torch.nn as nn

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

def disable_bn_layers(net):
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.requires_grad_(False)
            module.eval() 
            
def enable_bn_layers(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(True)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.requires_grad_(True)
            module.train()
            
def disable_model_layers(model):
    for param in model.parameters():
        param.requires_grad = False   
        
def enable_model_layers(model):
    for param in model.parameters():
        param.requires_grad = True  
        
def disable_fc(model):
    for param in model.fc.parameters():
        param.requires_grad = False   
        
def enable_fc(model):
    for param in model.fc.parameters():
        param.requires_grad = True
        

            
def disable_feature_layers(net):
    for module in net.modules():
        if not (isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.linear)):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.requires_grad_(False)
            module.eval()
            
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
        
    
  

def imagenet_noniid(dataset, num_users, tiny=True, num_data=20000, method = "dir", alpha=0.1):
    """
    Sample non-I.I.D client data from ImageNet dataset
    :param dataset:
    :param num_users:
    :return:
    """
    min_size = 0
    class_num = 200 if tiny else 1000
    _lst_sample = 0
    
    data_size = len(dataset)
    labels = np.array(dataset.targets)

    if method=="step":
      
      num_shards = num_users*2
      num_imgs = data_size// num_shards
      idx_shard = [i for i in range(num_shards)]
      
      idxs = np.arange(num_shards*num_imgs)

      #alien labels with idxs
      labels = labels[0:len(idxs)]
      
      # sort labels
      idxs_labels = np.vstack((idxs, labels))
      idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
      idxs = idxs_labels[0,:]
      
      least_idx = np.zeros((num_users, 200, _lst_sample), dtype=np.int)
      for i in range(200):
        idx_i = np.random.choice(np.where(labels==i)[0], num_users*_lst_sample, replace=False)
        least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))
      least_idx = np.reshape(least_idx, (num_users, -1))
      
      least_idx_set = set(np.reshape(least_idx, (-1)))
      server_idx = np.random.choice(list(set(range(data_size))-least_idx_set), data_size-num_data, replace=False)
      
      train_ids = []
      # divide and assign
      dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
      for i in range(num_users):
          rand_set = set(np.random.choice(idx_shard, num_shards//num_users, replace=False))
          idx_shard = list(set(idx_shard) - rand_set)
          for rand in rand_set:
              idx_i = list( set(range(rand*num_imgs, (rand+1)*num_imgs))   )
              add_idx = list(set(idxs[idx_i]) - set(server_idx)  )              
              train_ids+=add_idx

              dict_users[i] = np.concatenate((dict_users[i], add_idx), axis=0)
          dict_users[i] = np.concatenate((dict_users[i], least_idx[i]), axis=0)
          train_ids+=list(least_idx[i])

      train_ids = set()
      for u,v in dict_users.items():
        train_ids.update(v)
      train_ids = list(train_ids) 
    
    elif method == "dir":
        server_idx = np.random.choice(np.arange(data_size), data_size-num_data, replace=False)
        server_idx_set = set(server_idx.tolist())
        local_idx = set(range(data_size)) - server_idx_set
        N = labels.shape[0]
        net_dataidx_map = {}
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

        #while min_size < 20:
        idx_batch = [[] for _ in range(num_users)]
        # for each class in the dataset
        for k in range(class_num):
            idx_k = np.where(labels == k)[0]
            idx_k = list(set(idx_k) - server_idx_set)
            
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            proportions = np.array([p*(len(idx_j)<N/num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            #min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_users):
            np.random.shuffle(idx_batch[j])
            dict_users[j] = idx_batch[j]      

    cnts_dict = {}
    with open("%s_%d_u%d_%.1f.txt"%("Tiny" if tiny else "ImageNet", num_data, num_users, alpha), 'w') as f:
      all_class_cnts = np.array([0]*class_num)
      f.write("Centralized size %d\n"%data_size)
      for i in range(num_users):
        labels_i = labels[dict_users[i]]
        cnts = np.array([np.count_nonzero(labels_i == j) for j in range(class_num)] )
        cnts_dict[i] = cnts
        all_class_cnts += cnts
        f.write("User %s: %s sum: %d\n"%(i, " ".join([str(cnt) for cnt in cnts]), sum(cnts) ))  
      f.write("In total: %s sum: %d\n"%(" ".join([str(cnt) for cnt in all_class_cnts]), sum(all_class_cnts)))
      
    return dict_users, server_idx, cnts_dict        
    
    
def inat_geo_noniid(split, dataset, num_data=1203*5):
    """
    Sample non-I.I.D client data from Inat GEO dataset
    :param dataset:
    :param num_users:
    :return:
    """
    min_size = 0
    class_num = 1203
    
    data_size = len(dataset)
    labels = np.array(dataset.targets)


    server_idx = []
    for cls in range(class_num):
        server_idx.append(np.random.choice(np.where(labels==cls)[0], num_data//class_num, replace=False)) 
    server_idx = np.concatenate(server_idx)  
    
    dict_users = {}
    for i, (path, _) in enumerate(dataset.samples):
        user_id = path.split("/")[-2]
        if i in server_idx: continue
        if user_id not in dict_users:
            dict_users[user_id] = [int(i)]
        else:
            dict_users[user_id].append(int(i))

    for j in dict_users.keys():
        np.random.shuffle(dict_users[j])    
    
    num_users = len(dict_users)
    cnts_dict = {}
    with open("%s_%d_u%d.txt"%(split, num_data, num_users), 'w') as f:
      all_class_cnts = np.array([0]*class_num)
      f.write("Centralized size %d\n"%data_size)
      for i in dict_users.keys():
        labels_i = labels[dict_users[i]]
        cnts = np.array([np.count_nonzero(labels_i == j) for j in range(class_num)] )
        cnts_dict[i] = cnts
        all_class_cnts += cnts
        f.write("User %s: %s sum: %d\n"%(i, " ".join([str(cnt) for cnt in cnts]), sum(cnts) ))   
      f.write("\nIn total: %s sum: %d\n"%(" ".join([str(cnt) for cnt in all_class_cnts]), sum(all_class_cnts)))
      
    return dict_users, server_idx, cnts_dict    

def cifar100_noniid(dataset, num_users, num_data=50000, method="dir", alpha=0.1):

    """
    Sample non-I.I.D client data from CIFAR dataset
    :param dataset:
    :param num_users:
    :return:
    """
    
    labels = np.array(dataset.targets)
    _lst_sample = 0 #if num_users > 10 else 10 
    
    if method=="step":
      
      num_shards = num_users*2
      num_imgs = 50000// num_shards
      idx_shard = [i for i in range(num_shards)]
      
      idxs = np.arange(num_shards*num_imgs)
      # sort labels
      idxs_labels = np.vstack((idxs, labels))
      idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
      idxs = idxs_labels[0,:]
      
      least_idx = np.zeros((num_users, 100, _lst_sample), dtype=np.int)
      for i in range(100):
        idx_i = np.random.choice(np.where(labels==i)[0], num_users*_lst_sample, replace=False)
        least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))
      least_idx = np.reshape(least_idx, (num_users, -1))
      
      least_idx_set = set(np.reshape(least_idx, (-1)))
      server_idx = np.random.choice(list(set(range(50000))-least_idx_set), 50000-num_data, replace=False)
      
      train_ids = []
      # divide and assign
      dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
      for i in range(num_users):
          rand_set = set(np.random.choice(idx_shard, num_shards//num_users, replace=False))
          idx_shard = list(set(idx_shard) - rand_set)
          for rand in rand_set:
              idx_i = list( set(range(rand*num_imgs, (rand+1)*num_imgs))   )
              add_idx = list(set(idxs[idx_i]) - set(server_idx)  )              
              train_ids+=add_idx

              dict_users[i] = np.concatenate((dict_users[i], add_idx), axis=0)
          dict_users[i] = np.concatenate((dict_users[i], least_idx[i]), axis=0)
          train_ids+=list(least_idx[i])

      train_ids = set()
      for u,v in dict_users.items():
        train_ids.update(v)
      train_ids = list(train_ids) 
        
    elif method == "dir":
      min_size = 0
      K = 100
      y_train = labels
      
      _lst_sample = 0
      
      
      least_idx = np.zeros((num_users, 100, _lst_sample), dtype=np.int)
      for i in range(100):
        idx_i = np.random.choice(np.where(labels==i)[0], num_users*_lst_sample, replace=False)
        least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))
      least_idx = np.reshape(least_idx, (num_users, -1))
      
      least_idx_set = set(np.reshape(least_idx, (-1)))
      #least_idx_set = set([])
      server_idx = np.random.choice(list(set(range(50000))-least_idx_set), 50000-num_data, replace=False)
      local_idx = np.array([i for i in range(50000) if i not in server_idx and i not in least_idx_set])
      
      N = y_train.shape[0]
      net_dataidx_map = {}
      dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
      sum_size = 0
      while (50000-num_data) - sum_size > 100:
          idx_batch = [[] for _ in range(num_users)]
          # for each class in the dataset
          for k in range(K):
              idx_k = np.where(y_train == k)[0]
              idx_k = [id for id in idx_k if id not in server_idx]
              
              np.random.shuffle(idx_k)
              proportions = np.random.dirichlet(np.repeat(alpha, num_users))
              ## Balance
              proportions = np.array([p*(len(idx_j)<N/num_users) for p,idx_j in zip(proportions,idx_batch)])
              proportions = proportions/proportions.sum()
              proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
              idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
              sum_size = np.sum([len(idx_j) for idx_j in idx_batch])

      for j in range(num_users):
          np.random.shuffle(idx_batch[j])
          dict_users[j] = idx_batch[j]  
          dict_users[j] = np.concatenate((dict_users[j], least_idx[j]), axis=0)          
    
    cnts_dict = {}
    with open("cifar100_%d_u%d_%s.txt"%(num_data, num_users, method), 'w') as f:
      for i in range(num_users):
        labels_i = labels[dict_users[i]]
        cnts = np.array([np.count_nonzero(labels_i == j ) for j in range(100)] )
        cnts_dict[i] = cnts
        f.write("User %s: %s sum: %d\n"%(i, " ".join([str(cnt) for cnt in cnts]), sum(cnts) ))  
   
    return dict_users, server_idx, cnts_dict

def cifar_noniid(dataset, num_users, num_data=50000, method="dir", alpha=0.1):

    """
    Sample non-I.I.D client data from CIFAR dataset
    :param dataset:
    :param num_users:
    :return:
    """
    
    labels = np.array(dataset.targets)
    _lst_sample = 0 #if num_users > 10 else 10 
    
    if method=="step":
      
      num_shards = num_users*2
      num_imgs = 50000// num_shards
      idx_shard = [i for i in range(num_shards)]
      
      idxs = np.arange(num_shards*num_imgs)
      # sort labels
      idxs_labels = np.vstack((idxs, labels))
      idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
      idxs = idxs_labels[0,:]
      
      least_idx = np.zeros((num_users, 10, _lst_sample), dtype=np.int)
      for i in range(10):
        idx_i = np.random.choice(np.where(labels==i)[0], num_users*_lst_sample, replace=False)
        least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))
      least_idx = np.reshape(least_idx, (num_users, -1))
      
      least_idx_set = set(np.reshape(least_idx, (-1)))
      server_idx = np.random.choice(list(set(range(50000))-least_idx_set), 50000-num_data, replace=False)
      
      train_ids = []
      # divide and assign
      dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
      for i in range(num_users):
          rand_set = set(np.random.choice(idx_shard, num_shards//num_users, replace=False))
          idx_shard = list(set(idx_shard) - rand_set)
          for rand in rand_set:
              idx_i = list( set(range(rand*num_imgs, (rand+1)*num_imgs))   )
              add_idx = list(set(idxs[idx_i]) - set(server_idx)  )              
              train_ids+=add_idx

              dict_users[i] = np.concatenate((dict_users[i], add_idx), axis=0)
          dict_users[i] = np.concatenate((dict_users[i], least_idx[i]), axis=0)
          train_ids+=list(least_idx[i])

      train_ids = set()
      for u,v in dict_users.items():
        train_ids.update(v)
      train_ids = list(train_ids) 
        
    elif method == "dir":
      min_size = 0
      K = 10
      y_train = labels
      
      _lst_sample = 0
      
      
      least_idx = np.zeros((num_users, 10, _lst_sample), dtype=np.int)
      for i in range(10):
        idx_i = np.random.choice(np.where(labels==i)[0], num_users*_lst_sample, replace=False)
        least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))
      least_idx = np.reshape(least_idx, (num_users, -1))
      
      least_idx_set = set(np.reshape(least_idx, (-1)))
      #least_idx_set = set([])
      server_idx = np.random.choice(list(set(range(50000))-least_idx_set), 50000-num_data, replace=False)
      local_idx = np.array([i for i in range(50000) if i not in server_idx and i not in least_idx_set])
      
      N = y_train.shape[0]
      net_dataidx_map = {}
      dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
      sum_size = 0
      while num_data - sum_size > 100:
          idx_batch = [[] for _ in range(num_users)]
          # for each class in the dataset
          for k in range(K):
              idx_k = np.where(y_train == k)[0]
              idx_k = [id for id in idx_k if id not in server_idx]
              
              np.random.shuffle(idx_k)
              proportions = np.random.dirichlet(np.repeat(alpha, num_users))
              ## Balance
              proportions = np.array([p*(len(idx_j)<N/num_users) for p,idx_j in zip(proportions,idx_batch)])
              proportions = proportions/proportions.sum()
              proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
              idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
              sum_size = np.sum([len(idx_j) for idx_j in idx_batch])

      for j in range(num_users):
          np.random.shuffle(idx_batch[j])
          dict_users[j] = idx_batch[j]  
          dict_users[j] = np.concatenate((dict_users[j], least_idx[j]), axis=0)          
    
    cnts_dict = {}
    with open("data_%d_u%d_%s.txt"%(num_data, num_users, method), 'w') as f:
      for i in range(num_users):
        labels_i = labels[dict_users[i]]
        cnts = np.array([np.count_nonzero(labels_i == j ) for j in range(10)] )
        cnts_dict[i] = cnts
        f.write("User %s: %s sum: %d\n"%(i, " ".join([str(cnt) for cnt in cnts]), sum(cnts) ))  
   
    return dict_users, server_idx, cnts_dict
    
def inference(train_loader, criterion, model, args):
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    # switch to eval mode
    model.eval()
    
    with torch.no_grad():
        device  = torch.device("cuda")  
        model.to(device)
        activations = None
        imgs = []
        total_loss = None
        for i, (images, target) in enumerate(train_loader):
            images = images.to(device)
            target = target.to(device)
            # compute output
            output_ = model(images)  
            loss = criterion(output_, target)
            losses.update(loss.item(), images.size(0))
         
            for mod in model.modules():
                if(isinstance(mod, Activs_prober)):
                    output = mod.activs
                    break
            
            # output = torch.flatten(output, start_dim=1, end_dim=- 1)
            activations = output if activations is None else torch.cat((activations, output), axis=0)
        quantile = torch.quantile(activations, 0.95, dim=1)
        quantile_mean = torch.mean(quantile, axis=0).cpu().numpy()

        activs_var = torch.var(activations, axis = 1)
        activs_mean = torch.mean(activations, axis = 1)

        tr = torch.diag(activations).sum()
        opnom = torch.linalg.norm(activations, ord=2)
        activs_rank = (tr / opnom).item()

    return activs_mean, activs_var, activs_rank, quantile_mean, losses.avg
  
def inference_single(train_loader, model, args):

    # switch to eval mode
    model.eval()
    with torch.no_grad():
        device  = torch.device("cuda")  
        model.to(device)
        activations = None
        imgs = []
        for i, (images, target) in enumerate(train_loader):
            images = images.to(device)

            # compute output
            model(images)  

            for mod in model.modules():
                if(isinstance(mod, Activs_prober)):
                    output = mod.activs
                    break

            return output[0]


def update_bn(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.
    Example:
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)
    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    device  = torch.device("cuda") 
    was_training = model.training
    model.train()
    model.to(device)
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        if device is not None:
            input = input.to(device)

        model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)

def update_bn_with_step_w_mem(loader, model, args, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.
    Example:
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)
    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """

    device  = torch.device("cuda") 
    was_training = model.training
    model.train()
    model.to(device)
    train_loader_iter = iter(loader)
    n=0
    while n < args.bn_step:

        try:
            # Samples the batch
            images, target = next(train_loader_iter)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            train_loader_iter = iter(loader)
            images, target = next(train_loader_iter)
        images = images.to("cuda")
        model(images)
        n+=1
    model.train(was_training)

def update_bn_with_step(loader, model, args, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.
    Example:
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)
    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    device  = torch.device("cuda") 
    was_training = model.training
    model.train()
    model.to(device)
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    train_loader_iter = iter(loader)
    n = 0
    while n < args.bn_step:
            
        try:
            # Samples the batch
            images, target = next(train_loader_iter)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            train_loader_iter = iter(loader)
            images, target = next(train_loader_iter)
        images = images.to("cuda")
        model(images)
        n+=1

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)

### Activs_prober (i.e., probes activations) ###
class Activs_prober(nn.Module):
    def __init__(self):
        super(Activs_prober, self).__init__()
        # Activs

        self.activs = None

        class sim_activs(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                self.activs = input.clone()
                return input.clone()

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output.clone()
            
        self.cal_prop = sim_activs.apply

    def forward(self, input):
        if torch.is_grad_enabled():
            return input
        else:
            return self.cal_prop(input)

class GroupNorm2d(nn.Module):
    def __init__(self, num_channels):
        super(GroupNorm2d, self).__init__()
        self.norm = nn.GroupNorm(num_groups=2, num_channels=num_channels)
    def forward(self, x):
        x = self.norm(x)
        return x      

def create_activs_probe_path(args):
    output_dir = args.id.split("/")
    mean_dir = output_dir[0]+"/activs_log/MEAN/" 
    var_dir = output_dir[0]+"/activs_log/VAR/"
    rank_dir = output_dir[0]+"/activs_log/RANK/"
    single_dir = output_dir[0]+"/activs_log/SINGLE/"
    
    os.makedirs(mean_dir, exist_ok=True)
    os.makedirs(var_dir, exist_ok=True)
    os.makedirs(rank_dir, exist_ok=True)
    os.makedirs(single_dir, exist_ok=True)

    mean_file = mean_dir +"/"+ output_dir[2]+".txt"
    var_file = var_dir +"/"+ output_dir[2]+".txt"
    rank_file = rank_dir +"/"+ output_dir[2]+".txt"
    single_file = single_dir +"/"+ output_dir[2]+".txt"

    return mean_file, var_file, rank_file, single_file


def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))


def seed_everything(seed=2023):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


        
def adjust_learning_rate_with_step(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    rounds_per_epoch = args.num_data / (args.steps * args.num_users * args.batch_size * args.frac)


    if epoch > 128*rounds_per_epoch:
        lr = args.lr * 0.001
    elif epoch > 96*rounds_per_epoch:
        lr = args.lr * 0.01
    elif epoch > 64*rounds_per_epoch:  
        lr = args.lr * 0.1
    else:
        lr = args.lr   
    print("Current learning rate is ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



# def adjust_learning_rate_with_step(optimizer, epoch, args):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     rounds_per_epoch = args.num_data / (args.steps * args.num_users * args.batch_size * args.frac)


#     if epoch > 1*(args.rounds):
#         lr = args.lr * 0.001
#     elif epoch > 0.8*(args.rounds):
#         lr = args.lr * 0.01
#     elif epoch > 0.6*(args.rounds):  
#         lr = args.lr * 0.1
#     else:
#         lr = args.lr   
#     print("Current learning rate is ", lr)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


### Weight Normalization (Salimans and Kingma, 2016) ###
class WN_self(nn.Conv2d):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, eps=1e-5):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.gamma = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.eps = eps

    def get_weight(self):
        denom = torch.linalg.norm(self.weight, dim=[1, 2, 3], keepdim=True)
        weight = self.weight / (denom + self.eps)
        return self.gamma * weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)

# Scaled activation function for WeightNorm (Performs scaled/bias correction; Arpit et al., 2016)
class WN_scaledReLU(nn.ReLU):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return (2 * pi / (pi - 1))**0.5 * (F.relu(x, inplace=self.inplace) - (1 / (2 * pi))**(0.5))


def disable_bn_layers(net):
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.requires_grad_(False)
            module.train()