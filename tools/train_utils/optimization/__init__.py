from functools import partial

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle, OneCycle_with_WarmupLR
import matplotlib.pyplot as plt

def build_optimizer(model, optim_cfg):
    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=optim_cfg.LR,
            weight_decay=optim_cfg.WEIGHT_DECAY,
        )
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=optim_cfg.LR,
            weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM,
        )
    elif optim_cfg.OPTIMIZER == 'adamW_with_WarmupLR_cosdecay':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=optim_cfg.LR,
            betas=(optim_cfg.BETA1, optim_cfg.BETA2),
            weight_decay=optim_cfg.WEIGHT_DECAY,
            eps=optim_cfg.EPS,
        )
    # 如果采用adam_onecycle优化器
    elif optim_cfg.OPTIMIZER == 'adam_onecycle':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    # 如果采用adam_onecycle优化器
    elif optim_cfg.OPTIMIZER == 'OneCycle_with_WarmupLR':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    else:
        raise NotImplementedError

    return optimizer

def linear_warmup_with_cosdecay(cur_step, warmup_steps, total_steps, min_scale=1e-5):
    if cur_step < warmup_steps:
        return (1 - min_scale) * cur_step / warmup_steps + min_scale
    else:
        ratio = (cur_step - warmup_steps) / total_steps
        return (1 - min_scale) * 0.5 * (1 + np.cos(np.pi * ratio)) + min_scale

def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST] # [5, 10, 15, 20, 25, 30]
    def lr_lbmd(cur_epoch):
        cur_decay = 1 # 当前学习率设置为1
        for decay_step in decay_steps: 
            if cur_epoch >= decay_step: # 如果当前的epoch数>=节点值
                cur_decay = cur_decay * optim_cfg.LR_DECAY # 更新学习率
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR) # 防止学习率过小 LR_CLIP: 0.0000001

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs
    # 构建adam_onecycle学习率调度器
    if optim_cfg.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        ) # LR: 0.003, MOMS: [0.95, 0.85],DIV_FACTOR: 10, PCT_START: 0.4 
    elif optim_cfg.OPTIMIZER == 'OneCycle_with_WarmupLR':
        lr_scheduler = OneCycle_with_WarmupLR(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        ) # LR: 0.003, MOMS: [0.95, 0.85],DIV_FACTOR: 10, PCT_START: 0.4 
    
    elif optim_cfg.OPTIMIZER == 'adamW_with_WarmupLR_cosdecay':
        warmup_steps = optim_cfg.WARMUP_EPOCH * total_iters_each_epoch
        total_steps = total_epochs * total_iters_each_epoch
        lr_scheduler = lr_sched.LambdaLR(
            optimizer,
            lr_lambda = lambda x: linear_warmup_with_cosdecay(x, warmup_steps, total_steps),
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)
        # 热身：在刚刚开始训练时以很小的学习率进行训练，使得网络熟悉数据，随着训练的进行学习率慢慢变大，
        # 到了一定程度，以设置的初始学习率进行训练，接着过了一些inter后，学习率再慢慢变小；学习率变化：上升——平稳——下降；
        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )
    # 绘图 ####################################################################
    # # print("lr_warmup_scheduler is",lr_warmup_scheduler)
    lrs = []
    moms = []
    for i in range(total_steps):
        lr_scheduler.step(i)
        lrs.append(optimizer.lr)
        moms.append(optimizer.mom)
    plt.plot(lrs)
    plt.savefig("/home/Projects/MASS/tools/plot.png")  # 保存图片到当前目录
    ##########################################################################

    return lr_scheduler, lr_warmup_scheduler
