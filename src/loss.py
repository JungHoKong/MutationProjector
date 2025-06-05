## 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import *
import numpy as np
import os
home_dir = os.path.expanduser('~')


# multitask loss
def calc_multi_task_loss(pred, labels, loss_fn):
    total_loss = 0.
    for i in range(pred.shape[1]):
        loss = loss_fn(pred[:,i], labels[:,i])
        total_loss += loss
    return total_loss

