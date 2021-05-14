import os
import json
import time
import torch
import argparse
import numpy as np
from torch.autograd import Variable
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def sample_start_feature_and_mask(batch_size):
    padding = torch.zeros(batch_size, 2, dtype=torch.float)
    age = torch.tensor(np.random.randint(40, 60, size=(batch_size, 1)), dtype=torch.float)
    year = torch.tensor(np.random.rand(batch_size, 1) * 10 + 38, dtype=torch.float)
    gender = torch.nn.functional.one_hot(torch.tensor(np.random.randint(0, 2, size=batch_size)), num_classes=2).float()
    race = torch.nn.functional.one_hot(torch.tensor(np.random.randint(0, 3, size=batch_size)), num_classes=3).float()
    start_feature = torch.cat((padding, age, year, gender, race), 1)
    start_mask = torch.tensor(np.tile(np.expand_dims(np.array([0,0]+[1]*7), 0), [batch_size, 1]))

    return start_feature, start_mask


def sample_mask_from_prob(prob_mask, batch_size, steps):
    prob_mask = torch.tensor(prob_mask, dtype=torch.float).cuda()
    prob_mask = torch.squeeze(prob_mask)
    prob_mask = torch.unsqueeze(prob_mask, 0); prob_mask = torch.unsqueeze(prob_mask, 0)
    prob = torch.tile(prob_mask, (batch_size, steps, 1))
    return torch.bernoulli(prob)
