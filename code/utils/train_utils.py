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


def model_inference(args, decoder, zgen, prob_mask, **kwargs):
    # make up start feature
    start_feature, start_time, start_mask = sample_start_feature_time_mask(zgen.size(0))
    model_list = ["dgatt", "dgamt", "edgamt", "tgamt", "etgamt"]
    if args.model_type == "dgatt":
        kwargs["start_time"] = start_time
    elif args.model_type in model_list:
        kwargs["start_time"] = start_time
        sampled_gender, sampled_race = sample_gender_race(zgen.size(0))
        kwargs["gender"] = sampled_gender
        kwargs["race"] = sampled_race

    if args.model_type in model_list:
        if args.no_mask:
            Pgen, Tgen, Mgen = decoder.inference(start_feature=start_feature, start_mask=None, z=zgen, **kwargs)
        elif args.use_prob_mask:
            Pgen, Tgen, Mgen = decoder.inference(start_feature=start_feature, start_mask=start_mask, prob_mask=prob_mask, z=zgen, **kwargs)
        else:
            Pgen, Tgen, Mgen = decoder.inference(start_feature=start_feature, start_mask=start_mask, z=zgen, **kwargs)
    else:
        if args.no_mask:
            Pgen, Mgen = decoder.inference(start_feature=start_feature, start_mask=None, z=zgen, **kwargs)
        elif args.use_prob_mask:
            Pgen, Mgen = decoder.inference(start_feature=start_feature, start_mask=start_mask, prob_mask=prob_mask, z=zgen, **kwargs)
        else:
            Pgen, Mgen = decoder.inference(start_feature=start_feature, start_mask=start_mask, z=zgen, **kwargs)

    return Pgen, Mgen


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def sample_gender_race(batch_size):
    gender = torch.randint(0, 2, (batch_size, 1))
    race = torch.randint(0, 3, (batch_size, 1))
    return gender.int(), race.int()


def sample_start_feature_mask(batch_size):
    start_feature, _, start_mask = sample_start_feature_time_mask(batch_size)

    return start_feature, start_mask


def sample_start_feature_time_mask(batch_size):
    padding = torch.zeros(batch_size, 1, dtype=torch.float)
    age = torch.tensor(np.random.uniform(size=(batch_size, 1))*0.9, dtype=torch.float)
    year = torch.tensor(np.random.uniform(size=(batch_size, 1))*0.9, dtype=torch.float)
    start_feature = torch.cat((age, year, age, year, age, year, age, year, padding), 1)
    start_mask = torch.tensor(np.tile(np.expand_dims(np.array([1]*8+[0]), 0), [batch_size, 1]))

    return start_feature, year, start_mask


def extract_time_from_start_feature(start_feature):
    assert len(start_feature.shape) == 2
    return start_feature[:, 3] # [batch_size, 1]


def extract_incr_time_from_tempo_step(temporal_step_feature):
    assert len(temporal_step_feature.shape) == 2
    return temporal_step_feature[:, -1].unsqueeze(dim=1)  # [batch_size, 1]

def descale_time(scaled_time, shift, scale):
    return torch.floor(scaled_time * scale + shift).int()


def sample_mask_from_prob(prob_mask, batch_size, steps):
    prob_mask = torch.tensor(prob_mask, dtype=torch.float).cuda()
    prob_mask = torch.squeeze(prob_mask)
    prob_mask = torch.unsqueeze(prob_mask, 0); prob_mask = torch.unsqueeze(prob_mask, 0)
    prob = torch.tile(prob_mask, (batch_size, steps, 1))
    return torch.bernoulli(prob)
