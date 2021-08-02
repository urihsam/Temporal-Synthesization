import os
import json
import time
import torch
import argparse
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from dataset import EHR
from utils.aae_utils import train_model as train_aae
from utils.daae_utils import train_model as train_daae
from utils.dgat_utils import train_model as train_dgat
from utils.dgatt_utils import train_model as train_dgatt
from utils.vae_gan_utils import train_model as train_vae_gan
from utils.vae_utils import train_model as train_vae
from utils.vae_utils import train_seq2seq_model as train_seq2seq_vae



def main(args):
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)
    if args.test:
        assert args.test_model_filename != ""
    
    torch.cuda.set_device(args.gpu_devidx) 
    splits = ["train", "valid", "test"]

    datasets = OrderedDict()
    for split in splits:
        datasets[split] = EHR(
            data_dir=args.data_dir,
            split=split,
            max_length=args.max_length
        )
    
    if args.use_prob_mask:
        prob_mask = np.load(os.path.join(args.data_dir, args.prob_mask_filename))
    else:
        prob_mask = None

    infer_file_path = os.path.join(args.data_dir, args.struct_info_file)
    infer_info = np.load(infer_file_path, allow_pickle=True).item()
    max_year = infer_info["max"]["start_age_year"][0, 1]
    min_year = infer_info["min"]["start_age_year"][0, 1]
    time_shift = min_year
    time_scale = max_year - min_year

    if args.model_type == "daae":
        """ There are two GANs in daae, one is for output data x, another one is for hidden state z.
        """
        train_daae(args, datasets, prob_mask)
    elif args.model_type == "vae_gan":
        """ Only one GAN in vae_gan, which is for output data x, and z is constrained by KL divergence
        """
        train_vae_gan(args, datasets, prob_mask)
    elif args.model_type == "aae":
        """ Only one GAN in aae, which is for hidden state z, and x is only constrained by reconstruction loss
        """
        train_aae(args, datasets, prob_mask)
    elif args.model_type == "seq2seq_vae":
        """ No GAN in seq2seq_vae, z is constrained by KL divergence, and x is only constrained by reconstruction loss
        """
        train_seq2seq_vae(args, datasets, prob_mask)
    elif args.model_type == "vae":
        """ No GAN in vae, no seq2seq structure, only vanilla vae
        """
        train_vae(args, datasets, prob_mask)
    
    elif args.model_type == "dgat": # dual generative adversarial transformer
        """ There are two GANs in daae, one is for output data x, another one is for hidden state z.
        """
        train_dgat(args, datasets, prob_mask)

    elif args.model_type == "dgatt": # dual generative adversarial time-embedding transformer
        """ There are two GANs in daae, one is for output data x, another one is for hidden state z.
        """
        train_dgatt(args, datasets, prob_mask, time_shift=time_shift, time_scale=time_scale)
    
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, default='daae')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--pretrained_model_filename', type=str, default="")
    parser.add_argument('--test_model_filename', type=str, default="")
    parser.add_argument('--train_log', type=str, default='train_log_file')
    parser.add_argument('--valid_log', type=str, default='valid_log_file')
    parser.add_argument('--test_log', type=str, default='test_log_file')
    parser.add_argument('--model_path', type=str, default='models')
    parser.add_argument('--result_path', type=str, default='results')
    parser.add_argument('--struct_info_file', type=str, default='struct_info.npy')
    parser.add_argument('--max_length', type=int, default=40)
    parser.add_argument('--train_eval_freq', type=int, default=50)
    parser.add_argument('--valid_eval_freq', type=int, default=1)
    parser.add_argument('--critic_freq_base', type=int, default=5)
    parser.add_argument('--critic_freq_hit', type=int, default=1)
    parser.add_argument('--use_spectral_norm', type=bool, default=False)
    parser.add_argument('--no_mask', type=bool, default=False)
    parser.add_argument('--use_prob_mask', type=bool, default=False)
    parser.add_argument('--prob_mask_filename', type=str, default='not_nan_prob.npy')
    
    parser.add_argument('-ep', '--epochs', type=int, default=500)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-lr_decay', '--lr_decay_rate', type=float, default=0.99)
    parser.add_argument('-beta_r', '--beta_recon', type=float, default=10.0)
    parser.add_argument('-beta_m', '--beta_mask', type=float, default=1.0)
    parser.add_argument('-beta_k', '--beta_kld', type=float, default=1.0)
    parser.add_argument('-gs','--gendata_size', type=int, default=100000)
    parser.add_argument('-gd', '--gpu_devidx', type=int, default=0)

    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('-fts', '--feature_size', type=int, default=9)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=128)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ns', '--noise_size', type=int, default=128)
    parser.add_argument('-ls', '--latent_size', type=int, default=128)
    parser.add_argument('-fis', '--filter_size', type=int, default=16)
    parser.add_argument('-ws', '--window_sizes', nargs='+', type=int, default=[2, 3])
    parser.add_argument('-ed', '--encoder_dropout', type=float, default=0.5)
    parser.add_argument('-dd', '--decoder_dropout', type=float, default=0.5)
    parser.add_argument('-fd', '--feature_dropout', type=float, default=0.5)
    parser.add_argument('-ga', '--gmlp_archs', nargs='+', type=int, default=[128, 128])
    parser.add_argument('-da', '--dmlp_archs', nargs='+', type=int, default=[256, 128])

    parser.add_argument('--dp_sgd', type=bool, default=False)
    parser.add_argument('--noise_multiplier', type=float, default=1)
    parser.add_argument('--l2_norm_clip', type=float, default=0.5)
    parser.add_argument('--delta', type=float, default=1e-3)

    args = parser.parse_args()
    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']

    main(args)
