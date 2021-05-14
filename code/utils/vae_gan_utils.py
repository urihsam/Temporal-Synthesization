import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from utils.train_utils import to_var, sample_start_feature_and_mask, sample_mask_from_prob


def save_model(models, path):
    AE = models["AE"]
    Dx = models["Dx"]

    torch.save(AE, "{}_AE".format(path))
    torch.save(Dx, "{}_Dx".format(path))


def load_model(path):
    AE = torch.load("{}_AE".format(path))
    Dx = torch.load("{}_Dx".format(path))

    models = {
        "AE": AE,
        "Dx": Dx
    }
    return models


def model_evaluation(args, models, opts, lrs, data_loader, prob_mask, split, log_file):
    AE = models["AE"]
    Dx = models["Dx"]
    if split == 'train':
        # opts
        opt_enc = opts["enc"]
        opt_dec = opts["dec"]
        opt_dix = opts["dix"]
        # lr scheduler
        lr_enc = lrs["enc"]
        lr_dec = lrs["dec"]
        lr_dix = lrs["dix"]

    # init
    recon_total_loss, mask_total_loss, kld_total_loss = 0.0, 0.0, 0.0
    xCritic_total_loss = 0.0
    
    n_data = 0

    if split == 'train':
        AE.encoder_dropout=args.encoder_dropout
        AE.decoder_dropout=args.decoder_dropout
        AE.train()
        Dx.train()
    else:
        AE.encoder_dropout=0.0
        AE.decoder_dropout=0.0
        AE.eval()
        Dx.eval()

    for iteration, batch in enumerate(data_loader):
        batch_size = batch['tempo'].shape[0]
        n_data += batch_size
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = to_var(v)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1

        if torch.cuda.is_available():
            one = one.cuda()
            mone = mone.cuda()
        
        #import pdb; pdb.set_trace()
        # Step 0: Evaluate current loss
        if args.no_mask:
            mu, log_var, Pinput, Poutput, Moutput = AE(batch['tempo'], batch['target'], None, None)
            # loss
            recon_loss = args.beta_recon * AE.compute_recon_loss(Poutput, batch['target'], None, None)
        elif args.use_prob_mask:
            mu, log_var, Pinput, Poutput, Moutput = AE(batch['tempo'], batch['target'], batch["mask"], batch["target_mask"])
            output_mask = sample_mask_from_prob(prob_mask, batch["target_mask"].shape[0], batch["target_mask"].shape[1])
            # loss
            recon_loss = args.beta_recon * AE.compute_recon_loss(Poutput, batch['target'], output_mask, batch["target_mask"])
        else:
            mu, log_var, Pinput, Poutput, Moutput = AE(batch['tempo'], batch['target'], batch["mask"], batch["target_mask"])
            # loss
            recon_loss = args.beta_recon * AE.compute_recon_loss(Poutput, batch['target'],  Moutput, batch["target_mask"])
            mask_loss = args.beta_mask * AE.compute_mask_loss(Moutput, batch["target_mask"])

        kld_loss = args.beta_kld * AE.compute_kl_diver_loss(mu, log_var)

        # samples from N(0, I)
        zgen = torch.randn(log_var.size())
        # make up start feature
        start_feature, start_mask = sample_start_feature_and_mask(zgen.size(0))
        if args.no_mask:
            Pgen, Mgen = AE.decoder.inference(start_feature=start_feature, start_mask=None, z=zgen)
        elif args.use_prob_mask:
            Pgen, Mgen = AE.decoder.inference(start_feature=start_feature, start_mask=start_mask, prob_mask=prob_mask, z=zgen)
        else:
            Pgen, Mgen = AE.decoder.inference(start_feature=start_feature, start_mask=start_mask, z=zgen)

        Dinput, Doutput, Dgen = Dx(Pinput).mean(), Dx(Poutput, Moutput).mean(), Dx(Pgen, Mgen).mean()

        xCritic_loss = - Dinput + 0.5 * (Doutput + Dgen)
        
        if split == 'train':
            if iteration % args.critic_freq_base < args.critic_freq_hit:
                # Step 1: Update the Critic_x
                opt_dix.zero_grad()
                Dinput, Doutput = Dx(Pinput).mean(), Dx(Poutput, Moutput).mean()
                Dinput.backward(mone, retain_graph=True)
                Doutput.backward(one, retain_graph=True)
                Dx.cal_gradient_penalty(Pinput[:, :Poutput.size(1), :], Poutput, Moutput).backward(retain_graph=True)
                opt_dix.step()

                opt_dix.zero_grad()
                Dinput, Dgen = Dx(Pinput).mean(), Dx(Pgen, Mgen).mean()
                Dinput.backward(mone, retain_graph=True)
                Dgen.backward(one, retain_graph=True)
                Dx.cal_gradient_penalty(Pinput[:, :Pgen.size(1), :], Pgen, Mgen).backward(retain_graph=True)
                opt_dix.step()


            # Step 3, 4: Update the Decoder and the Encoder
            opt_dec.zero_grad()
            Doutput, Dgen = Dx(Poutput, Moutput).mean(), Dx(Pgen, Mgen).mean()
            Doutput.backward(mone, retain_graph=True)
            Dgen.backward(mone, retain_graph=True)
            
            opt_enc.zero_grad()
            recon_loss.backward(retain_graph=True)
            if not args.no_mask and not args.use_prob_mask:
                mask_loss.backward(retain_graph=True)
            kld_loss.backward(retain_graph=True)
            opt_dec.step()
            opt_enc.step()

        
        #import pdb; pdb.set_trace()
        #
        recon_total_loss += recon_loss.data
        if not args.no_mask and not args.use_prob_mask:
            mask_total_loss += mask_loss.data
        else:
            mask_total_loss = 0.0
            mask_loss = 0.0
        kld_total_loss += kld_loss.data
        xCritic_total_loss += xCritic_loss.data

        if split == 'train' and iteration % args.train_eval_freq == 0:
            # print the losses for each epoch
            print("Learning rate:\t%2.8f"%(lr_enc.get_last_lr()[0]))
            print("Batch loss:")
            print("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\tkld_loss\t%9.4f\txCritic_loss\t%9.4f"%(split.upper(), recon_loss/batch_size, mask_loss/batch_size, kld_loss/batch_size, xCritic_loss/batch_size))
            print("Accumulated loss:")
            print("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\tkld_loss\t%9.4f\txCritic_loss\t%9.4f"%(split.upper(), recon_total_loss/n_data, mask_total_loss/n_data, kld_total_loss/n_data, xCritic_total_loss/n_data))
            print()
            with open(log_file, "a+") as file:
                file.write("Learning rate:\t%2.8f\n"%(lr_enc.get_last_lr()[0]))
                file.write("Batch loss:\n")
                file.write("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\tkld_loss\t%9.4f\txCritic_loss\t%9.4f\n"%(split.upper(), recon_loss/batch_size, mask_loss/batch_size, kld_loss/batch_size, xCritic_loss/batch_size))
                file.write("Accumulated loss:\n")
                file.write("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\tkld_loss\t%9.4f\txCritic_loss\t%9.4f\n"%(split.upper(), recon_total_loss/n_data, mask_total_loss/n_data, kld_total_loss/n_data, xCritic_total_loss/n_data))
                file.write("===================================================\n")
    #
    # print the losses for each epoch
    if split == 'train':
        print("Learning rate:\t%2.8f"%(lr_enc.get_last_lr()[0]))
    print("Batch loss:")
    print("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\tkld_loss\t%9.4f\txCritic_loss\t%9.4f"%(split.upper(), recon_loss/batch_size, mask_loss/batch_size, kld_loss/batch_size, xCritic_loss/batch_size))
    print("Accumulated loss:")
    print("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\tkld_loss\t%9.4f\txCritic_loss\t%9.4f"%(split.upper(), recon_total_loss/n_data, mask_total_loss/n_data, kld_total_loss/n_data, xCritic_total_loss/n_data))
    print()
    with open(log_file, "a+") as file:
        if split == 'train':
            file.write("Learning rate:\t%2.8f\n"%(lr_enc.get_last_lr()[0]))
        file.write("Batch loss:\n")
        file.write("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\tkld_loss\t%9.4f\txCritic_loss\t%9.4f\n"%(split.upper(), recon_loss/batch_size, mask_loss/batch_size, kld_loss/batch_size, xCritic_loss/batch_size))
        file.write("Accumulated loss:\n")
        file.write("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\tkld_loss\t%9.4f\txCritic_loss\t%9.4f\n"%(split.upper(), recon_total_loss/n_data, mask_total_loss/n_data, kld_total_loss/n_data, xCritic_total_loss/n_data))
        file.write("===================================================\n")
    
    if split == 'train':
        lr_enc.step()
        lr_dec.step()
        lr_dix.step()
    
    return recon_total_loss/n_data