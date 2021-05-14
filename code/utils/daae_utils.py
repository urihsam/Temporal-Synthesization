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
    G = models["G"]
    Dz = models["Dz"]

    torch.save(AE, "{}_AE".format(path))
    torch.save(Dx, "{}_Dx".format(path))
    torch.save(G, "{}_G".format(path))
    torch.save(Dz, "{}_Dz".format(path))


def load_model(path):
    AE = torch.load("{}_AE".format(path))
    Dx = torch.load("{}_Dx".format(path))
    G = torch.load("{}_G".format(path))
    Dz = torch.load("{}_Dz".format(path))

    models = {
        "AE": AE,
        "Dx": Dx,
        "G": G,
        "Dz": Dz
    }
    return models


def model_evaluation(args, models, opts, lrs, data_loader, prob_mask, split, log_file):
    AE = models["AE"]
    Dx = models["Dx"]
    G = models["G"]
    Dz = models["Dz"]
    if split == 'train':
        # opts
        opt_enc = opts["enc"]
        opt_dec = opts["dec"]
        opt_dix = opts["dix"]
        opt_diz = opts["diz"]
        opt_gen = opts["gen"]
        # lr scheduler
        lr_enc = lrs["enc"]
        lr_dec = lrs["dec"]
        lr_dix = lrs["dix"]
        lr_diz = lrs["diz"]
        lr_gen = lrs["gen"]

    # init
    recon_total_loss, mask_total_loss = 0.0, 0.0
    xCritic_total_loss, zCritic_total_loss = 0.0, 0.0
    
    n_data = 0

    if split == 'train':
        AE.encoder_dropout=args.encoder_dropout
        AE.decoder_dropout=args.decoder_dropout
        AE.train()
        Dx.train()
        G.train()
        Dz.train()
    else:
        AE.encoder_dropout=0.0
        AE.decoder_dropout=0.0
        AE.eval()
        Dx.eval()
        G.eval()
        Dz.eval()

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
            z, Pinput, Poutput, Moutput = AE(batch['tempo'], batch['target'], None, None)
            # loss
            recon_loss = args.beta_recon * AE.compute_recon_loss(Poutput, batch['target'], None, None)
        elif args.use_prob_mask:
            z, Pinput, Poutput, Moutput = AE(batch['tempo'], batch['target'], batch["mask"], batch["target_mask"])
            output_mask = sample_mask_from_prob(prob_mask, batch["target_mask"].shape[0], batch["target_mask"].shape[1])
            # loss
            recon_loss = args.beta_recon * AE.compute_recon_loss(Poutput, batch['target'], output_mask, batch["target_mask"])
        else:
            z, Pinput, Poutput, Moutput = AE(batch['tempo'], batch['target'], batch["mask"], batch["target_mask"])
            # loss
            recon_loss = args.beta_recon * AE.compute_recon_loss(Poutput, batch['target'], Moutput, batch["target_mask"])
            mask_loss = args.beta_mask * AE.compute_mask_loss(Moutput, batch["target_mask"])

        zgen = G(batch_size=z.size(0))
        # make up start feature
        start_feature, start_mask = sample_start_feature_and_mask(z.size(0))
        if args.no_mask:
            Pgen, Mgen = AE.decoder.inference(start_feature=start_feature, start_mask=None, z=zgen)
        elif args.use_prob_mask:
            Pgen, Mgen = AE.decoder.inference(start_feature=start_feature, start_mask=start_mask, prob_mask=prob_mask, z=zgen)
        else:
            Pgen, Mgen = AE.decoder.inference(start_feature=start_feature, start_mask=start_mask, z=zgen)

        Dinput, Doutput, Dgen = Dx(Pinput).mean(), Dx(Poutput, Moutput).mean(), Dx(Pgen, Mgen).mean()
        Dreal, Dfake = Dz(z).mean(), Dz(zgen).mean()

        xCritic_loss = - Dinput + 0.5 * (Doutput + Dgen)
        zCritic_loss = - Dreal + Dfake
        
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
                    
                # Step 2: Update the Critic_z
                opt_diz.zero_grad()
                Dreal, Dfake = Dz(z).mean(), Dz(zgen).mean()
                Dreal.backward(mone, retain_graph=True)
                Dfake.backward(one, retain_graph=True)
                Dz.cal_gradient_penalty(z, zgen).backward()
                opt_diz.step()

            # Step 3, 4: Update the Decoder and the Encoder
            opt_dec.zero_grad()
            Doutput, Dgen = Dx(Poutput, Moutput).mean(), Dx(Pgen, Mgen).mean()
            Doutput.backward(mone, retain_graph=True)
            Dgen.backward(mone, retain_graph=True)
            
            opt_enc.zero_grad()
            Dreal = Dz(z).mean()
            Dreal.backward(one, retain_graph=True)

            recon_loss.backward(retain_graph=True)
            if not args.no_mask and not args.use_prob_mask:
                mask_loss.backward(retain_graph=True)
            opt_dec.step()
            opt_enc.step()

            # Step 5: Update the Generator
            opt_gen.zero_grad()
            Dfake = Dz(zgen).mean()
            Dfake.backward(mone, retain_graph=True)
            opt_gen.step()
        
        #import pdb; pdb.set_trace()
        #
        recon_total_loss += recon_loss.data
        if not args.no_mask and not args.use_prob_mask:
            mask_total_loss += mask_loss.data
        else:
            mask_total_loss = 0.0
            mask_loss = 0.0
        xCritic_total_loss += xCritic_loss.data
        zCritic_total_loss += zCritic_loss.data

        if split == 'train' and iteration % args.train_eval_freq == 0:
            # print the losses for each epoch
            print("Learning rate:\t%2.8f"%(lr_gen.get_last_lr()[0]))
            print("Batch loss:")
            print("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f"%(split.upper(), recon_loss/batch_size, mask_loss/batch_size, xCritic_loss/batch_size, zCritic_loss/batch_size))
            print("Accumulated loss:")
            print("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f"%(split.upper(), recon_total_loss/n_data, mask_total_loss/n_data, xCritic_total_loss/n_data, zCritic_total_loss/n_data))
            print()
            with open(log_file, "a+") as file:
                file.write("Learning rate:\t%2.8f\n"%(lr_gen.get_last_lr()[0]))
                file.write("Batch loss:\n")
                file.write("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\n"%(split.upper(), recon_loss/batch_size, mask_loss/batch_size, xCritic_loss/batch_size, zCritic_loss/batch_size))
                file.write("Accumulated loss:\n")
                file.write("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\n"%(split.upper(), recon_total_loss/n_data, mask_total_loss/n_data, xCritic_total_loss/n_data, zCritic_total_loss/n_data))
                file.write("===================================================\n")
    #
    # print the losses for each epoch
    if split == 'train':
        print("Learning rate:\t%2.8f"%(lr_gen.get_last_lr()[0]))
    print("Batch loss:")
    print("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f"%(split.upper(), recon_loss/batch_size, mask_loss/batch_size, xCritic_loss/batch_size, zCritic_loss/batch_size))
    print("Accumulated loss:")
    print("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f"%(split.upper(), recon_total_loss/n_data, mask_total_loss/n_data, xCritic_total_loss/n_data, zCritic_total_loss/n_data))
    print()
    with open(log_file, "a+") as file:
        if split == 'train':
            file.write("Learning rate:\t%2.8f\n"%(lr_gen.get_last_lr()[0]))
        file.write("Batch loss:\n")
        file.write("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\n"%(split.upper(), recon_loss/batch_size, mask_loss/batch_size, xCritic_loss/batch_size, zCritic_loss/batch_size))
        file.write("Accumulated loss:\n")
        file.write("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\n"%(split.upper(), recon_total_loss/n_data, mask_total_loss/n_data, xCritic_total_loss/n_data, zCritic_total_loss/n_data))
        file.write("===================================================\n")
    
    if split == 'train':
        lr_enc.step()
        lr_dec.step()
        lr_dix.step()
        lr_diz.step()
        lr_gen.step()
    
    return recon_total_loss/n_data