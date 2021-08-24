import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from utils.train_utils import to_var,sample_gender_race, sample_start_feature_time_mask, sample_mask_from_prob, model_inference

from pyvacy import optim, analysis
from pyvacy.optim.dp_optimizer import DPAdam, DPSGD
import pyvacy.analysis.moments_accountant as moments_accountant

from nn.transformers.mixed_embedding_transformer import Transformer, TransformerDecoder
from nn.generator import MLP_Generator
from nn.discriminator import MLP_Discriminator, CNN_Discriminator


def train_model(args, datasets, prob_mask, **kwargs):
    if not args.test:
        # model define
        if args.load_model:
            model_path = os.path.join(args.model_path, args.pretrained_model_filename)
            models = load_model(model_path)
            Trans = models["Trans"]
            Dx = models["Dx"]
            G = models["G"]
            Dz = models["Dz"]
            Imi = models["Imi"]
            Di = models["Di"]
            
        else:
            dim_time = int((kwargs["time_shift"]+kwargs["time_scale"])*1.5)

            Trans = Transformer(
                num_encoder_layers=args.num_encoder_layers, #6 #1
                num_decoder_layers=args.num_decoder_layers, #6 #1
                max_length=args.max_length,
                dim_feature=args.feature_size,
                dim_model=args.latent_size, #512 #128
                dim_time=dim_time, # 
                num_heads=args.num_heads, #6 #3
                dim_feedforward=args.hidden_size, #2048 #128
                encoder_dropout=args.encoder_dropout,
                decoder_dropout=args.decoder_dropout,
                use_prob_mask=args.use_prob_mask
                )

            Dx = CNN_Discriminator(
                feature_size=args.feature_size,
                feature_dropout=args.feature_dropout,
                filter_size=args.filter_size,
                window_sizes=args.window_sizes,
                use_spectral_norm = args.use_spectral_norm
                )

            G = MLP_Generator(
                input_size=args.noise_size,
                output_size=args.latent_size,
                archs=args.gmlp_archs
                )

            Dz = CNN_Discriminator(
                feature_size=args.latent_size*2,
                feature_dropout=args.feature_dropout,
                filter_size=args.filter_size,
                window_sizes=args.window_sizes,
                use_spectral_norm = args.use_spectral_norm
                )

            # for imitation of Trans' Decoder
            Imi = TransformerDecoder(
                time_embedding=None,
                gender_embedding=None,
                race_embedding=None,
                num_layers=args.num_decoder_layers,
                max_length=args.max_length,
                dim_feature=args.feature_size,
                dim_model=args.latent_size,
                dim_time=dim_time,
                num_heads=args.num_heads,
                dim_feedforward=args.hidden_size,
                dropout=args.decoder_dropout,
                use_prob_mask=args.use_prob_mask
            )

            Di = CNN_Discriminator(
                feature_size=args.feature_size,
                feature_dropout=args.feature_dropout,
                filter_size=args.filter_size,
                window_sizes=args.window_sizes,
                use_spectral_norm = args.use_spectral_norm
                )
            

        if torch.cuda.is_available():
            Trans = Trans.cuda()
            Dx = Dx.cuda()
            G = G.cuda()
            Dz = Dz.cuda()
            Imi = Imi.cuda()
            Di = Di.cuda()
        

        opt_enc = torch.optim.Adam(Trans.encoder.parameters(), lr=args.enc_learning_rate)
        opt_dec = torch.optim.Adam(Trans.decoder.parameters(), lr=args.dec_learning_rate)
        opt_dix = torch.optim.Adam(Dx.parameters(), lr=args.dx_learning_rate)
        opt_diz = torch.optim.Adam(Dz.parameters(), lr=args.dz_learning_rate)
        opt_gen = torch.optim.Adam(G.parameters(), lr=args.g_learning_rate)
        opt_imi = torch.optim.Adam(Imi.parameters(), lr=args.imi_learning_rate)
        opt_dii = torch.optim.Adam(Di.parameters(), lr=args.di_learning_rate)
        #
        if args.dp_sgd == True: # Different from others
            opt_diz = DPSGD(params=Dz.parameters(), lr=args.dz_learning_rate, minibatch_size=args.batch_size, microbatch_size=args.batch_size,
                                        l2_norm_clip=args.l2_norm_clip, noise_multiplier=args.noise_multiplier)
            opt_dii = DPSGD(params=Di.parameters(), lr=args.di_learning_rate, minibatch_size=args.batch_size, microbatch_size=args.batch_size,
                                        l2_norm_clip=args.l2_norm_clip, noise_multiplier=args.noise_multiplier)
            epsilon = moments_accountant.epsilon(len(datasets['train'].data), args.batch_size, args.noise_multiplier, args.epochs, args.delta)

            print('Training procedure satisfies (%f, %f)-DP' % (epsilon, args.delta)) # 


        lr_enc = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_enc, gamma=args.enc_lr_decay_rate)
        lr_dec = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_dec, gamma=args.dec_lr_decay_rate)
        lr_dix = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_dix, gamma=args.dx_lr_decay_rate)
        lr_diz = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_diz, gamma=args.dz_lr_decay_rate)
        lr_gen = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_gen, gamma=args.g_lr_decay_rate)
        lr_imi = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_imi, gamma=args.imi_lr_decay_rate)
        lr_dii = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_dii, gamma=args.di_lr_decay_rate)

        
        tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        models = {
                "Trans": Trans,
                "Dx": Dx,
                "G": G,
                "Dz": Dz,
                "Imi": Imi,
                "Di": Di
            } 
        
        opts = {
            "enc": opt_enc,
            "dec": opt_dec,
            "dix": opt_dix,
            "diz": opt_diz,
            "gen": opt_gen,
            "imi": opt_imi,
            "dii": opt_dii
        }
        lrs = {
            "enc": lr_enc,
            "dec": lr_dec,
            "dix": lr_dix,
            "diz": lr_diz,
            "gen": lr_gen,
            "imi": lr_imi,
            "dii": lr_dii
        }
        min_valid_loss = float("inf")
        min_valid_path = ""
        for epoch in range(args.epochs):

            print("Epoch\t%02d/%i"%(epoch, args.epochs))
            
            data_loader = DataLoader(
                dataset=datasets["train"],
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )
        
            log_file = os.path.join(args.result_path, args.train_log)
            model_evaluation(args, models, opts, lrs, data_loader, prob_mask, "train", log_file, **kwargs)
        
            if epoch % args.valid_eval_freq == 0:
                data_loader = DataLoader(
                    dataset=datasets["valid"],
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=cpu_count(),
                    pin_memory=torch.cuda.is_available()
                )
            
                print("Validation:")
                log_file = os.path.join(args.result_path, args.valid_log)
                valid_loss = model_evaluation(args, models, opts, lrs, data_loader, prob_mask, "valid", log_file, **kwargs)
                print("****************************************************")
                print()
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    path = "{}/model_vloss_{}".format(args.model_path, valid_loss)
                    min_valid_path = path

                    models = {
                        "Trans": Trans,
                        "Dx": Dx,
                        "G": G,
                        "Dz": Dz,
                        "Imi": Imi,
                        "Di": Di
                    }
                    save_model(models, path)

            
        # Generate the synthetic sequences as many as you want 
        
        model_path = min_valid_path
    else:
        model_path = os.path.join(args.model_path, args.test_model_filename)
    
    models = load_model(model_path)
    Imi = models["Imi"]
    G = models["G"]
    Imi.eval()
    G.eval()
    gen_zs, gen_xs, gen_ms = [], [], []
    for i in range(args.gendata_size//args.batch_size):
        zgen = G(batch_size=args.batch_size*args.max_length)
        zgen = torch.reshape(zgen, (args.batch_size, args.max_length, -1))
        Pimi, Mimi = model_inference(args, Imi, zgen, prob_mask, **kwargs)
        
        gen_zs.append(zgen)
        gen_xs.append(Pimi)
        gen_ms.append(Mimi)

    gen_zlist = torch.cat(gen_zs).cpu().detach().numpy()
    gen_xlist = torch.cat(gen_xs).cpu().detach().numpy()
    
    np.save(os.path.join(args.result_path, '{}_generated_codes.npy'.format(args.model_type)), gen_zlist)
    np.save(os.path.join(args.result_path, '{}_generated_patients.npy'.format(args.model_type)), gen_xlist) 
    
    if not args.no_mask and not args.use_prob_mask:
        gen_mlist = torch.cat(gen_ms).cpu().detach().numpy()
        np.save(os.path.join(args.result_path, '{}_generated_masks.npy'.format(args.model_type)), gen_mlist)



def save_model(models, path):
    Trans = models["Trans"]
    Dx = models["Dx"]
    G = models["G"]
    Dz = models["Dz"]
    Imi = models["Imi"]
    Di = models["Di"]

    torch.save(Trans, "{}_Trans".format(path))
    torch.save(Dx, "{}_Dx".format(path))
    torch.save(G, "{}_G".format(path))
    torch.save(Dz, "{}_Dz".format(path))
    torch.save(Imi, "{}_Imi".format(path))
    torch.save(Di, "{}_Di".format(path))


def load_model(path):
    Trans = torch.load("{}_Trans".format(path))
    Dx = torch.load("{}_Dx".format(path))
    G = torch.load("{}_G".format(path))
    Dz = torch.load("{}_Dz".format(path))
    Imi = torch.load("{}_Imi".format(path))
    Di = torch.load("{}_Di".format(path))

    models = {
        "Trans": Trans,
        "Dx": Dx,
        "G": G,
        "Dz": Dz,
        "Imi": Imi,
        "Di": Di
    }
    return models


def model_evaluation(args, models, opts, lrs, data_loader, prob_mask, split, log_file, **kwargs):
    Trans = models["Trans"]
    Dx = models["Dx"]
    G = models["G"]
    Dz = models["Dz"]
    Imi = models["Imi"]
    Di = models["Di"]
    if split == 'train':
        # opts
        opt_enc = opts["enc"]
        opt_dec = opts["dec"]
        opt_dix = opts["dix"]
        opt_diz = opts["diz"]
        opt_gen = opts["gen"]
        opt_imi = opts["imi"]
        opt_dii = opts["dii"]
        # lr scheduler
        lr_enc = lrs["enc"]
        lr_dec = lrs["dec"]
        lr_dix = lrs["dix"]
        lr_diz = lrs["diz"]
        lr_gen = lrs["gen"]
        lr_imi = lrs["imi"]
        lr_dii = lrs["dii"]

    # init
    recon_total_loss, mask_total_loss = 0.0, 0.0
    xCritic_total_loss, zCritic_total_loss, iCritic_total_loss = 0.0, 0.0, 0.0
    
    n_data = 0

    if split == 'train':
        Trans.encoder_dropout=args.encoder_dropout
        Trans.decoder_dropout=args.decoder_dropout
        Trans.train()
        Dx.train()
        G.train()
        Dz.train()
        Imi.dropout = args.decoder_dropout
        Imi.train()
        Di.train()
    else:
        Trans.encoder_dropout=0.0
        Trans.decoder_dropout=0.0
        Trans.eval()
        Dx.eval()
        G.eval()
        Dz.eval()
        Imi.dropout = 0.0
        Imi.eval()
        Di.eval()

    for iteration, batch in enumerate(data_loader):
        batch_size = batch['src_tempo'].shape[0]
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
        
        #print("max src_time", torch.amax(batch['src_time'], [0,1]), " -- min src_time", torch.amin(batch['src_time'], [0,1]))
        #print("max tgt_time", torch.amax(batch['tgt_time'], [0,1]), " -- min tgt_time", torch.amin(batch['tgt_time'], [0,1]))
        src_tempo = batch['src_tempo']; tgt_tempo = batch['tgt_tempo']
        src_time = batch['src_time']; tgt_time = batch['tgt_time']
        gender = batch['gender']
        race = batch['race']
        src_mask = batch['src_mask']; tgt_mask = batch['tgt_mask']
        src_ava = batch['src_ava']; tgt_ava = batch['tgt_ava']

        if args.no_mask:
            z, Pinput, Poutput, _, Moutput = Trans(src_tempo, tgt_tempo, src_time, tgt_time, gender, race, 
                                                None, None, src_ava, tgt_ava)
            # loss
            recon_loss = args.beta_recon * Trans.compute_recon_loss(Poutput, tgt_tempo, None, None)
        elif args.use_prob_mask:
            z, Pinput, Poutput, _, Moutput = Trans(src_tempo, tgt_tempo, src_time, tgt_time, gender, race,
                                                src_mask, tgt_mask, src_ava, tgt_ava)
            output_mask = sample_mask_from_prob(prob_mask, tgt_mask.shape[0], tgt_mask.shape[1])
            # loss
            recon_loss = args.beta_recon * Trans.compute_recon_loss(Poutput, tgt_tempo, output_mask, tgt_mask)
        else:
            z, Pinput, Poutput, _, Moutput = Trans(src_tempo, tgt_tempo, src_time, tgt_time, gender, race,
                                                src_mask, tgt_mask, src_ava, tgt_ava)
            # loss
            recon_loss = args.beta_recon * Trans.compute_recon_loss(Poutput, tgt_tempo, Moutput, tgt_mask)
            mask_loss = args.beta_mask * Trans.compute_mask_loss(Moutput, tgt_mask)

        zgen = G(batch_size=z.size(0)*args.max_length)
        zgen = torch.reshape(zgen, (z.size(0), args.max_length, -1))
        # make up start feature
        start_feature, start_time, start_mask = sample_start_feature_time_mask(z.size(0))
        kwargs["start_time"] = start_time
        sampled_gender, sampled_race = sample_gender_race(z.size(0))
        kwargs["gender"] = sampled_gender
        kwargs["race"] = sampled_race
        if args.no_mask:
            Pgen, _, Mgen = Trans.decoder.inference(start_feature=start_feature, start_mask=None, memory=zgen, **kwargs)
            Pimi, _, Mimi = Imi.inference(start_feature=start_feature, start_mask=None, memory=zgen, **kwargs)
        elif args.use_prob_mask:
            Pgen, _, Mgen = Trans.decoder.inference(start_feature=start_feature, start_mask=start_mask, prob_mask=prob_mask, memory=zgen, **kwargs)
            Pimi, _, Mimi = Imi.inference(start_feature=start_feature, start_mask=start_mask, prob_mask=prob_mask, memory=zgen, **kwargs)
        else:
            Pgen, _, Mgen = Trans.decoder.inference(start_feature=start_feature, start_mask=start_mask, memory=zgen, **kwargs)
            Pimi, _, Mimi = Imi.inference(start_feature=start_feature, start_mask=start_mask, memory=zgen, **kwargs)

        #import pdb; pdb.set_trace()
        Dinput, Doutput, Dgen = Dx(Pinput).mean(), Dx(Poutput, Moutput).mean(), Dx(Pgen, Mgen).mean()
        Diinput, Dioutput, Diimi = Di(Pinput).mean(), Di(Poutput, Moutput).mean(), Di(Pimi, Mimi).mean()
        # reshape z, zgen
        Dreal, Dfake = Dz(z).mean(), Dz(zgen).mean()

        xCritic_loss = - Dinput + 0.5 * (Doutput + Dgen)
        zCritic_loss = - Dreal + Dfake
        #
        iCritic_loss = - Diinput + 0.5 * (Dioutput + Diimi)
            

        if split == 'train':
            if iteration % args.critic_freq_base < args.critic_freq_hit:
                # Step 1: Update the Critic_x
                params = tuple(Dx.parameters())
                opt_dix.zero_grad()
                Dinput, Doutput = Dx(tgt_tempo, tgt_mask).mean(), Dx(Poutput, Moutput).mean()
                Dinput.backward(mone, inputs=params, retain_graph=True)
                Doutput.backward(one, inputs=params, retain_graph=True)
                Dx.cal_gradient_penalty(tgt_tempo[:, :Poutput.size(1), :], Poutput, tgt_mask, Moutput).backward(inputs=params, retain_graph=True)
                opt_dix.step()

                opt_dix.zero_grad()
                Dinput, Dgen = Dx(tgt_tempo, tgt_mask).mean(), Dx(Pgen, Mgen).mean()
                Dinput.backward(mone, inputs=params, retain_graph=True)
                Dgen.backward(one, inputs=params, retain_graph=True)
                Dx.cal_gradient_penalty(tgt_tempo[:, :Pgen.size(1), :], Pgen, tgt_mask, Mgen).backward(inputs=params, retain_graph=True)
                opt_dix.step()
                    
                # Step 2: Update the Critic_z
                params = tuple(Dz.parameters())
                opt_diz.zero_grad()
                Dreal, Dfake = Dz(z).mean(), Dz(zgen).mean()
                Dreal.backward(mone, inputs=params, retain_graph=True)
                Dfake.backward(one, inputs=params, retain_graph=True)
                Dz.cal_gradient_penalty(z, zgen).backward()
                opt_diz.step()
                

                

            # Step 3, 4: Update the Decoder and the Encoder
            params = tuple(Trans.parameters())
            opt_dec.zero_grad()
            Doutput, Dgen = Dx(Poutput, Moutput).mean(), Dx(Pgen, Mgen).mean()
            Doutput.backward(mone, inputs=params, retain_graph=True)
            Dgen.backward(mone, inputs=params, retain_graph=True)
            
            opt_enc.zero_grad()
            Dreal = Dz(z).mean()
            Dreal.backward(one, inputs=params, retain_graph=True)

            recon_loss.backward(inputs=params, retain_graph=True)
            if not args.no_mask and not args.use_prob_mask:
                mask_loss.backward(inputs=params, retain_graph=True)
            
            opt_dec.step()
            opt_enc.step()

            # Step 5: Update the Generator
            params = tuple(G.parameters())
            opt_gen.zero_grad()
            Dfake = Dz(zgen).mean()
            Dfake.backward(mone, inputs=params, retain_graph=True)
            opt_gen.step()
            
            if iteration % args.critic_freq_base < args.critic_freq_hit:
                # Step 6: Update the Critic_i
                params = tuple(Di.parameters())
                opt_dii.zero_grad()
                Diinput, Dioutput = Di(tgt_tempo).mean(), Di(Poutput).mean()
                Diinput.backward(mone, inputs=params, retain_graph=True)
                Dioutput.backward(one, inputs=params, retain_graph=True)
                Di.cal_gradient_penalty(tgt_tempo[:, :Poutput.size(1), :], Poutput, tgt_mask, Moutput).backward(inputs=params, retain_graph=True)
                opt_dii.step()

                opt_dii.zero_grad()
                Diinput, Diimi = Di(tgt_tempo).mean(), Di(Pimi).mean()
                Diinput.backward(mone, inputs=params, retain_graph=True)
                Diimi.backward(one, inputs=params, retain_graph=True)
                Di.cal_gradient_penalty(tgt_tempo[:, :Pimi.size(1), :], Pimi, tgt_mask, Mimi).backward(inputs=params, retain_graph=True)
                opt_dii.step()

            # Step 7: Update the Imitator
            params = tuple(Imi.parameters())
            opt_imi.zero_grad()
            Diimi = Di(Pimi, Mimi).mean()
            #Dioutput = Di(Poutput, Moutput).mean()
            Diimi.backward(mone, inputs=params, retain_graph=True)
            #Dioutput.backward(mone, retain_graph=True)
            opt_imi.step()

            
        
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
        iCritic_total_loss += iCritic_loss.data

        if split == 'train' and iteration % args.train_eval_freq == 0:
            # print the losses for each epoch
            print("Learning rate:\t%2.8f"%(lr_gen.get_last_lr()[0]))
            print("Batch loss:")
            print("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\tiCritic_loss\t%9.4f"%(
                    split.upper(), recon_loss/batch_size, mask_loss/batch_size, xCritic_loss/batch_size, zCritic_loss/batch_size, iCritic_loss/batch_size))
            print("Accumulated loss:")
            print("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\tiCritic_loss\t%9.4f"%(
                    split.upper(), recon_total_loss/n_data, mask_total_loss/n_data, xCritic_total_loss/n_data, zCritic_total_loss/n_data, iCritic_total_loss/n_data))
            print()
            with open(log_file, "a+") as file:
                file.write("Learning rate:\t%2.8f\n"%(lr_gen.get_last_lr()[0]))
                file.write("Batch loss:\n")
                file.write("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\tiCritic_loss\t%9.4f\n"%(
                        split.upper(), recon_loss/batch_size, mask_loss/batch_size, xCritic_loss/batch_size, zCritic_loss/batch_size, iCritic_loss/batch_size))
                file.write("Accumulated loss:\n")
                file.write("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\tiCritic_loss\t%9.4f\n"%(
                        split.upper(), recon_total_loss/n_data, mask_total_loss/n_data, xCritic_total_loss/n_data, zCritic_total_loss/n_data, iCritic_total_loss/n_data))
                file.write("===================================================\n")
    #
    # print the losses for each epoch
    if split == 'train':
        print("Learning rate:\t%2.8f"%(lr_gen.get_last_lr()[0]))
    print("Batch loss:")
    print("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\tiCritic_loss\t%9.4f"%(
            split.upper(), recon_loss/batch_size, mask_loss/batch_size, xCritic_loss/batch_size, zCritic_loss/batch_size, iCritic_loss/batch_size))
    print("Accumulated loss:")
    print("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\tiCritic_loss\t%9.4f"%(
            split.upper(), recon_total_loss/n_data, mask_total_loss/n_data, xCritic_total_loss/n_data, zCritic_total_loss/n_data, iCritic_total_loss/n_data))
    print()
    with open(log_file, "a+") as file:
        if split == 'train':
            file.write("Learning rate:\t%2.8f\n"%(lr_gen.get_last_lr()[0]))
        file.write("Batch loss:\n")
        file.write("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\tiCritic_loss\t%9.4f\n"%(
            split.upper(), recon_loss/batch_size, mask_loss/batch_size, xCritic_loss/batch_size, zCritic_loss/batch_size, iCritic_loss/batch_size))
        file.write("Accumulated loss:\n")
        file.write("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\tiCritic_loss\t%9.4f\n"%(
            split.upper(), recon_total_loss/n_data, mask_total_loss/n_data, xCritic_total_loss/n_data, zCritic_total_loss/n_data, iCritic_total_loss/n_data))
        file.write("===================================================\n")
    
    if split == 'train':
        lr_enc.step()
        lr_dec.step()
        lr_dix.step()
        lr_diz.step()
        lr_gen.step()
        lr_imi.step()
        lr_dii.step()
    
    return recon_total_loss/n_data