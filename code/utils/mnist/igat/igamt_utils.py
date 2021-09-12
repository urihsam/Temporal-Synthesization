import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from utils.train_utils import to_var, mnist_model_inference as model_inference


from pyvacy import optim, analysis
from pyvacy.optim.dp_optimizer import DPAdam, DPSGD
import pyvacy.analysis.moments_accountant as moments_accountant

from nn.transformers.naive_transformer import Transformer, GeneralTransformerDecoder
from nn.generator import MLP_Generator
from nn.discriminator import MLP_Discriminator, CNN_Discriminator, CNN_Auxiliary_Discriminator, CNN_Net

lin_params_size =  2

def train_model(args, datasets, prob_mask, **kwargs):
    max_length = args.img_w//args.slice_w * args.img_h//args.slice_h + 1
    if not args.test:
        # model define
        if args.load_model:
            model_path = os.path.join(args.model_path, args.pretrained_model_filename)
            models = load_model(model_path)
            Trans = models["Trans"]
            Dx = models["Dx"]
            Dm = models["Dm"]
            G = models["G"]
            Dz = models["Dz"]
            Imi = models["Imi"]
            
        else:
            Trans = Transformer(
                num_encoder_layers=args.num_encoder_layers, #6 #1
                num_decoder_layers=args.num_decoder_layers, #6 #1
                max_length=max_length,
                dim_feature=args.feature_size,
                dim_model=args.latent_size, #512 #128
                num_heads=args.num_heads, #6 #3
                dim_feedforward=args.hidden_size, #2048 #128
                encoder_dropout=args.encoder_dropout,
                decoder_dropout=args.decoder_dropout,
                no_generated_mask=True,
                use_prob_mask=False
                )

            Dx = CNN_Auxiliary_Discriminator(
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
                feature_size=args.latent_size,#*2,
                feature_dropout=args.feature_dropout,
                filter_size=args.filter_size,
                window_sizes=args.window_sizes,
                use_spectral_norm = args.use_spectral_norm
                )
            # imi
            Imi = GeneralTransformerDecoder(
                num_layers=args.num_decoder_layers,
                max_length=max_length,
                dim_feature=args.feature_size,
                dim_model=args.latent_size,
                num_heads=args.num_heads,
                dim_feedforward=args.hidden_size,
                dropout=args.decoder_dropout,
                no_generated_mask=True,
                use_prob_mask=False,
                linear=Trans.decoder.linear
            )
            

        if torch.cuda.is_available():
            Trans = Trans.cuda()
            Dx = Dx.cuda()
            G = G.cuda()
            Dz = Dz.cuda()
            Imi = Imi.cuda()

        
        
        opt_enc = torch.optim.Adam(Trans.encoder.parameters(), lr=args.enc_learning_rate)
        opt_dec = torch.optim.Adam(Trans.decoder.parameters(), lr=args.dec_learning_rate)
        opt_dx = torch.optim.Adam(Dx.parameters(), lr=args.dx_learning_rate)
        opt_dz = torch.optim.Adam(Dz.parameters(), lr=args.dz_learning_rate)
        opt_gen = torch.optim.Adam(G.parameters(), lr=args.g_learning_rate)
        opt_imi = torch.optim.Adam(Imi.parameters(), lr=args.imi_learning_rate)
        #opt_imi = torch.optim.Adam(filter(lambda p: p.requires_grad, Imi.parameters()), lr=args.imi_learning_rate)
        #
        if args.dp_sgd == True: # opt_dx and opt_dz access origin data too?
            opt_dec = torch.optim.Adam(list(Trans.decoder.parameters())[:-lin_params_size], lr=args.dec_learning_rate)
            opt_lin = DPSGD(params=list(Trans.decoder.parameters())[-lin_params_size:], lr=args.dec_learning_rate, minibatch_size=args.batch_size, microbatch_size=args.batch_size,
                                        l2_norm_clip=args.l2_norm_clip, noise_multiplier=args.noise_multiplier)
            opt_dx = DPSGD(params=Dx.parameters(), lr=args.dx_learning_rate, minibatch_size=args.batch_size, microbatch_size=args.batch_size,
                                        l2_norm_clip=args.l2_norm_clip, noise_multiplier=args.noise_multiplier)
            opt_dz = DPSGD(params=Dz.parameters(), lr=args.dz_learning_rate, minibatch_size=args.batch_size, microbatch_size=args.batch_size, 
                                        l2_norm_clip=args.l2_norm_clip, noise_multiplier=args.noise_multiplier)
            opt_imi = torch.optim.Adam(list(Imi.parameters())[:-lin_params_size], lr=args.imi_learning_rate)
            
            epsilon = moments_accountant.epsilon(datasets.train_size, args.batch_size, args.noise_multiplier, args.epochs, args.delta)

            print('Training procedure satisfies (%f, %f)-DP' % (epsilon, args.delta)) # ?? question, why 2 epsilon?


            lr_lin = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_lin, gamma=args.dec_lr_decay_rate)


        lr_enc = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_enc, gamma=args.enc_lr_decay_rate)
        lr_dec = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_dec, gamma=args.dec_lr_decay_rate)
        lr_dx = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_dx, gamma=args.dx_lr_decay_rate)
        lr_dz = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_dz, gamma=args.dz_lr_decay_rate)
        lr_gen = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_gen, gamma=args.g_lr_decay_rate)
        lr_imi = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_imi, gamma=args.imi_lr_decay_rate)

        
        tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        models = {
                "Trans": Trans,
                "Dx": Dx,
                "G": G,
                "Dz": Dz,
                "Imi": Imi
            } 
        
        opts = {
            "enc": opt_enc,
            "dec": opt_dec,
            "dx": opt_dx,
            "dz": opt_dz,
            "gen": opt_gen,
            "imi": opt_imi
        }
        lrs = {
            "enc": lr_enc,
            "dec": lr_dec,
            "dx": lr_dx,
            "dz": lr_dz,
            "gen": lr_gen,
            "imi": lr_imi
        }

        if args.dp_sgd == True: 
            opts["lin"] = opt_lin
            lrs["lin"] = lr_lin
        

        min_valid_loss = float("inf")
        min_valid_path = ""
        for epoch in range(args.epochs):

            print("Epoch\t%02d/%i"%(epoch, args.epochs))
            
            log_file = os.path.join(args.result_path, args.train_log)
            _, models = model_evaluation(args, models, opts, lrs, datasets, prob_mask, "train", log_file, **kwargs)

            torch.cuda.empty_cache()
            if epoch % args.valid_eval_freq == 0:
                torch.cuda.empty_cache()
            
                print("Validation:")
                log_file = os.path.join(args.result_path, args.valid_log)
                valid_loss, models = model_evaluation(args, models, opts, lrs, datasets, prob_mask, "valid", log_file, **kwargs)
                print("****************************************************")
                print()
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    path = "{}/model_vloss_{}".format(args.model_path, valid_loss)
                    min_valid_path = path

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
    model_generation(args, G, Imi, max_length, prob_mask, **kwargs)
    #
    Dec = models["Trans"].decoder
    Dec.eval()
    model_generation(args, G, Dec, max_length, prob_mask, prefix=args.model_type+"_pub", **kwargs)


def model_generation(args, G_0, G_1, max_len, path=None, prefix=None, **kwargs,):
    if path is None:
        path = args.result_path
    if prefix is None:
        prefix = args.model_type
    gen_zs, gen_xs, gen_ms = [], [], []
    for i in range(args.gendata_size//args.batch_size):
        zgen = G_0(batch_size=args.batch_size*max_len)
        zgen = torch.reshape(zgen, (args.batch_size, max_len, -1))
        Pimi, Mimi = model_inference(args, G_1, zgen, args.batch_size, args.feature_size, **kwargs)
        
        gen_zs.append(zgen)
        gen_xs.append(Pimi)

    gen_zlist = torch.cat(gen_zs).cpu().detach().numpy()
    gen_xlist = torch.cat(gen_xs).cpu().detach().numpy()
    
    np.save(os.path.join(path, '{}_generated_codes.npy'.format(prefix)), gen_zlist)
    np.save(os.path.join(path, '{}_generated_patients.npy'.format(prefix)), gen_xlist) 





def save_model(models, path):
    Trans = models["Trans"]
    Dx = models["Dx"]
    G = models["G"]
    Dz = models["Dz"]
    Imi = models["Imi"]

    torch.save(Trans, "{}_Trans".format(path))
    torch.save(Dx, "{}_Dx".format(path))
    torch.save(G, "{}_G".format(path))
    torch.save(Dz, "{}_Dz".format(path))
    torch.save(Imi, "{}_Imi".format(path))


def load_model(path):
    Trans = torch.load("{}_Trans".format(path))
    Dx = torch.load("{}_Dx".format(path))
    G = torch.load("{}_G".format(path))
    Dz = torch.load("{}_Dz".format(path))
    Imi = torch.load("{}_Imi".format(path))

    models = {
        "Trans": Trans,
        "Dx": Dx,
        "G": G,
        "Dz": Dz,
        "Imi": Imi
    }
    return models


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def defreeze_params(params):
    for param in params:
        param.requires_grad = True


def model_evaluation(args, models, opts, lrs, data_loader, prob_mask, split, log_file, **kwargs):
    Trans = models["Trans"]
    Dx = models["Dx"]
    G = models["G"]
    Dz = models["Dz"]
    Imi = models["Imi"]

    if split == 'train':
        # opts
        opt_enc = opts["enc"]
        opt_dec = opts["dec"]
        opt_dx = opts["dx"]
        opt_dz = opts["dz"]
        opt_gen = opts["gen"]
        opt_imi = opts["imi"]

        # lr scheduler
        lr_enc = lrs["enc"]
        lr_dec = lrs["dec"]
        lr_dx = lrs["dx"]
        lr_dz = lrs["dz"]
        lr_gen = lrs["gen"]
        lr_imi = lrs["imi"]

        if args.dp_sgd == True: 
            opt_lin = opts["lin"]
            lr_lin = lrs["lin"]

    
    recon_total_loss, mask_total_loss = 0.0, 0.0
    xCritic_total_loss, zCritic_total_loss, mCritic_total_loss = 0.0, 0.0, 0.0
    input_match_total_loss, output_match_total_loss, gen_match_total_loss = 0.0, 0.0, 0.0
    input_m_match_total_loss, output_m_match_total_loss, gen_m_match_total_loss = 0.0, 0.0, 0.0

    if split == 'train':
        Trans.encoder_dropout=args.encoder_dropout
        Trans.decoder_dropout=args.decoder_dropout
        Trans.train()
        Dx.train()
        G.train()
        Dz.train()
        Imi.train()
    else:
        Trans.encoder_dropout=0.0
        Trans.decoder_dropout=0.0
        Trans.eval()
        Dx.eval()
        G.eval()
        Dz.eval()
        Imi.eval()

    if split == "train":
        data_size = data_loader.train_size
        get_batch = data_loader.next_train_batch
    elif split == "valid":
        data_size = data_loader.valid_size
        get_batch = data_loader.next_valid_batch
    else:
        data_size = data_loader.test_size
        get_batch = data_loader.next_test_batch
    batch_size = args.batch_size
    #import pdb; pdb.set_trace()
    for iteration in range(data_size//batch_size):
        #
        batch = get_batch(batch_size)
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = to_var(v)
            else:
                batch[k] = torch.tensor(v)
                if torch.cuda.is_available():
                    batch[k] = batch[k].cuda()

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1

        if torch.cuda.is_available():
            one = one.cuda()
            mone = mone.cuda()
        
        
        # Step 0: Evaluate current loss
        src_tempo = batch['src_tempo']; tgt_tempo = batch['tgt_tempo']
        
        #
        z, Pinput, Poutput, Moutput = Trans(src_tempo, tgt_tempo, None, None)
        # loss
        recon_loss = args.beta_recon * Trans.compute_recon_loss(Poutput, tgt_tempo, None, None)

        batch_size, max_len, fea_size = src_tempo.shape
        zgen = G(batch_size=z.size(0)*max_len)
        zgen = torch.reshape(zgen, (z.size(0), max_len, -1))
        # make up start feature
        
        start_feature = torch.tensor(np.zeros((batch_size, fea_size)))
        

        Pgen, Mgen = Trans.decoder.inference(start_feature=start_feature, start_mask=None, memory=zgen, **kwargs)
        Pimi, Mimi = Imi.inference(start_feature=start_feature, start_mask=None, memory=zgen, **kwargs)

        
        # match
        gen_match_loss = args.beta_match_g*Imi.compute_recon_loss(Pimi, Pgen, None, None, type="mse")
        input_match_loss = args.beta_match_i*Imi.compute_recon_loss(Pimi, tgt_tempo, None, None, type="mse")
        
        
        if split == 'train':
            if iteration % args.gen_freq_base < args.gen_freq_hit:
                if iteration % args.critic_freq_base < args.critic_freq_hit:
                    # Step 1: Update the Critic_x
                    params = list(Dx.parameters())
                    # generated data
                    opt_dx.zero_grad()
                    Dinput = Dx(tgt_tempo)
                    Doutput = Dx(Poutput)
                    Dinput = Dinput.mean()
                    Doutput = Doutput.mean()
                    Dinput.backward(mone, inputs=params, retain_graph=True) # maximize
                    Doutput.backward(one, inputs=params, retain_graph=True) # minimize
                    Dx.cal_gradient_penalty(tgt_tempo[:, :Poutput.size(1), :], Poutput).backward(inputs=params, retain_graph=True)
                    opt_dx.step()

                    opt_dx.zero_grad()
                    Dinput = Dx(tgt_tempo)
                    Dgen = Dx(Pgen)
                    Dinput = Dinput.mean()
                    Dgen = Dgen.mean()
                    Dinput.backward(mone, inputs=params, retain_graph=True)
                    Dgen.backward(one, inputs=params, retain_graph=True)
                    Dx.cal_gradient_penalty(tgt_tempo[:, :Pgen.size(1), :], Pgen).backward(inputs=params, retain_graph=True)
                    opt_dx.step()

                    
                    opt_dx.zero_grad()
                    Dinput = Dx(tgt_tempo)
                    Dimi = Dx(Pimi)
                    Dinput = Dinput.mean()
                    Dimi = Dimi.mean()
                    Dinput.backward(mone, inputs=params, retain_graph=True)
                    Dimi.backward(one, inputs=params, retain_graph=True)
                    Dx.cal_gradient_penalty(tgt_tempo[:, :Pimi.size(1), :], Pimi).backward(inputs=params, retain_graph=True)
                    opt_dx.step()

                        
                    # Step 3: Update the Critic_z
                    params = list(Dz.parameters())
                    opt_dz.zero_grad()
                    Dreal, Dfake = Dz(z).mean(), Dz(zgen).mean()
                    Dreal.backward(mone, inputs=params, retain_graph=True)
                    Dfake.backward(one, inputs=params, retain_graph=True)
                    Dz.cal_gradient_penalty(z, zgen).backward(inputs=params)
                    opt_dz.step()
                
            

                # Step 4, 5: Update the Decoder and the Encoder
                if args.dp_sgd:
                    params = list(Trans.parameters())[:-lin_params_size]
                else:
                    params = list(Trans.parameters())
                
                opt_dec.zero_grad()
                Doutput = Dx(Poutput)
                Dgen =  Dx(Pgen)
                Doutput = Doutput.mean()
                Dgen = Dgen.mean()
                Doutput.backward(mone, inputs=params, retain_graph=True)
                Dgen.backward(mone, inputs=params, retain_graph=True)
                

                opt_enc.zero_grad()

                Dreal = Dz(z).mean()
                Dreal.backward(one, inputs=params, retain_graph=True)
                
                if args.no_recon == False:
                    recon_loss.backward(inputs=params, retain_graph=True)

                opt_dec.step()
                opt_enc.step()
            
                if args.dp_sgd:
                    #import pdb; pdb.set_trace()
                    params = list(Trans.parameters())[-lin_params_size:]
                    
                    #z, Pinput, Poutput, Moutput = Trans(src_tempo, tgt_tempo, None, None)
                    # loss
                    #recon_loss = args.beta_recon * Trans.compute_recon_loss(Poutput, tgt_tempo, None, None)
                    opt_lin.zero_grad()
                    Doutput = Dx(Poutput)
                    Dgen =  Dx(Pgen)
                    Doutput = Doutput.mean()
                    Dgen = Dgen.mean()
                    Doutput.backward(mone, inputs=params, retain_graph=True)
                    Dgen.backward(mone, inputs=params, retain_graph=True)
                    
                    if args.no_recon == False:
                        recon_loss.backward(inputs=params, retain_graph=True)
                        
                    opt_lin.step()


                # step 6: Update Imi
                opt_imi.zero_grad()
                #import pdb; pdb.set_trace()
                params = list(Imi.parameters())[:-lin_params_size]
                #freeze_params(params)
                Dimi =  Dx(Pimi)
                Dimi = Dimi.mean()
                Dimi.backward(mone, inputs=params, retain_graph=True)
                opt_imi.step()

                opt_imi.zero_grad()
                # match
                gen_match_loss.backward(inputs=params, retain_graph=True)
                
                #opt_imi.zero_grad()
                # match
                input_match_loss.backward(inputs=params, retain_graph=True)
                
                Pimi, Mimi = Imi.inference(start_feature=start_feature, start_mask=None, memory=zgen, **kwargs)
                z, Pinput, Poutput, Moutput = Trans(src_tempo, tgt_tempo, None, None)
                output_match_loss = args.beta_match_o*Imi.compute_recon_loss(Pimi, Poutput, type="mse")
                #opt_imi.zero_grad()
                output_match_loss.backward(inputs=params, retain_graph=True)
                opt_imi.step()
                
                #defreeze_params(params)
                #


            # Step 6: Update the Generator
            params = list(G.parameters())
            opt_gen.zero_grad()
            Dfake = Dz(zgen).mean()
            Dfake.backward(mone, inputs=params, retain_graph=True)
            opt_gen.step()


        
        else:
            #import pdb; pdb.set_trace()
            # only for tempo data without mask
            Dinput = Dx(tgt_tempo)
            Doutput = Dx(Poutput)
            Dgen = Dx(Pgen)
            Dimi= Dx(Pimi)
            #
            Dinput = Dinput.mean()
            Doutput = Doutput.mean()
            Dgen = Dgen.mean()
            Dimi = Dimi.mean()

            # reshape z, zgen
            #z = torch.reshape(z, (-1, z.size(-1)))
            #zgen = torch.reshape(zgen, (-1, zgen.size(-1)))
            Dreal, Dfake = Dz(z).mean(), Dz(zgen).mean()

            
            output_match_loss = args.beta_match_o*Imi.compute_recon_loss(Pimi, Poutput, None, None, type="mse")
            #
            
        #
        #src_tempo = src_tempo.cpu(); tgt_tempo = tgt_tempo.cpu()
        #del src_tempo; del tgt_tempo
        #torch.cuda.empty_cache()
        #   
        xCritic_loss = (- Dinput + (Doutput + Dgen + Dimi) / 3.0).data
        zCritic_loss = (- Dreal + Dfake).data
        mCritic_loss = 0.0
        
        #
        recon_loss = recon_loss.data
        mask_loss = 0.0
        input_m_match_loss, output_m_match_loss, gen_m_match_loss = 0.0, 0.0, 0.0
        
        if split != "train":
            recon_total_loss += recon_loss
            mask_total_loss = 0.0
            xCritic_total_loss += xCritic_loss
            zCritic_total_loss += zCritic_loss
            input_match_total_loss += input_match_loss
            output_match_total_loss += output_match_loss
            gen_match_total_loss += gen_match_loss
            
        if split == 'train' and iteration % args.train_eval_freq == 0:
            # print the losses for each epoch
            print("Learning rate:\t%2.8f"%(lr_gen.get_last_lr()[0]))
            print("Batch loss:")
            print("%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\tmCritic_loss\t%9.4f\n\tinput_match_loss\t%9.4f\toutput_match_loss\t%9.4f\tgen_match_loss\t%9.4f\n\tinput_mask_match_loss\t%9.4f\toutput_mask_match_loss\t%9.4f\timi_mask_match_loss\t%9.4f"%(
                    split.upper(), recon_loss, mask_loss, xCritic_loss, zCritic_loss, mCritic_loss,
                    input_match_loss, output_match_loss, gen_match_loss, 
                    input_m_match_loss, output_m_match_loss, gen_m_match_loss))
            print()
            with open(log_file, "a+") as file:
                file.write("Learning rate:\t%2.8f\n"%(lr_gen.get_last_lr()[0]))
                file.write("Batch loss:\n")
                file.write("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\tmCritic_loss\t%9.4f\n\tinput_match_loss\t%9.4f\toutput_match_loss\t%9.4f\tgen_match_loss\t%9.4f\n\tinput_mask_match_loss\t%9.4f\toutput_mask_match_loss\t%9.4f\timi_mask_match_loss\t%9.4f\n"%(
                    split.upper(), recon_loss, mask_loss, xCritic_loss, zCritic_loss, mCritic_loss,
                    input_match_loss, output_match_loss, gen_match_loss, 
                    input_m_match_loss, output_m_match_loss, gen_m_match_loss))
                file.write("===================================================\n")
    #
    # print the losses for each epoch
    if split == 'train':
        print("Learning rate:\t%2.8f"%(lr_gen.get_last_lr()[0]))
    print("Batch loss:")
    print("%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\tmCritic_loss\t%9.4f\n\tinput_match_loss\t%9.4f\toutput_match_loss\t%9.4f\tgen_match_loss\t%9.4f\n\tinput_mask_match_loss\t%9.4f\toutput_mask_match_loss\t%9.4f\timi_mask_match_loss\t%9.4f"%(
            split.upper(), recon_loss, mask_loss, xCritic_loss, zCritic_loss, mCritic_loss,
            input_match_loss, output_match_loss, gen_match_loss, 
            input_m_match_loss, output_m_match_loss, gen_m_match_loss))
    if split != "train":
        print("Accumulated loss:")
        print("%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\tmCritic_loss\t%9.4f\n\tinput_match_loss\t%9.4f\toutput_match_loss\t%9.4f\tgen_match_loss\t%9.4f\n\tinput_mask_match_loss\t%9.4f\toutput_mask_match_loss\t%9.4f\timi_mask_match_loss\t%9.4f"%(
                split.upper(), recon_total_loss/iteration, mask_total_loss/iteration, xCritic_total_loss/iteration, zCritic_total_loss/iteration, mCritic_total_loss/iteration,
                input_match_total_loss/iteration, output_match_total_loss/iteration, gen_match_total_loss/iteration, 
                input_m_match_total_loss/iteration, output_m_match_total_loss/iteration, gen_m_match_total_loss/iteration))
    print()
    with open(log_file, "a+") as file:
        if split == 'train':
            file.write("Learning rate:\t%2.8f\n"%(lr_gen.get_last_lr()[0]))
        file.write("Batch loss:\n")
        file.write("%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\tmCritic_loss\t%9.4f\n\tinput_match_loss\t%9.4f\toutput_match_loss\t%9.4f\tgen_match_loss\t%9.4f\n\tinput_mask_match_loss\t%9.4f\toutput_mask_match_loss\t%9.4f\timi_mask_match_loss\t%9.4f\n"%(
            split.upper(), recon_loss, mask_loss, xCritic_loss, zCritic_loss, mCritic_loss,
            input_match_loss, output_match_loss, gen_match_loss, 
            input_m_match_loss, output_m_match_loss, gen_m_match_loss))
        if split != "train":
            file.write("Accumulated loss:\n")
            file.write("%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\tmCritic_loss\t%9.4f\n\tinput_match_loss\t%9.4f\toutput_match_loss\t%9.4f\tgen_match_loss\t%9.4f\n\tinput_mask_match_loss\t%9.4f\toutput_mask_match_loss\t%9.4f\timi_mask_match_loss\t%9.4f\n"%(
                split.upper(), recon_total_loss/iteration, mask_total_loss/iteration, xCritic_total_loss/iteration, zCritic_total_loss/iteration, mCritic_total_loss/iteration,
                input_match_total_loss/iteration, output_match_total_loss/iteration, gen_match_total_loss/iteration, 
                input_m_match_total_loss/iteration, output_m_match_total_loss/iteration, gen_m_match_total_loss/iteration))
        file.write("===================================================\n")
    
    if split == 'train':
        lr_enc.step()
        lr_dec.step()
        lr_dx.step()
        lr_dz.step()
        lr_gen.step()
        lr_imi.step()
    
    models = {
        "Trans": Trans,
        "Dx": Dx,
        "G": G,
        "Dz": Dz,
        "Imi": Imi
    }
    
    return recon_total_loss/iteration, models