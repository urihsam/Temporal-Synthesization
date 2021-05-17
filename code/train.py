import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from dataset import EHR
from nn.vae import Variational_Autoencoder
from nn.seq2seq_ae import Seq2seq_Autoencoder
from nn.seq2seq_vae import Seq2seq_Variational_Autoencoder
from nn.discriminator import CNN_Discriminator
from nn.gan import MLP_Generator, MLP_Discriminator

from utils.train_utils import sample_start_feature_and_mask
from utils.daae_utils import save_model as save_daae
from utils.daae_utils import load_model as load_daae
from utils.daae_utils import model_evaluation as daae_evaluation

from utils.vae_gan_utils import save_model as save_vae_gan
from utils.vae_gan_utils import load_model as load_vae_gan
from utils.vae_gan_utils import model_evaluation as vae_gan_evaluation

from utils.aae_utils import save_model as save_aae
from utils.aae_utils import load_model as load_aae
from utils.aae_utils import model_evaluation as aae_evaluation

from utils.vae_utils import save_model as save_vae
from utils.vae_utils import load_model as load_vae
from utils.vae_utils import model_evaluation as vae_evaluation



def model_inference(args, AE, zgen, prob_mask):
    # make up start feature
    start_feature, start_mask = sample_start_feature_and_mask(zgen.size(0))
    if args.no_mask:
        Pgen, Mgen = AE.decoder.inference(start_feature=start_feature, start_mask=None, z=zgen)
    elif args.use_prob_mask:
        Pgen, Mgen = AE.decoder.inference(start_feature=start_feature, start_mask=start_mask, prob_mask=prob_mask, z=zgen)
    else:
        Pgen, Mgen = AE.decoder.inference(start_feature=start_feature, start_mask=start_mask, z=zgen)

    return Pgen, Mgen


def train_daae(args, datasets, prob_mask):
    if not args.test:
        # model define
        AE = Seq2seq_Autoencoder(
            max_length=args.max_length,
            rnn_type=args.rnn_type,
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            latent_size=args.latent_size,
            encoder_dropout=args.encoder_dropout,
            decoder_dropout=args.decoder_dropout,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional,
            use_prob_mask=args.use_prob_mask
            )

        Dx = CNN_Discriminator(
            feature_size=args.feature_size,
            feature_dropout=args.feature_dropout,
            filter_size=args.filter_size,
            window_sizes=args.window_sizes,
            )

        G = MLP_Generator(
            input_size=args.noise_size,
            output_size=args.latent_size,
            archs=args.gmlp_archs
            )

        Dz = MLP_Discriminator(
            input_size=args.latent_size*2,
            output_size=1,
            archs=args.dmlp_archs
            ) 

        if torch.cuda.is_available():
            AE = AE.cuda()
            Dx = Dx.cuda()
            G = G.cuda()
            Dz = Dz.cuda()
        
        

        opt_enc = torch.optim.Adam(AE.encoder.parameters(), lr=args.learning_rate)
        opt_dec = torch.optim.Adam(AE.decoder.parameters(), lr=args.learning_rate)
        opt_dix = torch.optim.Adam(Dx.parameters(), lr=args.learning_rate)
        opt_diz = torch.optim.Adam(Dz.parameters(), lr=args.learning_rate)
        opt_gen = torch.optim.Adam(G.parameters(), lr=args.learning_rate)
        #
        lr_enc = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_enc, gamma=args.lr_decay_rate)
        lr_dec = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_dec, gamma=args.lr_decay_rate)
        lr_dix = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_dix, gamma=args.lr_decay_rate)
        lr_diz = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_diz, gamma=args.lr_decay_rate)
        lr_gen = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_gen, gamma=args.lr_decay_rate)

        if args.dp_sgd == True: # ??? why dec, gen?
            import pyvacy
            opt_dec = pyvacy.optim.DPAdam(params=AE.decoder.parameters(), lr=args.learning_rate, batch_size=args.batch_size,
                                        l2_norm_clip=args.l2_norm_clip, noise_multiplier=args.noise_multiplier)
            opt_gen = pyvacy.optim.DPAdam(params=G.parameters(), lr=args.learning_rate, batch_size=args.batch_size,
                                        l2_norm_clip=args.l2_norm_clip, noise_multiplier=args.noise_multiplier)
            epsilon = pyvacy.analysis.moments_accountant(len(datasets['train'].data), args.batch_size, args.noise_multiplier, args.epochs, args.delta)

            print('Training procedure satisfies (%f, %f)-DP' % (2*epsilon, args.delta)) # ?? question, why 2 epsilon?


        tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        models = {
            "AE": AE,
            "Dx": Dx,
            "G": G,
            "Dz": Dz
        }
        opts = {
            "enc": opt_enc,
            "dec": opt_dec,
            "dix": opt_dix,
            "diz": opt_diz,
            "gen": opt_gen
        }
        lrs = {
            "enc": lr_enc,
            "dec": lr_dec,
            "dix": lr_dix,
            "diz": lr_diz,
            "gen": lr_gen
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
            daae_evaluation(args, models, opts, lrs, data_loader, prob_mask, "train", log_file)
        
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
                valid_loss = daae_evaluation(args, models, opts, lrs, data_loader, prob_mask, "valid", log_file)
                print("****************************************************")
                print()
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    path = "{}/daae_vloss_{}".format(args.model_path, valid_loss)
                    min_valid_path = path

                    models = {
                        "AE": AE,
                        "Dx": Dx,
                        "G": G,
                        "Dz": Dz
                    }
                    save_daae(models, path)

            
        # Generate the synthetic sequences as many as you want 
        
        model_path = min_valid_path
    else:
        model_path = os.path.join(args.model_path, args.test_model_filename)
    
    models = load_daae(model_path)
    AE = models["AE"]
    G = models["G"]
    AE.eval()
    G.eval()
    gen_zs, gen_xs, gen_ms = [], [], []
    for i in range(args.gendata_size//args.batch_size):
        zgen = G(batch_size=args.batch_size)
        Pgen, Mgen = model_inference(args, AE, zgen, prob_mask)
        
        gen_zs.append(zgen)
        gen_xs.append(Pgen)
        gen_ms.append(Mgen)

    gen_zlist = torch.cat(gen_zs).cpu().detach().numpy()
    gen_xlist = torch.cat(gen_xs).cpu().detach().numpy()
    gen_mlist = torch.cat(gen_ms).cpu().detach().numpy()
    
    np.save(os.path.join(args.result_path, 'daae_generated_codes.npy'), gen_zlist)
    np.save(os.path.join(args.result_path, 'daae_generated_masks.npy'), gen_mlist)
    np.save(os.path.join(args.result_path, 'daae_generated_patients.npy'), gen_xlist) 


def train_vae_gan(args, datasets, prob_mask):
    if not args.test:
        # model define
        AE = Seq2seq_Variational_Autoencoder(
            max_length=args.max_length,
            rnn_type=args.rnn_type,
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            latent_size=args.latent_size,
            encoder_dropout=args.encoder_dropout,
            decoder_dropout=args.decoder_dropout,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional,
            use_prob_mask=args.use_prob_mask
            )

        Dx = CNN_Discriminator(
            feature_size=args.feature_size,
            feature_dropout=args.feature_dropout,
            filter_size=args.filter_size,
            window_sizes=args.window_sizes,
            )

        if torch.cuda.is_available():
            AE = AE.cuda()
            Dx = Dx.cuda()
        
        

        opt_enc = torch.optim.Adam(AE.encoder.parameters(), lr=args.learning_rate)
        opt_dec = torch.optim.Adam(AE.decoder.parameters(), lr=args.learning_rate)
        opt_dix = torch.optim.Adam(Dx.parameters(), lr=args.learning_rate)
        #
        lr_enc = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_enc, gamma=args.lr_decay_rate)
        lr_dec = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_dec, gamma=args.lr_decay_rate)
        lr_dix = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_dix, gamma=args.lr_decay_rate)

        if args.dp_sgd == True: # ??? why dec, gen?
            import pyvacy
            opt_dec = pyvacy.optim.DPAdam(params=AE.decoder.parameters(), lr=args.learning_rate, batch_size=args.batch_size,
                                        l2_norm_clip=args.l2_norm_clip, noise_multiplier=args.noise_multiplier)
            epsilon = pyvacy.analysis.moments_accountant(len(datasets['train'].data), args.batch_size, args.noise_multiplier, args.epochs, args.delta)

            print('Training procedure satisfies (%f, %f)-DP' % (2*epsilon, args.delta)) # ?? question, why 2 epsilon?


        tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        models = {
            "AE": AE,
            "Dx": Dx
        }
        opts = {
            "enc": opt_enc,
            "dec": opt_dec,
            "dix": opt_dix
        }
        lrs = {
            "enc": lr_enc,
            "dec": lr_dec,
            "dix": lr_dix
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
            vae_gan_evaluation(args, models, opts, lrs, data_loader, prob_mask, "train", log_file)
        
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
                valid_loss = vae_gan_evaluation(args, models, opts, lrs, data_loader, prob_mask, "valid", log_file)
                print("****************************************************")
                print()
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    path = "{}/vae_gan_vloss_{}".format(args.model_path, valid_loss)
                    min_valid_path = path

                    models = {
                        "AE": AE,
                        "Dx": Dx
                    }
                    save_vae_gan(models, path)

            
        # Generate the synthetic sequences as many as you want 
        model_path = min_valid_path
    else:
        model_path = os.path.join(args.model_path, args.test_model_filename)
    
    models = load_vae_gan(model_path)
    AE = models["AE"]
    AE.eval()
    gen_zs, gen_xs, gen_ms = [], [], []
    for i in range(args.gendata_size//args.batch_size):
        zgen = torch.randn((args.batch_size, args.latent_size))
        Pgen, Mgen = model_inference(args, AE, zgen, prob_mask)
        
        gen_zs.append(zgen)
        gen_xs.append(Pgen)
        gen_ms.append(Mgen)

    gen_zlist = torch.cat(gen_zs).cpu().detach().numpy()
    gen_xlist = torch.cat(gen_xs).cpu().detach().numpy()
    gen_mlist = torch.cat(gen_ms).cpu().detach().numpy()
    
    np.save(os.path.join(args.result_path, 'daae_generated_codes.npy'), gen_zlist)
    np.save(os.path.join(args.result_path, 'daae_generated_masks.npy'), gen_mlist)
    np.save(os.path.join(args.result_path, 'daae_generated_patients.npy'), gen_xlist) 


def train_aae(args, datasets, prob_mask):
    if not args.test:
        # model define
        AE = Seq2seq_Autoencoder(
            max_length=args.max_length,
            rnn_type=args.rnn_type,
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            latent_size=args.latent_size,
            encoder_dropout=args.encoder_dropout,
            decoder_dropout=args.decoder_dropout,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional,
            use_prob_mask=args.use_prob_mask
            )

        G = MLP_Generator(
            input_size=args.noise_size,
            output_size=args.latent_size,
            archs=args.gmlp_archs
            )

        Dz = MLP_Discriminator(
            input_size=args.latent_size*2,
            output_size=1,
            archs=args.dmlp_archs
            ) 

        if torch.cuda.is_available():
            AE = AE.cuda()
            G = G.cuda()
            Dz = Dz.cuda()
        
        

        opt_enc = torch.optim.Adam(AE.encoder.parameters(), lr=args.learning_rate)
        opt_dec = torch.optim.Adam(AE.decoder.parameters(), lr=args.learning_rate)
        opt_diz = torch.optim.Adam(Dz.parameters(), lr=args.learning_rate)
        opt_gen = torch.optim.Adam(G.parameters(), lr=args.learning_rate)
        #
        lr_enc = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_enc, gamma=args.lr_decay_rate)
        lr_dec = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_dec, gamma=args.lr_decay_rate)
        lr_diz = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_diz, gamma=args.lr_decay_rate)
        lr_gen = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_gen, gamma=args.lr_decay_rate)

        if args.dp_sgd == True: # ??? why dec, gen?
            import pyvacy
            opt_dec = pyvacy.optim.DPAdam(params=AE.decoder.parameters(), lr=args.learning_rate, batch_size=args.batch_size,
                                        l2_norm_clip=args.l2_norm_clip, noise_multiplier=args.noise_multiplier)
            opt_gen = pyvacy.optim.DPAdam(params=G.parameters(), lr=args.learning_rate, batch_size=args.batch_size,
                                        l2_norm_clip=args.l2_norm_clip, noise_multiplier=args.noise_multiplier)
            epsilon = pyvacy.analysis.moments_accountant(len(datasets['train'].data), args.batch_size, args.noise_multiplier, args.epochs, args.delta)

            print('Training procedure satisfies (%f, %f)-DP' % (2*epsilon, args.delta)) # ?? question, why 2 epsilon?


        tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        models = {
            "AE": AE,
            "G": G,
            "Dz": Dz
        }
        opts = {
            "enc": opt_enc,
            "dec": opt_dec,
            "diz": opt_diz,
            "gen": opt_gen
        }
        lrs = {
            "enc": lr_enc,
            "dec": lr_dec,
            "diz": lr_diz,
            "gen": lr_gen
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
            aae_evaluation(args, models, opts, lrs, data_loader, prob_mask, "train", log_file)
        
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
                valid_loss = aae_evaluation(args, models, opts, lrs, data_loader, prob_mask, "valid", log_file)
                print("****************************************************")
                print()
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    path = "{}/aae_vloss_{}".format(args.model_path, valid_loss)
                    min_valid_path = path

                    models = {
                        "AE": AE,
                        "G": G,
                        "Dz": Dz
                    }
                    save_aae(models, path)

            
        # Generate the synthetic sequences as many as you want 
        model_path = min_valid_path
    else:
        model_path = os.path.join(args.model_path, args.test_model_filename)
    
    models = load_aae(model_path)
    AE = models["AE"]
    G = models["G"]
    AE.eval()
    G.eval()
    gen_zs, gen_xs, gen_ms = [], [], []
    for i in range(args.gendata_size//args.batch_size):
        zgen = G(batch_size=args.batch_size)
        Pgen, Mgen = model_inference(args, AE, zgen, prob_mask)
        
        gen_zs.append(zgen)
        gen_xs.append(Pgen)
        gen_ms.append(Mgen)

    gen_zlist = torch.cat(gen_zs).cpu().detach().numpy()
    gen_xlist = torch.cat(gen_xs).cpu().detach().numpy()
    gen_mlist = torch.cat(gen_ms).cpu().detach().numpy()
    
    np.save(os.path.join(args.result_path, 'daae_generated_codes.npy'), gen_zlist)
    np.save(os.path.join(args.result_path, 'daae_generated_masks.npy'), gen_mlist)
    np.save(os.path.join(args.result_path, 'daae_generated_patients.npy'), gen_xlist) 


def train_seq2seq_vae(args, datasets, prob_mask):
    if not args.test:
        # model define
        AE = Seq2seq_Variatonal_Autoencoder(
            max_length=args.max_length,
            rnn_type=args.rnn_type,
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            latent_size=args.latent_size,
            encoder_dropout=args.encoder_dropout,
            decoder_dropout=args.decoder_dropout,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional,
            use_prob_mask=args.use_prob_mask
            )

        if torch.cuda.is_available():
            AE = AE.cuda()
        
        

        opt_enc = torch.optim.Adam(AE.encoder.parameters(), lr=args.learning_rate)
        opt_dec = torch.optim.Adam(AE.decoder.parameters(), lr=args.learning_rate)
        #
        lr_enc = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_enc, gamma=args.lr_decay_rate)
        lr_dec = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_dec, gamma=args.lr_decay_rate)

        if args.dp_sgd == True: # ??
            import pyvacy
            opt_dec = pyvacy.optim.DPAdam(params=AE.decoder.parameters(), lr=args.learning_rate, batch_size=args.batch_size,
                                        l2_norm_clip=args.l2_norm_clip, noise_multiplier=args.noise_multiplier)
            
            epsilon = pyvacy.analysis.moments_accountant(len(datasets['train'].data), args.batch_size, args.noise_multiplier, args.epochs, args.delta)

            print('Training procedure satisfies (%f, %f)-DP' % (2*epsilon, args.delta)) # ?? question, why 2 epsilon?


        tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        models = {
            "AE": AE
        }
        opts = {
            "enc": opt_enc,
            "dec": opt_dec
        }
        lrs = {
            "enc": lr_enc,
            "dec": lr_dec
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
            vae_evaluation(args, models, opts, lrs, data_loader, prob_mask, "train", log_file)
        
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
                valid_loss = vae_evaluation(args, models, opts, lrs, data_loader, prob_mask, "valid", log_file)
                print("****************************************************")
                print()
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    path = "{}/seq2seq_vae_vloss_{}".format(args.model_path, valid_loss)
                    min_valid_path = path

                    models = {
                        "AE": AE
                    }
                    save_vae(models, path)

            
        # Generate the synthetic sequences as many as you want 
        model_path = min_valid_path
    else:
        model_path = os.path.join(args.model_path, args.test_model_filename)
    
    models = load_vae(model_path)
    AE = models["AE"]
    AE.eval()
    
    gen_zs, gen_xs, gen_ms = [], [], []
    for i in range(args.gendata_size//args.batch_size):
        zgen = torch.randn((args.batch_size, args.latent_size))
        Pgen, Mgen = model_inference(args, AE, zgen, prob_mask)
        
        gen_zs.append(zgen)
        gen_xs.append(Pgen)
        gen_ms.append(Mgen)

    gen_zlist = torch.cat(gen_zs).cpu().detach().numpy()
    gen_xlist = torch.cat(gen_xs).cpu().detach().numpy()
    gen_mlist = torch.cat(gen_ms).cpu().detach().numpy()
    
    np.save(os.path.join(args.result_path, 'daae_generated_codes.npy'), gen_zlist)
    np.save(os.path.join(args.result_path, 'daae_generated_masks.npy'), gen_mlist)
    np.save(os.path.join(args.result_path, 'daae_generated_patients.npy'), gen_xlist) 
    


def train_vae(args, datasets, prob_mask):
    if not args.test:
        # model define
        AE = Variational_Autoencoder(
            max_length=args.max_length,
            rnn_type=args.rnn_type,
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            latent_size=args.latent_size,
            encoder_dropout=args.encoder_dropout,
            decoder_dropout=args.decoder_dropout,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional,
            no_mask=args.no_mask or args.use_prob_mask
            )

        if torch.cuda.is_available():
            AE = AE.cuda()
        
        

        opt_enc = torch.optim.Adam(AE.encoder.parameters(), lr=args.learning_rate)
        opt_dec = torch.optim.Adam(AE.decoder.parameters(), lr=args.learning_rate)
        #
        lr_enc = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_enc, gamma=args.lr_decay_rate)
        lr_dec = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_dec, gamma=args.lr_decay_rate)

        if args.dp_sgd == True: # ??
            import pyvacy
            opt_dec = pyvacy.optim.DPAdam(params=AE.decoder.parameters(), lr=args.learning_rate, batch_size=args.batch_size,
                                        l2_norm_clip=args.l2_norm_clip, noise_multiplier=args.noise_multiplier)
            
            epsilon = pyvacy.analysis.moments_accountant(len(datasets['train'].data), args.batch_size, args.noise_multiplier, args.epochs, args.delta)

            print('Training procedure satisfies (%f, %f)-DP' % (2*epsilon, args.delta)) # ?? question, why 2 epsilon?


        tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        models = {
            "AE": AE
        }
        opts = {
            "enc": opt_enc,
            "dec": opt_dec
        }
        lrs = {
            "enc": lr_enc,
            "dec": lr_dec
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
            vae_evaluation(args, models, opts, lrs, data_loader, prob_mask, "train", log_file)
        
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
                valid_loss = vae_evaluation(args, models, opts, lrs, data_loader, prob_mask, "valid", log_file)
                print("****************************************************")
                print()
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    path = "{}/vae_vloss_{}".format(args.model_path, valid_loss)
                    min_valid_path = path

                    models = {
                        "AE": AE
                    }
                    save_vae(models, path)

            
        # Generate the synthetic sequences as many as you want 
        model_path = min_valid_path
    else:
        model_path = os.path.join(args.model_path, args.test_model_filename)
    
    models = load_vae(model_path)
    AE = models["AE"]
    AE.eval()
    
    gen_zs, gen_xs, gen_ms = [], [], []
    for i in range(args.gendata_size//args.batch_size):
        zgen = torch.randn((args.batch_size, args.latent_size))
        Pgen, Mgen = model_inference(args, AE, zgen, prob_mask)
        
        gen_zs.append(zgen)
        gen_xs.append(Pgen)
        gen_ms.append(Mgen)

    gen_zlist = torch.cat(gen_zs).cpu().detach().numpy()
    gen_xlist = torch.cat(gen_xs).cpu().detach().numpy()
    gen_mlist = torch.cat(gen_ms).cpu().detach().numpy()
    
    np.save(os.path.join(args.result_path, 'daae_generated_codes.npy'), gen_zlist)
    np.save(os.path.join(args.result_path, 'daae_generated_masks.npy'), gen_mlist)
    np.save(os.path.join(args.result_path, 'daae_generated_patients.npy'), gen_xlist) 


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
    
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, default='daae')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--test_model_filename', type=str, default="")
    parser.add_argument('--train_log', type=str, default='train_log_file')
    parser.add_argument('--valid_log', type=str, default='valid_log_file')
    parser.add_argument('--test_log', type=str, default='test_log_file')
    parser.add_argument('--model_path', type=str, default='models')
    parser.add_argument('--result_path', type=str, default='results')
    parser.add_argument('--max_length', type=int, default=40)
    parser.add_argument('--train_eval_freq', type=int, default=50)
    parser.add_argument('--valid_eval_freq', type=int, default=1)
    parser.add_argument('--critic_freq_base', type=int, default=5)
    parser.add_argument('--critic_freq_hit', type=int, default=1)
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
