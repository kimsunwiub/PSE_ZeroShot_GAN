import os
from argparse import ArgumentParser
import logging

import torch
torch.set_num_threads(1)

import torch.nn as nn

from data import *
from models import *
from se_kd_utils import *
from utils import *
from gan_utils import *

def parse_arguments():
    parser = ArgumentParser()    
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-e", "--tot_epoch", type=int, default=200)
    parser.add_argument("-r", "--G_num_layers", type=int, default=-1)
    parser.add_argument("-m", "--D_num_layers", type=int, default=-1)
    parser.add_argument("-g", "--G_hidden_size", type=int, default=-1)
    parser.add_argument("-d", "--D_hidden_size", type=int, default=-1)
    
    parser.add_argument("--data_dir", type=str, default="/home/kimsunw/data/")
    parser.add_argument("--save_dir", type=str, default="/home/kimsunw/workspace/pse/gan_adv_models/")
    
    parser.add_argument("--load_SEmodel", type=str, default=None)
    parser.add_argument("--load_SErundata", type=str, default=None)
    parser.add_argument('--load_discriminator', type=str, default=None)
    parser.add_argument('--load_D_rundata', type=str, default=None)
    
    parser.add_argument("--snr_ranges", nargs='+', type=int, default=[-5,10])
    
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--validate_every", type=int, default=2)
    
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--duration", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--fft_size", type=int, default=1024)
    parser.add_argument("--hop_size", type=int, default=256)
    
    parser.add_argument('--is_save', action='store_true')
    parser.add_argument('--is_si', action='store_true')
        
    parser.add_argument('--is_g_ctn', action='store_true')
    parser.add_argument('--is_wgan', action='store_true')
    parser.add_argument("--g_iter_x", type=int, default=1)
    parser.add_argument('--is_anchor', action='store_true')
        
    return parser.parse_args()
      
args = parse_arguments()
args.is_adv = True
logging.getLogger().setLevel(logging.INFO)
args.is_train_kd = True
args.is_train_disc = False
args.stft_features = int(args.fft_size//2+1)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.device)
args.device = 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Data
tr_speech_ds = torchaudio.datasets.LIBRISPEECH("{}/".format(args.data_dir), url="train-clean-100", download=True)
va_speech_ds = torchaudio.datasets.LIBRISPEECH("{}/".format(args.data_dir), url="dev-clean", download=True)
kwargs = {'num_workers': 0, 'pin_memory': True, 'drop_last': True}
tr_speech_dataloader = data.DataLoader(dataset=tr_speech_ds,
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn= lambda x: data_processing(x, args.n_frames, "speech"),
                            **kwargs)
va_speech_dataloader = data.DataLoader(dataset=va_speech_ds,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn= lambda x: data_processing(x, args.n_frames, "speech"),
                            **kwargs)

snr_ranges_all = [-5,0,5,10]
loss_fn = nn.MSELoss()
for seed in range(40):
    args.seed = seed
    print("Running for seed ", args.seed)
        
    # Load baseline original student model
    if args.is_g_ctn:
        from asteroid.models import ConvTasNet
        Orig_G_model = ConvTasNet(n_src=1)
        ctn_dir = "new_models_results/Gs/expr04221819_SE_G3x1024_lr5e-04_bs20_ctnTruesm-1_nfrms16000_GPU1/"
        Orig_G_model.load_state_dict(torch.load("{}/Dmodel_best.pt".format(ctn_dir)))
    else:
        Orig_G_model = SpeechEnhancementModel(
            args.G_hidden_size, args.G_num_layers, args.stft_features)
        load_model(Orig_G_model, args.load_SEmodel)
    Orig_G_model = Orig_G_model.to(args.device)

    for snr_ranges in snr_ranges_all:
        args.snr_ranges = [snr_ranges]
        output_directory = setup_gan_expr(args)

        tot_s, tot_n = init_pers_set(args)
        tr_s, tr_n, va_s, va_n, te_s, te_n = mixup(args, tot_s, tot_n)

        va_x = mix_signals_batch(va_s, va_n, args.snr_ranges)
        te_x = mix_signals_batch(te_s, te_n, args.snr_ranges)

        # Load student model for personalization
        if args.is_g_ctn:
            from asteroid.models import ConvTasNet
            G_model = ConvTasNet(n_src=1)
            ctn_dir = "new_models_results/Gs/expr04221819_SE_G3x1024_lr5e-04_bs20_ctnTruesm-1_nfrms16000_GPU1/"
            G_model.load_state_dict(torch.load("{}/Dmodel_best.pt".format(ctn_dir)))
        else:
            G_model = SpeechEnhancementModel(
                args.G_hidden_size, args.G_num_layers, args.stft_features)
            load_model(G_model, args.load_SEmodel)
        G_model = G_model.to(args.device)
        G_optimizer = torch.optim.Adam(params=G_model.parameters(),lr=args.learning_rate)

        if args.is_wgan:
            D_model = Discriminator_RNN_Utt_Jit(
                args.D_hidden_size, args.D_num_layers, args.stft_features,
                args.stft_frames)
        else:
            D_model = Discriminator_RNN_BC(
                args.D_hidden_size, args.D_num_layers, args.stft_features,
                args.stft_frames)
        D_optimizer = torch.optim.Adam(
            params=D_model.parameters(),
            lr=args.learning_rate
        )
        if args.load_discriminator:
            load_model(D_model, args.load_discriminator)
        D_model = D_model.to(args.device)

        tr_losses_gf = []
        tr_losses_r = []
        tr_losses_f = []
        tr_losses_gp = []
        va_losses_gf = []
        va_losses_r = []
        va_losses_f = []
        va_losses_gp = []
        te_losses_gf = []
        te_losses_r = []
        te_losses_f = []
        te_losses_gp = []

        tr_upd_sdr = []
        tr_ori_sdr = []
        va_upd_sdr = []
        va_ori_sdr = []
        te_upd_sdr = []
        te_ori_sdr = []
        
        prev_te_p = -100
        
        tr_speech_iter = iter(tr_speech_dataloader)
        va_speech_iter = iter(va_speech_dataloader)
        
        for ep in range(args.tot_epoch):
            # Train
            tr_s = shuffle_set(tr_s)
            tr_x = mix_signals_batch(tr_s, shuffle_set(tr_n)[:len(tr_s)], args.snr_ranges)
            tr_len = len(tr_x)
            tr_s = tr_s[:tr_len-(tr_len%args.batch_size)]
            tr_x = tr_x[:tr_len-(tr_len%args.batch_size)]

            L_gf, L_r, L_f, gp = run_gan_iter(
                args, tr_x, tr_speech_iter, G_model, Orig_G_model, D_model, G_optimizer, D_optimizer)
            upd_sdr, ori_sdr = run_adv_te_iter(args, tr_s, tr_x, G_model, Orig_G_model)
            tr_upd_sdr.append(upd_sdr)
            tr_ori_sdr.append(ori_sdr)
            tr_losses_gf.append(L_gf)
            tr_losses_r.append(L_r)
            tr_losses_f.append(L_f)
            tr_losses_gp.append(gp)

            # Eval
            L_gf, L_r, L_f, gp = run_gan_iter(
                args, va_x, va_speech_iter, G_model, Orig_G_model, D_model)
            upd_sdr, ori_sdr = run_adv_te_iter(args, va_s, va_x, G_model, Orig_G_model)
            va_upd_sdr.append(upd_sdr)
            va_ori_sdr.append(ori_sdr)
            va_losses_gf.append(L_gf)
            va_losses_r.append(L_r)
            va_losses_f.append(L_f)
            va_losses_gp.append(gp)

            # Test
            upd_sdr, ori_sdr = run_adv_te_iter(args, te_s, te_x, G_model, Orig_G_model)
            te_upd_sdr.append(upd_sdr)
            te_ori_sdr.append(ori_sdr)

            if (ep+1) % args.print_every == 0:
                if args.is_anchor:
                    print("Anchor")
                logging.info("Epoch {} Training. USDR: {:.2f} | OSDR: {:.2f}. Losses. G: {:.2f} | Real: {:.2f} | Fake: {:.2f}".format(
                    ep, 
                    tr_upd_sdr[-1],
                    tr_ori_sdr[-1],
                    tr_losses_gf[-1],
                    tr_losses_r[-1],
                    tr_losses_f[-1],
                ))
                logging.info("Epoch {} Validation. USDR: {:.2f} | OSDR: {:.2f}. Losses. G: {:.2f} | Real: {:.2f} | Fake: {:.2f}".format(
                    ep, 
                    va_upd_sdr[-1],
                    va_ori_sdr[-1],
                    va_losses_gf[-1],
                    va_losses_r[-1],
                    va_losses_f[-1],
                ))
                
                if args.is_wgan:
                    logging.info("GP. Tr: {:.2f}, Va:{:.2f}".format(tr_losses_gp[-1], va_losses_gp[-1]))
                if args.is_anchor:
                    print("Anchor")
                logging.info("Epoch {} Testing. USDR: {:.2f} | OSDR: {:.2f}".format(
                    ep, 
                    te_upd_sdr[-1],
                    te_ori_sdr[-1],
                ))
            
#             if ep > 10:
#                 curr_te_p = np.mean(te_upd_sdr[-5:])
#                 if (curr_te_p - prev_te_p) < 0:
#                     logging.info("Epoch {} Exiting.".format(ep))
#                     break
#                 prev_te_p = curr_te_p

        seed_dict = {
            "tr_losses_gf": tr_losses_gf,
            "tr_losses_r": tr_losses_r,
            "tr_losses_f": tr_losses_f,
            "tr_losses_gp": tr_losses_gp,
            "va_losses_gf": va_losses_gf,
            "va_losses_r": va_losses_r,
            "va_losses_f": va_losses_f,
            "va_losses_gp": va_losses_gp,
            
            "tr_upd_sdr": tr_upd_sdr,
            "tr_ori_sdr": tr_ori_sdr,
            "va_upd_sdr": va_upd_sdr,
            "va_ori_sdr": va_ori_sdr,
            "te_upd_sdr": te_upd_sdr,
            "te_ori_sdr": te_ori_sdr,
            
            "epoch": 666,
        }

        if args.is_save:
            save_model(D_model, output_directory, seed_dict, is_last=True, model2=G_model)
            
logging.info("Finished KD")