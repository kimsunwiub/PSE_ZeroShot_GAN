import sys
import pickle
import torchaudio
import torch.nn as nn
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import librosa
import os
from time import time
from datetime import datetime
from argparse import ArgumentParser
import logging
import random
from torch.autograd import grad
from torch.autograd import Variable

import torch
import torch.jit as jit
import torch.nn as nn
from torch.nn import Parameter
from typing import List, Tuple
from torch import Tensor
import math

from data import scale_amplitude, apply_scale_invariance, mix_signals_batch
from se_kd_utils import prep_sig_ml, stft, get_magnitude, calculate_sisdr, denoise_signal, denoise_signal_ctn, loss_sisdr
from models import initialize_weights

eps = 1e-6
logging.getLogger().setLevel(logging.DEBUG)


def setup_gan_expr(args):
    args.n_frames = args.sr * args.duration
    args.stft_features = int(args.fft_size//2+1)
    args.stft_frames = int(np.ceil(args.n_frames/args.hop_size))+1

    t_stamp = '{0:%m%d%H%M}'.format(datetime.now())

    if args.is_g_ctn:
        G_opt = 'CTN'
    else:
        G_opt = '{}x{}'.format(args.G_num_layers, args.G_hidden_size)
    
    if args.load_discriminator:
        D_opt = '{}x{}'.format(args.D_num_layers, args.D_hidden_size)
    else:
        D_opt = 'None'
    
    save_dir = "{}/D{}_G{}".format(args.save_dir, D_opt, G_opt)
    if args.is_anchor:
        save_dir += "_Anc"
    if args.is_adv:
        save_dir = "{}/gx{}/lr{:.0e}/seed{}/snr{}/".format(save_dir, args.g_iter_x, args.learning_rate, args.seed, args.snr_ranges[0])
    
    train_opt = "Pre"
    if args.is_adv:
        train_opt = "Adv"
        
    output_directory = "{}/expr{}_{}_bs{}_nfrm{}_GPU{}".format(
        save_dir, train_opt, t_stamp,
        args.batch_size, args.n_frames,
        args.device)

    print("Output Dir: {}".format(output_directory))
    if args.is_save:
        os.makedirs(output_directory, exist_ok=True)
        print("Created dir...")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [PID %(process)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(os.path.join(output_directory, "training.log")),
                logging.StreamHandler(),
            ],
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [PID %(process)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    return output_directory

# MM GAN
class Discriminator_RNN_BC(nn.Module):
    def __init__(self, hidden_size, num_layers, stft_features, seq_len):
        super(Discriminator_RNN_BC, self).__init__()
        self.rnn = torch.nn.GRU(
            input_size=stft_features, hidden_size=hidden_size, 
            num_layers=num_layers, batch_first=True,
        )
        self.dnn = nn.Linear(hidden_size, 1)
        self.dnn_out1 = nn.Linear(63, 32)
        self.dnn_out2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        initialize_weights(self)

    def forward(self, x, rep=False):
        (batch_size, seq_len, num_features) = x.shape
        x2, hn = self.rnn(x)
        x3 = self.dnn(x2)
        x3 = x3.squeeze(2)
        x4 = self.dnn_out1(x3)
        out = self.sigmoid(self.dnn_out2(x4))
        return out

def adv_pass_bc_gan(args, s, e, D_model, idx, D_optimizer=None):
    ret_vals = []
    S_mag = get_magnitude(stft(s, args.fft_size, args.hop_size)).permute(0,2,1)
    E_mag = get_magnitude(stft(e, args.fft_size, args.hop_size)).permute(0,2,1)

    real_loss = torch.log(D_model(S_mag))
    real_loss = (-real_loss).mean()
    if D_optimizer and idx % args.g_iter_x == 0:
        D_optimizer.zero_grad()
        real_loss.backward()
        D_optimizer.step()
    ret_vals.append(float(real_loss))
    del real_loss
    torch.cuda.empty_cache()
    
    fake_loss = torch.log(1 - D_model(E_mag) + eps)
    fake_loss = (-fake_loss).mean()
    if D_optimizer and idx % args.g_iter_x == 0:
        D_optimizer.zero_grad()
        fake_loss.backward()
        D_optimizer.step()
    ret_vals.append(float(fake_loss))
    del fake_loss
    torch.cuda.empty_cache()

    return ret_vals
    
def temp_checker(args, x, s, e, D_model):
    ret_vals = []
    S_mag = get_magnitude(stft(s, args.fft_size, args.hop_size)).permute(0,2,1)[0].unsqueeze(0)
    X_mag = get_magnitude(stft(x, args.fft_size, args.hop_size)).permute(0,2,1)[0].unsqueeze(0)
    E_mag = get_magnitude(stft(e, args.fft_size, args.hop_size)).permute(0,2,1)[0].unsqueeze(0)

    s_loss = float(D_model(S_mag))
    x_loss = float(D_model(X_mag))
    e_loss = float(D_model(E_mag))
    print("S: {:.2f}, X: {:.2f}, E: {:.2f}".format(s_loss, x_loss, e_loss))
    
# WGAN
class JitGRUCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(JitGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    @jit.script_method
    def forward(self, x, hidden):
        # type: (Tensor, Tensor) -> Tensor
        x = x.view(-1, x.size(1))
        x_results = torch.mm(x, self.weight_ih.t()) + self.bias_ih
        h_results = torch.mm(hidden, self.weight_hh.t()) + self.bias_hh

        i_r, i_z, i_n = x_results.chunk(3, 1)
        h_r, h_z, h_n = h_results.chunk(3, 1)

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)

        return n - torch.mul(n, z) + torch.mul(z, hidden)

class JitGRULayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(JitGRULayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, x, hidden):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        inputs = x.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])

        for i in range(len(inputs)):
            hidden = self.cell(inputs[i], hidden)
            outputs += [hidden]

        return torch.stack(outputs), hidden

class JitGRU(jit.ScriptModule):
    __constants__ = ['hidden_size', 'num_layers', 'batch_first', 'layers']

    def __init__(self, input_size, hidden_size, num_layers, batch_first=False, bias=True):
        super(JitGRU, self).__init__()
        # The following are not implemented.
        assert bias

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        if num_layers == 1:
            self.layers = nn.ModuleList([JitGRULayer(JitGRUCell, input_size, hidden_size)])
        else:
            self.layers = nn.ModuleList([JitGRULayer(JitGRUCell, input_size, hidden_size)] + [JitGRULayer(JitGRUCell, hidden_size, hidden_size)
                                                                                              for _ in range(num_layers - 1)])

    @jit.script_method
    def forward(self, x, h=None):
        # type: (Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        output_states = jit.annotate(List[Tensor], [])

        # Handle batch_first cases
        if self.batch_first:
            x = x.permute(1, 0, 2)

        if h is None:
            h = torch.zeros(self.num_layers, x.shape[1], self.hidden_size, dtype=x.dtype, device=x.device)

        output = x
        i = 0

        for rnn_layer in self.layers:
            output, hidden = rnn_layer(output, h[i])
            output_states += [hidden]
            i += 1

        # Don't forget to handle batch_first cases for the output too!
        if self.batch_first:
            output = output.permute(1, 0, 2)

        return output, torch.stack(output_states)
    
# W GAN
# Gradient Penalty
def grad_norm_sqrd(gradients):
    gp = ((
        torch.max(
            (gradients.norm(2, dim=1)-1),
            torch.tensor([0.]).cuda()
        )
    )**2).mean()
    return gp

def get_GP(args, D_model, real_sample, fake_sample):
    N = len(real_sample)
    alpha = torch.rand((N, 1)).to(args.device)
    x_interp = alpha * real_sample.data + (1-alpha) * fake_sample.data
    X_interp_mag = get_magnitude(stft(x_interp, args.fft_size, args.hop_size)).permute(0,2,1)
    D_in = X_interp_mag
    D_in.requires_grad = True
    
    pred_hat = D_model(D_in)
    gradients = grad(
        outputs=pred_hat, inputs=D_in, 
        grad_outputs=torch.ones(pred_hat.size()).to(args.device),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.reshape(N, -1)
    GP = grad_norm_sqrd(gradients) * 10
    return GP.mean()

class Discriminator_RNN_Utt_Jit(nn.Module):
    def __init__(self, hidden_size, num_layers, stft_features, seq_len):
        super(Discriminator_RNN_Utt_Jit, self).__init__()
        self.rnn = JitGRU(
            input_size=stft_features, hidden_size=hidden_size, 
            num_layers=num_layers, batch_first=True,
        )
        self.dnn = nn.Linear(hidden_size, 1)
        self.dnn_out1 = nn.Linear(63, 32)
        self.dnn_out2 = nn.Linear(32, 16)
        self.dnn_out3 = nn.Linear(16, 1)
        initialize_weights(self)

    def forward(self, x, rep=False):
        (batch_size, seq_len, num_features) = x.shape
        x2, hn = self.rnn(x)
        x3 = self.dnn(x2)
        x3 = x3.squeeze(2)
        x4 = self.dnn_out1(x3)
        x5 = self.dnn_out2(x4)
        out = self.dnn_out3(x5)
        return out

def adv_pass(args, s, e, D_model, idx, D_optimizer=None):
    S = stft(s, args.fft_size, args.hop_size)
    S_mag = get_magnitude(S).permute(0,2,1)
    E = stft(e, args.fft_size, args.hop_size)
    E_mag = get_magnitude(E).permute(0,2,1)
    
    real_loss = -torch.mean(D_model(S_mag))
    fake_loss = torch.mean(D_model(E_mag))
    GP = get_GP(args, D_model, s, e)

    if D_optimizer:
        D_optimizer.zero_grad()                       
        tot_loss = real_loss + fake_loss + GP
        tot_loss.backward()
        D_optimizer.step()

    ret_vals = []
    ret_vals.append(float(real_loss))
    ret_vals.append(float(fake_loss))
    ret_vals.append(float(GP))
    
    del real_loss
    del fake_loss
    del GP
    torch.cuda.empty_cache()

    return ret_vals

# General
def run_gan_iter(args, tot_x, speech_dataloader, G_model, Orig_G_model, D_model, G_optimizer=None, D_optimizer=None):
    upd_res = []
    ori_res = []
    gf_res = []
    r_res = []
    f_res = []
    gp_res = []
    speech_iter = iter(speech_dataloader)
    for idx in range(0,len(tot_x),args.batch_size):
        iter_ = idx//args.batch_size
        try:
            speech_batch = next(speech_iter)
        except StopIteration:
            speech_iter = iter(speech_dataloader)
            speech_batch = next(speech_iter)
        speech_batch = speech_batch.to(args.device)
        mix_batch = tot_x[idx:idx+args.batch_size].to(args.device)
        speech_batch = speech_batch[:len(mix_batch)]
        
        if args.is_g_ctn:
            e = denoise_signal_ctn(args, mix_batch, G_model)
        else:
            e = denoise_signal(args, mix_batch, G_model)

        # Truncate to same lengths
        _, s, _ = prep_sig_ml(speech_batch, e)
        _, x, _ = prep_sig_ml(mix_batch, e)

        # Standardize
        s = s/(s.std(1)[:,None] + eps)
        x = x/(x.std(1)[:,None] + eps)
        e = e/(e.std(1)[:,None] + eps)
        
        if args.is_wgan:
            if args.is_anchor and (iter_+1) % 2 == 0:
                L_r, L_f, gp = adv_pass(args, e, x, D_model, idx, D_optimizer)
            else:
                L_r, L_f, gp = adv_pass(args, s, e, D_model, idx, D_optimizer)
            gp_res.append(gp)
        else:
            if args.is_anchor:
                assert 1==0 # TODO
            L_r, L_f = adv_pass_bc_gan(args, s, e, D_model, idx, D_optimizer)
        r_res.append(L_r)
        f_res.append(L_f)
        assert not torch.isnan(torch.Tensor(gf_res)).any()
            
        del e
            
        # Update Generator
        if args.is_adv:
            if args.is_g_ctn:
                e = denoise_signal_ctn(args, mix_batch, G_model)
            else:
                e = denoise_signal(args, mix_batch, G_model)
            e = e/(e.std(1)[:,None] + eps)
            
            if G_optimizer:
                G_optimizer.zero_grad()
            E_mag = get_magnitude(stft(e, args.fft_size, args.hop_size)).permute(0,2,1)
            if args.is_wgan:
                fake_loss = D_model(E_mag)
            else:
                fake_loss = torch.log(1 - D_model(E_mag) + eps)
            assert not torch.isnan(torch.Tensor(fake_loss.detach().cpu())).any()
            fake_loss = (-fake_loss).mean()
            
            if G_optimizer and (iter_+1) % args.g_iter_x == 0:
                fake_loss.backward()
                G_optimizer.step()
                
            L_gf = float(fake_loss)
            gf_res.append(L_gf)
            del fake_loss
        
            del e
        
    return np.mean(gf_res), np.mean(r_res), np.mean(f_res), np.mean(gp_res)

def run_adv_te_iter(args, tot_s, tot_x, G_model, Orig_G_model):
    upd_res = []
    ori_res = []
    loss_res = []
    for idx in range(0,len(tot_s),args.batch_size):
        speech_batch = tot_s[idx:idx+args.batch_size].to(args.device)
        mix_batch = tot_x[idx:idx+args.batch_size]
        
        if args.is_g_ctn:
            upd_e = []
            ori_e = []
            for x in mix_batch:
                x = x[None,:]
                upd_e_i = denoise_signal_ctn(args, x.to(args.device), G_model).squeeze(1).detach().cpu()
                ori_e_i = denoise_signal_ctn(args, x.to(args.device), Orig_G_model).squeeze(1).detach().cpu()
                upd_e.append(upd_e_i)
                ori_e.append(ori_e_i)
            upd_e = torch.stack(upd_e).squeeze(1)
            ori_e = torch.stack(ori_e).squeeze(1)
        else:
            upd_e = denoise_signal(args, mix_batch.to(args.device), G_model).detach().cpu()
            ori_e = denoise_signal(args, mix_batch.to(args.device), Orig_G_model).detach().cpu()

        # Truncate to same lengths
        _, s, _ = prep_sig_ml(speech_batch, upd_e)
        _, x, _ = prep_sig_ml(mix_batch, upd_e)

        # Standardize
        s = s/(s.std(1)[:,None] + eps)
        x = x/(x.std(1)[:,None] + eps)
        upd_e = upd_e/(upd_e.std(1)[:,None] + eps)
        ori_e = ori_e/(ori_e.std(1)[:,None] + eps)

        upd_sdr = float(calculate_sisdr(s.detach().cpu(), upd_e).mean())
        ori_sdr = float(calculate_sisdr(s.detach().cpu(), ori_e).mean())
        
        upd_res.append(upd_sdr)
        ori_res.append(ori_sdr)

    return np.mean(upd_res), np.mean(ori_res)

def run_baseline_se(args, tot_s, tot_x, G_model, G_optimizer, D_model=None):
    total_loss = []
    for idx in range(0,len(tot_s),args.batch_size):
        speech_batch = tot_s[idx:idx+args.batch_size].to(args.device)
        mix_batch = tot_x[idx:idx+args.batch_size].to(args.device)
        
        G_optimizer.zero_grad()
            
        if args.is_g_ctn:
            e = denoise_signal_ctn(args, mix_batch, G_model)
        else:
            e = denoise_signal(args, mix_batch, G_model)

        # Truncate to same lengths
        _, s, _ = prep_sig_ml(speech_batch, e)
        _, x, _ = prep_sig_ml(mix_batch, e)

        # Standardize
        s = s/(s.std(1)[:,None] + eps)
        x = x/(x.std(1)[:,None] + eps)
        e = e/(e.std(1)[:,None] + eps)

        e_sdr = float(calculate_sisdr(s, e).mean())
        
        if D_model:
            X_mag = get_magnitude(stft(x, args.fft_size, args.hop_size)).permute(0,2,1)[0].unsqueeze(0)
            E_mag = get_magnitude(stft(e, args.fft_size, args.hop_size)).permute(0,2,1)[0].unsqueeze(0)
            loss_i = -(D_model(E_mag) - D_model(X_mag)).mean() # loss_sisdr(s, e, actual_sisdr)
        else:
            actual_sisdr = calculate_sisdr(s, x)
            loss_i = loss_sisdr(s, e, actual_sisdr)
        
        loss_i.backward()
        G_optimizer.step()
        
        del loss_i
            
        loss_i = float(calculate_sisdr(s, e).mean())
        total_loss.append(float(loss_i))
        
        del e
        
    return np.mean(total_loss)