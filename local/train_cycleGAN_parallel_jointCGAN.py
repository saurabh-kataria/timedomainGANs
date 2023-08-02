#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''unpaired learning
TODO:
    1. add backuping of log file
NOTES:
    1. use python3.9
'''
EPS=1e-9
LARGE_NUM=1e10

import os
import sys
for path in ['.', 'hyperion']:
    assert os.path.exists(path)
    sys.path.append(path)
import numpy as np
import argparse
import time
import wandb
import math
import glob
import subprocess
import distutils
import bitsandbytes as bnb
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.cuda.amp import autocast, GradScaler
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as PowerSGD

from local.supp_fxns import *
from local.pytorch_fxns import *
from local.models_emonet import Emo_Raw_TDNN

from hyperion.torch.data import AudioDataset as AD
from hyperion.torch.data import AudioDataset2 as AD2
from hyperion.torch.data import ClassWeightedSeqSampler as Sampler
from hyperion.torch.utils import open_device
from hyperion.torch.utils import ddp, TorchDDP

from denoiser.demucs import Demucs
from denoiser.stft_loss import MultiResolutionSTFTLoss
#rom vits.models import Generator as vitsGenerator
from conv_tasnet.conv_tasnet import TasNet
from conv_tasnet2.src.conv_tasnet import ConvTasNet as TasNet2
from conv_tasnet3.model import ConvTasNet as TasNet3
from conv_tasnet4.model import ConvTasNet as TasNet4
from DPTNet.models import DPTNet_base
from DNN_based_source_separation.src.models.dprnn_tasnet import DPRNNTasNet
from DNN_based_source_separation.src.models.dptnet import DPTNet
from asteroid.models import DPTNet as DPTNet3
from parallel_wavegan.models.parallel_wavegan import ParallelWaveGANGenerator, ParallelWaveGANDiscriminator, ResidualParallelWaveGANDiscriminator
from parallel_wavegan.models.melgan import MelGANDiscriminator, MelGANMultiScaleDiscriminator
from parallel_wavegan.models.hifigan import HiFiGANScaleDiscriminator, HiFiGANPeriodDiscriminator, HiFiGANMultiScaleDiscriminator, HiFiGANMultiPeriodDiscriminator, HiFiGANMultiScaleMultiPeriodDiscriminator
from parallel_wavegan.models.style_melgan import StyleMelGANGenerator, StyleMelGANDiscriminator

from sats.optims.ranger2020 import Ranger


def get_generator(rank=0):
    if class_generator == 'convtasnet':
        G = TasNet(num_spk=1, layer=ctn_layer, enc_dim=ctn_enc_dim, stack=1, kernel=ctn_kernel, win=1, TCN_dilationFactor=ctn_TCN_dilationFactor, feature_dim=ctn_feature_dim, masks_type='mul', audio_scale=audio_scale, masking_nonlinearity='sigmoid', support_noise=G_support_noise, dim_noise=dim_noise, std_noise=std_noise)
    elif class_generator == 'demucs':
        G = Demucs(causal=False, hidden=15, device=device)
    else:
        raise NotImplementedError(f'{class_generator=}')
    out = [G]
    if add_noise_to_G_input:
        out = [GaussianNoise(sigma=GaussianNoise_sigma)] + out
    if len(out) > 1:
        G = nn.Sequential(*out)
    else:
        G = out[0]
    if rank == 0:
        print(G)
        print(f'{get_ntrainableparams_nn(G)=}')
    return G


def get_discriminator(rank=0):
    if class_discriminator == 'ParallelWaveGANDiscriminator':
        D = ParallelWaveGANDiscriminator(conv_channels=pwg_disc_conv_channels, layers=pwg_layers)
    elif class_discriminator == 'ResidualParallelWaveGANDiscriminator':
        D = ResidualParallelWaveGANDiscriminator(layers=pwg_layers, stacks=pwg_layers//10)
    elif class_discriminator == 'MelGANDiscriminator':
        D = MelGANDiscriminator()
    elif class_discriminator == 'MelGANMultiScaleDiscriminator':
        D = MelGANMultiScaleDiscriminator()
    elif class_discriminator == 'HiFiGANPeriodDiscriminator':
        D = HiFiGANPeriodDiscriminator(out_chs_multiplier=hifi_D_out_chs_multiplier, channels=hifi_D_channels)
    elif class_discriminator == 'HiFiGANMultiPeriodDiscriminator':
        discriminator_params = HiFiGANMultiPeriodDiscriminator().discriminator_params
        discriminator_params['channels'] = hifi_D_channels
        D = HiFiGANMultiPeriodDiscriminator(discriminator_params=discriminator_params)
    elif class_discriminator == 'HiFiGANScaleDiscriminator':
        D = HiFiGANScaleDiscriminator(channels=hifi_D_channels)
    elif class_discriminator == 'HiFiGANMultiScaleDiscriminator':
        discriminator_params = HiFiGANMultiScaleDiscriminator().discriminator_params
        discriminator_params['channels'] = hifi_D_channels
        D = HiFiGANMultiScaleDiscriminator(discriminator_params=discriminator_params)
    elif class_discriminator == 'HiFiGANMultiScaleMultiPeriodDiscriminator':
        scale_discriminator_params = HiFiGANMultiScaleMultiPeriodDiscriminator().scale_discriminator_params
        scale_discriminator_params['channels'] = hifi_D_scale_channels
        period_discriminator_params = HiFiGANMultiScaleMultiPeriodDiscriminator().period_discriminator_params
        period_discriminator_params['channels'] = hifi_D_period_channels
        D = HiFiGANMultiScaleMultiPeriodDiscriminator(scale_discriminator_params=scale_discriminator_params,period_discriminator_params=period_discriminator_params)
    elif class_discriminator == 'StyleMelGANDiscriminator':
        D = StyleMelGANDiscriminator()
    else:
        raise NotImplementedError(f'{class_discriminator=}')
    out = [D]
    if add_noise_to_D_input:
        out = [GaussianNoise(sigma=GaussianNoise_sigma)] + out
    if append_sigmoid_to_discriminator:
        out = out + [nn.Sigmoid(sigma=GaussianNoise_sigma)]
    if len(out) > 1:
        D = nn.Sequential(*out)
    else:
        D = out[0]
    if rank == 0:
        print(D)
        print(f'{get_ntrainableparams_nn(D)=}')
    return D


def torch_calc_MSE(x, t):
    def _torch_calc_MSE(x, t):
        return torch.mean((x - t)**2)
    if isinstance(x, list):
        assert isinstance(x[0], torch.Tensor)   # only lists are allowed
        return torch.mean(torch.stack([_torch_calc_MSE(TMP,t) for TMP in x]))
    else:
        assert isinstance(x, torch.Tensor)
        return _torch_calc_MSE(x,t)


def get_criterion():
    if metric_criterion == 'l1':
        c = nn.L1Loss()
    elif metric_criterion == 'l2':
        c = nn.MSELoss()
    elif metric_criterion == 'mrstft':
        c = MultiResolutionSTFTLoss(factor_sc=0.5,factor_mag=0.5)
    else:
        raise NotImplementedError(f'{metric_criterion=}')
    return c


def model_for_ddp(model):
    if num_gpus > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #    model = TorchDDP(model, device_ids=[device], output_device=device)
        model = DDP(model, find_unused_parameters=True, device_ids=[device], output_device=device)
        if use_PowerSGD:
            state = PowerSGD.PowerSGDState(
              process_group=None, 
              matrix_approximation_rank=1,
              start_powerSGD_iter=3,
            )
            model.register_comm_hook(state, PowerSGD.powerSGD_hook)
    return model


def get_learnable_loss_wrapper_fxn():
    learnable_loss_wrapper_fxn_dict = {'softplus': nn.Softplus(), 'relu': nn.ReLU(), 'identity': nn.Identity()}
    return learnable_loss_wrapper_fxn_dict[learnable_loss_wrapper_fxn]  # get new instance every time this is called


def torch_calc_error(x, t, metric):
    'asymmetric function, t:target'
    def _torch_calc_error(x, t, metric):
        if isinstance(x, torch.Tensor) and t_is_number: # t is int or float or torch.tensor (of dimension 0)
            t = t * torch.ones_like(x)
        elif isinstance(x, torch.Tensor) and isinstance(t, torch.Tensor):
            assert x.shape == t.shape
        else:
            raise Exception('types not understood')
        return metric(x, t)

    t_is_number = isinstance(t, (int,float)) or (isinstance(t, torch.Tensor) and len(t.shape)==0)
    if isinstance(x, torch.Tensor):
        return _torch_calc_error(x, t, metric)
    elif isinstance(x, list):
        if isinstance(x[0], list):  # list of list
            assert not isinstance(x[0][0], list), 'x is deeper than list of list; not supported'
            D = len(x)  # how many lists
            if t_is_number:
                return torch.mean(torch.stack([torch_calc_error(x[_],t,metric) for _ in range(D)]))
            else:
                assert len(t) == D
                return torch.mean(torch.stack([torch_calc_error(x[_],t[_],metric) for _ in range(D)]))
        else:
            D = len(x)
            if t_is_number:
                return torch.mean(torch.stack([_torch_calc_error(x[_], t*torch.ones_like(x[_]), metric) for _ in range(D)]))
            else:
                return torch.mean(torch.stack([_torch_calc_error(x[_], t[_], metric) for _ in range(D)]))
    else:
        raise NotImplementedError(f'not supported {type(x)=}')


def get_B2A_preproc():
    m = TasNet(num_spk=1, layer=8, enc_dim=128, stack=1, kernel=3, win=1, TCN_dilationFactor=2)
    filepattern = os.path.join(B2A_preproc_dir, f'*.pt')
    states = glob.glob(filepattern)
    assert len(states) > 0, f'{B2A_preproc_dir=}'
    B2A_preproc_path = subprocess.run(f'ls -1tv {filepattern}', shell=True, check=True, stdout=subprocess.PIPE).stdout.decode("UTF-8").split()[-1]
    state_dict = torch.load(B2A_preproc_path, map_location=device_cpu)['G_B2A']
    for key in list(state_dict):
        if key.startswith('module.'):
            state_dict['.'.join(key.split('.')[1:])] = state_dict.pop(key)
    m.load_state_dict(state_dict)
    m.eval()
    freeze_nn(m)
    return m


class BWEtrainer_CycleGAN:
    def __init__(self):
        #
        if num_gpus > 1:
            self.rank = dist.get_rank()
        else:
            self.rank = 0
        # mark global variables
        global disable_wandb
        if not disable_wandb and self.rank == 0:
            try:
                __spec__ = None
                wandb.login()
                self.wandb_run = wandb.init(project=projectID, name=experimentID, config=dict_args, settings=wandb.Settings(start_method="fork"))
            except Exception as e:
                print(e)
                disable_wandb = True
                warn('wandb: initialization FAILED')
        # models
        self.G_A2B = get_generator(rank=self.rank).to(device)
        self.G_A2B = model_for_ddp(self.G_A2B)

        self.G_B2A = get_generator(rank=self.rank).to(device)
        self.G_B2A = model_for_ddp(self.G_B2A)

        self.G_B2C = get_generator(rank=self.rank).to(device)
        self.G_B2C = model_for_ddp(self.G_B2C)

        self.D_A = get_discriminator(rank=self.rank).to(device)
        self.D_A = model_for_ddp(self.D_A)

        self.D_B = get_discriminator(rank=self.rank).to(device)
        self.D_B = model_for_ddp(self.D_B)

        self.D_C = get_discriminator(rank=self.rank).to(device)
        self.D_C = model_for_ddp(self.D_C)

        if clr_cyclegan:
            self.E_A = Emo_Raw_TDNN(dim_noise=dim_noise).to(device)
            self.E_A = model_for_ddp(self.E_A)
            if identical_encoding_nws:
                self.E_B = self.E_A
            else:
                self.E_B = Emo_Raw_TDNN(dim_noise=dim_noise).to(device)
                self.E_B = model_for_ddp(self.E_B)

        if sigmoid_in_disc_loss_calc:
            self.disc_output_masker = nn.Sigmoid()
        else:
            self.disc_output_masker = nn.Identity()
        self.disc_output_masker = self.disc_output_masker.to(device)

        if B2A_preproc_dir:
            self.B2A_preproc = get_B2A_preproc().to(device)

        # criterions
        self.criterion = get_criterion().to(device)     # this is main criterion
#        self.criterion_base = nn.MSELoss().to(device)
        self.MSE_criterion = nn.MSELoss().to(device)
        self.KLDivcriterion = nn.KLDivLoss(reduction='batchmean').to(device)
        if type_loss_clr == 'l2':
            self.criterion_clr = nn.MSELoss().to(device)
        elif type_loss_clr == 'l1':
            self.criterion_clr = nn.L1Loss().to(device)
        else:
            raise NotImplementedError(f'{type_loss_clr=}')

        # optimizer
        G_params = list(self.G_A2B.parameters()) + list(self.G_B2A.parameters()) + list(self.G_B2C.parameters())
        if clr_cyclegan:
            G_params = G_params + list(self.E_A.parameters()) + list(self.E_B.parameters())
        D_params = list(self.D_A.parameters()) + list(self.D_B.parameters()) + list(self.D_C.parameters())
        self.G_optimizer = self.create_optimizer(G_params, lr_G)
        self.D_optimizer = self.create_optimizer(D_params, lr_D)

        # scheduler

        # constants
        self.lambda_CGAN = lambda_CGAN
        if not learnable_cycle_loss:
            self.lambda_cycle = torch.tensor(float(lambda_cycle))
            self.lambda_cycle_wrapper = nn.Identity()
        else:
            self.lambda_cycle = torch.nn.Parameter(torch.tensor(random.random() if learnable_loss_rand_init else lambda_cycle), requires_grad=True)
            self.G_optimizer.add_param_group({'params': self.lambda_cycle})
            self.lambda_cycle_wrapper = get_learnable_loss_wrapper_fxn()

        if not learnable_identity_loss:
            self.lambda_identity = torch.tensor(float(lambda_identity))
            self.lambda_identity_wrapper = nn.Identity()
        else:
            self.lambda_identity = torch.nn.Parameter(torch.tensor(random.random() if learnable_loss_rand_init else lambda_identity), requires_grad=True)
            self.G_optimizer.add_param_group({'params': self.lambda_identity})
            self.lambda_identity_wrapper = get_learnable_loss_wrapper_fxn()
            
        if clr_cyclegan:
            self.lambda_encode = lambda_encode
            if kldivloss_on_encoding:
                self.lambda_kldiv = lambda_kldiv

        # resume model
        self.latest_iteration = self.resume_model()     # how many iterations done already (0:= training not started yet)
        if self.rank == 0:
            if self.latest_iteration == 0:
                backup_if_exists(log_training)
                backup_if_exists(log_progress)
                print('iteration', 'G_A2B [T]', 'G_B2A [T]', 'G_B2C [T]', 'D_A [T]', 'D_B [T]', 'D_C [T]', 'cycle [T]', 'iden [T]', 't [T]',
                                   'G_A2B [V]', 'G_B2A [V]', 'G_B2C [V]', 'D_A [V]', 'D_B [V]', 'D_C [V]', 'cycle [V]', 'iden [V]', 't [V]', 'lr_G', 'lr_D', sep=',', file=open(log_progress, 'a'))    # T:train, V:validation
            else:
                print(f'RESUMING @ {getcurrtimestamp()}', file=open(log_training, 'a'))
                print(f'RESUMING @ {getcurrtimestamp()}', file=open(log_progress, 'a'))

    def create_optimizer(self, params, lr):
        if class_optimizer == 'adam':
            optimizer_base = optim.Adam
        elif class_optimizer == 'ranger':
            optimizer_base = Ranger
        elif class_optimizer == 'adam_8k':
            optimizer_base = bnb.optim.Adam8bit
        elif class_optimizer == 'SGD':
            optimizer_base = optim.SGD
        else:
            raise NotImplementedError(f'{class_optimizer=}')
        if class_optimizer == 'SGD':
            optimizer = optimizer_base(params, lr=lr)
        else:
            optimizer = optimizer_base(params, lr=lr, betas=(0.5, adam_beta2))
        return optimizer

    def criterion_disc(self, d, d2=None, domain=None):
        'criterion for discriminating real/fake samples'
        if isinstance(d, (list,tuple)):
            if isinstance(d[0], (list,tuple)):
                assert not isinstance(d[0][0], (list,tuple))    # it should not be list of list of list
                d = [self.disc_output_masker(TMP[-1]) for TMP in d]
            else:
                d = self.disc_output_masker(d[-1])
        else:
            d = self.disc_output_masker(d)
        if d2 is not None:
            if isinstance(d2, (list,tuple)):
                if isinstance(d2[0], (list,tuple)):
                    assert not isinstance(d2[0][0], (list,tuple))    # it should not be list of list of list
                    d2 = [self.disc_output_masker(TMP[-1]) for TMP in d2]
                else:
                    d2 = self.disc_output_masker(d2[-1])
            else:
                d2 = self.disc_output_masker(d2)
        if type_adv_loss == 'LSGAN':
            if 'real' in domain:
                target = 1 - LSGAN_target_offset
            elif 'generated' in domain or 'fake' in domain:
                target = 0 - LSGAN_target_offset
            else:
                raise NotImplementedError(f'{domain=}')
            loss = torch_calc_error(d, target, self.MSE_criterion)
        elif type_adv_loss == 'hinge':
            if domain == 'real_D':
                if isinstance(d, list):
                    loss = -torch.mean(torch.stack([torch.mean(torch.minimum(torch.zeros_like(d[_]),-1+d[_])) for _ in range(len(d))]))
                else:
                    assert isinstance(d, torch.Tensor)
                    loss = -torch.mean(torch.minimum(torch.zeros_like(d),-1+d))
            elif domain == 'generated_D':
                if isinstance(d, list):
                    loss = -torch.mean(torch.stack([torch.mean(torch.minimum(torch.zeros_like(d[_]),-1-d[_])) for _ in range(len(d))]))
                else:
                    assert isinstance(d, torch.Tensor)
                    loss = -torch.mean(torch.minimum(torch.zeros_like(d),-1-d))
            elif domain == 'real_G':
                if isinstance(d, list):
                    loss = -torch.mean(torch.stack([torch.mean(d[_]) for _ in range(len(d))]))
                else:
                    assert isinstance(d, torch.Tensor)
                    loss = -torch.mean(d)
            else:
                raise NotImplementedError(f'{domain=}')
        elif type_adv_loss == 'gan':
            if domain == 'real_D':
                if isinstance(d, list):
                    loss = -torch.mean(torch.stack([torch.mean(torch.log(d[_])) for _ in range(len(d))]))
                else:
                    assert isinstance(d, torch.Tensor)
                    loss = -torch.mean(torch.log(d))
            elif domain == 'generated_D':
                if isinstance(d, list):
                    loss = -torch.mean(torch.stack([torch.mean(torch.log(1 - d[_])) for _ in range(len(d))]))
                else:
                    assert isinstance(d, torch.Tensor)
                    loss = -torch.mean(torch.log(1 - d))
            elif domain == 'real_G':
                if isinstance(d, list):
                    loss = torch.mean(torch.stack([torch.mean(torch.log(1 - d[_])) for _ in range(len(d))]))
                else:
                    assert isinstance(d, torch.Tensor)
                    loss = torch.mean(torch.log(1 - d))
            else:
                raise NotImplementedError(f'{domain=}')
        elif type_adv_loss == 'wgan':
            if domain == 'real_D':
                if isinstance(d, list):
                    loss = -torch.mean(torch.stack([torch.mean(d[_]) for _ in range(len(d))]))
                else:
                    assert isinstance(d, torch.Tensor)
                    loss = -torch.mean(d)
            elif domain == 'generated_D':
                if isinstance(d, list):
                    loss = torch.mean(torch.stack([torch.mean(d[_]) for _ in range(len(d))]))
                else:
                    assert isinstance(d, torch.Tensor)
                    loss = torch.mean(d)
            elif domain == 'real_G':    # same as real_D
                if isinstance(d, list):
                    loss = torch.mean(torch.stack([torch.mean(d[_]) for _ in range(len(d))]))
                else:
                    assert isinstance(d, torch.Tensor)
                    loss = torch.mean(d)
            else:
                raise NotImplementedError(f'{domain=}')
        elif type_adv_loss == 'dcl':
            assert d2 is not None
            # assumption: d is real, d2 is generated
            assert domain in ['D', 'G']
            loss = 0
#            print([_.shape for _ in d])
#            print([_.shape for _ in d2])
            if not isinstance(d, (list,tuple)):
                d = [d]
                d2 = [d2]
            for idx in range(len(d)):
                _d = d[idx]
                _d2 = d2[idx]
                assert isinstance(_d, torch.Tensor)
                assert isinstance(_d2, torch.Tensor)
                B = _d.shape[0]
                assert _d2.shape[0] == B
                d_orig = _d.reshape(B, -1)   # real
                d2_orig = _d2.reshape(B, -1) # fake

                _d = d_orig.repeat_interleave(B, dim=0)     # B*B x L
                _d2 = d2_orig.repeat(B,1)   # B*B x L
                d3 = _d2 - _d
                d4 = torch.tensor_split(d3, B, dim=0)   # tuple
                d4 = torch.stack(d4, dim=0) # B, B, L
                d4 = d4.reshape(B,-1)   # B,L
                if topk_in_DCL:
                    d4 = torch.topk(d4, k=math.floor(topk_in_DCL_perc*d4.shape[0]/100), sorted=False, largest=not topk_in_DCL_smallest)[0]
                d4 = d4.mean(-1)    # B or B//2
                L_contr_real = -torch.mean(torch.log(1+d4))   # torch.tensor

                _d = d_orig.repeat(B,1)   # B*B x L
                _d2 = d2_orig.repeat_interleave(B, dim=0)     # B*B x L
                d3 = _d2 - _d
                d4 = torch.tensor_split(d3, B, dim=0)   # tuple
                d4 = torch.stack(d4, dim=0) # B, B, L
                d4 = d4.reshape(B,-1)   # B,L
                if topk_in_DCL:
                    d4 = torch.topk(d4, k=math.floor(topk_in_DCL_perc*d4.shape[0]/100), sorted=False, largest=not topk_in_DCL_smallest)[0]
                d4 = d4.mean(-1)    # B or B//2
                L_contr_fake = -torch.mean(torch.log(1+d4))   # torch.tensor

                if domain == 'D':
                    loss = loss - (L_contr_real + L_contr_fake)
                elif domain == 'G':
                    loss = loss + (L_contr_real + L_contr_fake)
        else:
            raise NotImplementedError(f'{type_adv_loss=}')
        return loss

    def criterion_cycle(self, x, y):
        if len(x.shape) == 2 and len(y.shape) == 3:
            y = y.squeeze(1)
        elif len(x.shape) == 3 and len(y.shape) == 2:
            x = x.squeeze(1)
        else:
            assert len(x.shape) == len(y.shape)
        return self.criterion(x, y)

    def do_B2A_preproc(self, dataA, dataB, dataC):
        # order-preserving pre-processing
        if 0 < p_B2A <= 1:
            B = dataA.shape[0]
            set_idx_1 = list(*np.where((np.random.random(B) <= p_B2A) == True))
            if len(set_idx_1) > 0:
#                set_idx_2 = [_ for _ in range(B) if _ not in set_idx_1]
#                assert len(set_idx_2) > 0
                if B2A_preproc_domain == 'C':
                    data2adapt = dataC
                elif B2A_preproc_domain == 'B':
                    data2adapt = dataB
                dataB2A = torch.index_select(data2adapt, 0, torch.tensor(set_idx_1, dtype=torch.int64, device=data2adapt.device))
                with torch.no_grad():
                    dataB2A = self.B2A_preproc(dataB2A).squeeze(1)
#                dataA_subset = torch.index_select(dataA, 0, set_idx_2) # does not refer to same memory
                if B2A_keepA:
                    if B2A_removeC:
                        dataA = torch.cat((dataA, dataB2A))
                    else:
                        try:
                            dataA = torch.cat((dataA, dataB2A, dataC))
                        except Exception as e:
                            raise Exception(f'{e=} {dataA.shape=} {dataB2A.shape=} {dataC.shape=}')
                    set_idx_3 = random.choices(range(dataA.shape[0]), k=B)
                    dataA = torch.index_select(dataA, 0, torch.tensor(set_idx_3, dtype=torch.int64, device=dataA.device))
                else:
                    for ii, index in enumerate(set_idx_1):
                        dataA[index] = dataB2A[ii]
        return dataA

    def do_train(self):
        for iteration in range(self.latest_iteration+1, n_iterations+int(do_last_iteration)):
            loss_G_A2B_mean, loss_G_A2B_std, loss_G_B2A_mean, loss_G_B2A_std, loss_G_B2C_mean, loss_G_B2C_std, loss_CGAN_mean, loss_CGAN_std, loss_cycle_mean, loss_cycle_std, loss_iden_mean, loss_iden_std, \
                loss_D_A_mean, loss_D_A_std, loss_D_B_mean, loss_D_B_std, loss_D_C_mean, loss_D_C_std, \
                lr_G_curr, lr_D_curr, time_per_iteration = self.do_train_single_epoch(iteration)
            if self.rank == 0:
                print(iteration, f'{loss_G_A2B_mean} ({loss_G_A2B_std})', f'{loss_G_B2A_mean} ({loss_G_B2A_std})', f'{loss_G_B2C_mean} ({loss_G_B2C_std})', f'{loss_CGAN_mean} ({loss_CGAN_std})', f'{loss_cycle_mean} ({loss_cycle_std})', f'{loss_iden_mean} ({loss_iden_std})',
                    f'{loss_D_A_mean} ({loss_D_A_std})', f'{loss_D_B_mean} ({loss_D_B_std})', f'{loss_D_C_mean} ({loss_D_C_std})',
                    round(time_per_iteration), sep=',', end=',', file=open(log_progress, 'a')) # std in brackets this time
            if not skip_do_validate:
                loss_G_A2B_mean, loss_G_A2B_std, loss_G_B2A_mean, loss_G_B2A_std, loss_G_B2C_mean, loss_G_B2C_std, loss_CGAN_mean, loss_CGAN_std, loss_cycle_mean, loss_cycle_std, loss_iden_mean, loss_iden_std, \
                    loss_D_A_mean, loss_D_A_std, loss_D_B_mean, loss_D_B_std, loss_D_C_mean, loss_D_C_std, \
                    lr_G_curr_after, lr_D_curr_after, time_per_iteration, last_val_batch = self.do_validate(iteration)
                assert lr_G_curr == lr_G_curr_after, f'{lr_G_curr=} {lr_G_curr_after=}'
                assert lr_D_curr == lr_D_curr_after, f'{lr_D_curr=} {lr_D_curr_after=}'
                if self.rank == 0:
                    print(iteration, f'{loss_G_A2B_mean} ({loss_G_A2B_std})', f'{loss_G_B2A_mean} ({loss_G_B2A_std})', f'{loss_G_B2C_mean} ({loss_G_B2C_std})', f'{loss_CGAN_mean} ({loss_CGAN_std})', f'{loss_cycle_mean} ({loss_cycle_std})', f'{loss_cycle_mean} ({loss_cycle_std})', f'{loss_iden_mean} ({loss_iden_std})',
                        f'{loss_D_A_mean} ({loss_D_A_std})', f'{loss_D_B_mean} ({loss_D_B_std})', f'{loss_D_C_mean} ({loss_D_C_std})',
                        round(time_per_iteration), lr_G_curr, lr_D_curr, sep=',', file=open(log_progress, 'a')) # std in brackets this time
                loss_G_mean = round((loss_G_A2B_mean + loss_G_B2A_mean + loss_G_B2C_mean)/3, 4)
                loss_D_mean = round((loss_D_A_mean + loss_D_B_mean + loss_D_C_mean)/3, 4)
            else:
                if self.rank == 0:
                    print('\n', file=open(log_progress, 'a'))
                loss_G_mean = loss_D_mean = last_val_batch = 0
            self.save_model(iteration, loss_G_mean, loss_D_mean, last_val_batch)
        #
        if not disable_wandb and self.rank == 0:
            self.wandb_run.finish()
        ddp.ddp_cleanup()

    def do_train_single_epoch(self, curr_iteration):
        time_start_iteration = time.time()
        _ = self.G_A2B.train(mode=True)
        _ = self.G_B2A.train(mode=True)
        _ = self.G_B2C.train(mode=True)
        _ = self.D_A.train(mode=True)
        _ = self.D_B.train(mode=True)
        _ = self.D_C.train(mode=True)
        if clr_cyclegan:
            _ = self.E_A.train(mode=True)
            _ = self.E_B.train(mode=True)
            toggle_grad(self.E_A, True)
            toggle_grad(self.E_B, True)
        # update these vectors at logging frequency (minimize memory consumption)
        G_A2B_lossvec = [EPS, 2*EPS]
        G_B2A_lossvec = [EPS, 2*EPS]
        G_B2C_lossvec = [EPS, 2*EPS]
        CGAN_lossvec = [EPS, 2*EPS]
        cycle_lossvec = [EPS, 2*EPS]
        iden_lossvec = [EPS, 2*EPS]
        D_A_lossvec = [EPS, 2*EPS]
        D_B_lossvec = [EPS, 2*EPS]
        D_C_lossvec = [EPS, 2*EPS]
        if self.rank == 0:
            time_start_step = time.time()
        unpaired_traindataiterator = iter(unpaired_trainloader)     # 1/2
        for ii, (dataB,dataC) in enumerate(paired_trainloader):
            try:
                dataA = next(unpaired_traindataiterator)
            except StopIteration:
                unpaired_traindataiterator = iter(unpaired_trainloader)     # 2/2
                dataA = next(unpaired_traindataiterator)
            # set learning rate ahead of training
            curr_step = int(ii + (curr_iteration-1)*len_trainsampler)
            self.adjust_lr(curr_iteration, curr_step)
            # load data
            dataA = dataA[0].to(device, non_blocking=True).float() / audio_scale
            dataB = dataB[0].to(device, non_blocking=True).float() / audio_scale
            dataC = dataC[0].to(device, non_blocking=True).float() / audio_scale
            # B2A pre-processing
            dataA = self.do_B2A_preproc(dataA, dataB, dataC)
            if n_subbatches == 1:
                data = (dataA,), (dataB,), (dataC,)
            else:
                data = torch.tensor_split(dataA, n_subbatches, 0), torch.tensor_split(dataB, n_subbatches, 0), torch.tensor_split(dataC, n_subbatches, 0)
            self.G_optimizer.zero_grad()
            self.D_optimizer.zero_grad()
            for jj, (real_A,real_B,real_C) in enumerate(zip(*data)):
                assert real_A.shape == real_B.shape == real_C.shape, f'{real_A.shape=} {real_B.shape=} {real_C.shape=}'
                # update discriminators
                if ii % D_update_periodicity == 0:
                    toggle_grad(self.G_A2B, False or GD_algo == 'simultaneous')
                    toggle_grad(self.G_B2A, False or GD_algo == 'simultaneous')
                    toggle_grad(self.G_B2C, False or GD_algo == 'simultaneous')
                    toggle_grad(self.D_A, True)
                    toggle_grad(self.D_B, True)
                    toggle_grad(self.D_C, True)
                    with autocast(enabled=use_amp):
                        d_real_A = self.D_A(real_A)
                        z_B = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                        generated_A = self.G_B2A(real_B, z=z_B)
                        d_generated_A = self.D_A(generated_A)
                        d_real_B = self.D_B(real_B)
                        z_A = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                        generated_B = self.G_A2B(real_A, z=z_A)
                        d_generated_B = self.D_B(generated_B)
#                        generated_B__detached = ...
                        generated_C = self.G_B2C(generated_B)
                        d_generated_C = self.D_C(generated_C)
                        generated_C__sup = self.G_B2C(real_B)
                        d_generated_C__sup = self.D_C(generated_C__sup)
                        d_real_C = self.D_C(real_C)
                        if type_adv_loss == 'dcl':
                            loss_D_A_real = EPS*torch.tensor(1).to(device)
                            loss_D_B_real = EPS*torch.tensor(1).to(device)
                            loss_D_C_real = EPS*torch.tensor(1).to(device)
                            loss_D_A_generated = EPS*torch.tensor(1).to(device)
                            loss_D_B_generated = EPS*torch.tensor(1).to(device)
                            loss_D_C_generated = EPS*torch.tensor(1).to(device)
                            loss_D_A = self.criterion_disc(d_real_A, d2=d_generated_A, domain='D')
                            loss_D_B = self.criterion_disc(d_real_B, d2=d_generated_B, domain='D')
                            loss_D_C = (self.criterion_disc(d_real_C, d2=d_generated_C, domain='D') + self.criterion_disc(d_real_C, d2=d_generated_C__sup, domain='D')) / 2
                            if identity_discriminator:  # not done for C
                                z_iden_B = std_noise*torch.randn(real_A.shape[0], dim_noise, device=device) if clr_cyclegan else None
                                generated_iden_A = self.G_B2A(real_A, z=z_iden_B)
                                d_generated_iden_A = self.D_A(generated_iden_A)
                                z_iden_A = std_noise*torch.randn(real_B.shape[0], dim_noise, device=device) if clr_cyclegan else None
                                generated_iden_B = self.G_A2B(real_B, z=z_iden_A)
                                d_generated_iden_B = self.D_B(generated_iden_B)
                                loss_D_A = (loss_D_A + self.criterion_disc(d_real_A, d2=d_generated_iden_A, domain='D')) / 2
                                loss_D_B = (loss_D_B + self.criterion_disc(d_real_B, d2=d_generated_iden_B, domain='D')) / 2
                        else:
                            loss_D_A_real = self.criterion_disc(d_real_A, domain='real_D')
                            loss_D_B_real = self.criterion_disc(d_real_B, domain='real_D')
                            loss_D_C_real = self.criterion_disc(d_real_C, domain='real_D')
                            loss_D_A_generated = self.criterion_disc(d_generated_A, domain='generated_D')
                            loss_D_B_generated = self.criterion_disc(d_generated_B, domain='generated_D')
                            loss_D_C_generated = (self.criterion_disc(d_generated_C, domain='generated_D') + self.criterion_disc(d_generated_C__sup, domain='generated_D')) / 2
                            if identity_discriminator:
                                z_iden_B = std_noise*torch.randn(real_A.shape[0], dim_noise, device=device) if clr_cyclegan else None
                                generated_iden_A = self.G_B2A(real_A, z=z_iden_B)
                                d_generated_iden_A = self.D_A(generated_iden_A)
                                loss_D_A_generated = (loss_D_A_generated + self.criterion_disc(d_generated_iden_A, domain='generated_D'))/2
                                z_iden_A = std_noise*torch.randn(real_B.shape[0], dim_noise, device=device) if clr_cyclegan else None
                                generated_iden_B = self.G_A2B(real_B, z=z_iden_A)
                                d_generated_iden_B = self.D_B(generated_iden_B)
                                loss_D_B_generated = (loss_D_B_generated + self.criterion_disc(d_generated_iden_B, domain='generated_D'))/2
                            loss_D_real = (loss_D_A_real + loss_D_B_real + loss_D_C_real) / 3   # these are just calculated for no reason
                            loss_D_generated = (loss_D_A_generated + loss_D_B_generated + loss_D_C_generated) / 3
                            loss_D_A = (loss_D_A_real + loss_D_A_generated) / 2
                            loss_D_B = (loss_D_B_real + loss_D_B_generated) / 2
                            loss_D_C = (loss_D_C_real + loss_D_C_generated) / 2
                        loss_D = (loss_D_A + loss_D_B + loss_D_C) / 3
                    if ii % log_periodicity_steps == 0:
                        D_A_lossvec.append(loss_D_A.item())
                        D_B_lossvec.append(loss_D_B.item())
                        D_C_lossvec.append(loss_D_C.item())
                    scaler.scale(loss_D).backward()
                    if jj % n_subbatches == 0 and GD_algo == 'alternative':
                        scaler.step(self.D_optimizer)
                        scaler.update()
                        if correct_optimzero:
                            self.D_optimizer.zero_grad()
                # update generators
                if ii % G_update_periodicity == 0:
                    toggle_grad(self.G_A2B, True)
                    toggle_grad(self.G_B2A, True)
                    toggle_grad(self.G_B2C, True)
                    toggle_grad(self.D_A, False or GD_algo == 'simultaneous')
                    toggle_grad(self.D_B, False or GD_algo == 'simultaneous')
                    toggle_grad(self.D_C, False or GD_algo == 'simultaneous')
                    with autocast(enabled=use_amp):
                        #
                        if GD_algo == 'alternative':
                            z_B = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                            generated_A = self.G_B2A(real_B, z=z_B)
                        else:   # generated_A exists, just detach it
                            generated_A = generated_A.detach()
                        if clr_cyclegan:
                            e_generated_A = self.E_A(generated_A)
                        d_generated_A = self.D_A(generated_A)
                        #
                        if GD_algo == 'alternative':
                            z_A = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                            generated_B = self.G_A2B(real_A, z=z_A)
                        else:   # similar to generated_A, generated_B exists
                            generated_B = generated_B.detach()
                        if clr_cyclegan:
                            e_generated_B = self.E_B(generated_B)
                        d_generated_B = self.D_B(generated_B)
                        #
                        if GD_algo == 'alternative':
                            z_B = std_noise*torch.randn(real_C.shape[0],dim_noise,device=device) if clr_cyclegan else None
                            generated_C = self.G_B2C(generated_B, z=z_B)
                            generated_C__sup = self.G_B2C(real_B, z=z_B)
                        else:   # similar to generated_A, generated_B exists
                            generated_C = generated_C.detach()
                            generated_C__sup = generated_C__sup.detach()
#                        if clr_cyclegan:
#                            e_generated_B = self.E_B(generated_B)
                        d_generated_C = self.D_C(generated_C)
                        d_generated_C__sup = self.D_C(generated_C__sup)
                        #
                        if bicyclegan:
                            if bicyclegan_ver == 'v1':
                                e_real_A = self.E_A(real_A)
                                e_real_B = self.E_B(real_B)
                            elif bicyclegan_ver == 'v2':
                                e_real_A = self.E_B(real_A)
                                e_real_B = self.E_A(real_B)
                        cycle_A = self.G_B2A(generated_B, z=e_real_A if bicyclegan else z_A)
                        cycle_B = self.G_A2B(generated_A, z=e_real_B if bicyclegan else z_B)    # use same noise
                        if GD_algo == 'alternative' or (GD_algo == 'simultaneous' and not identity_discriminator):
                            z_iden_B = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                            generated_iden_A = self.G_B2A(real_A, z=z_iden_B)
                            z_iden_A = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                            generated_iden_B = self.G_A2B(real_B, z=z_iden_A)

                        if type_adv_loss == 'dcl':
                            d_real_A = self.D_A(real_A)
                            d_real_B = self.D_B(real_B)
                            d_real_C = self.D_C(real_C)
                            loss_G_A2B_disc = self.criterion_disc(d_real_B, d2=d_generated_B, domain='G')
                            loss_G_B2A_disc = self.criterion_disc(d_real_A, d2=d_generated_A, domain='G')
                            loss_G_B2C_disc = (self.criterion_disc(d_real_C, d2=d_generated_C, domain='G') + self.criterion_disc(d_real_C, d2=d_generated_C__sup, domain='G')) / 2
                        else:
                            loss_G_A2B_disc = self.criterion_disc(d_generated_B, domain='real_G')
                            loss_G_B2A_disc = self.criterion_disc(d_generated_A, domain='real_G')
                            loss_G_B2C_disc = (self.criterion_disc(d_generated_C, domain='real_G') + self.criterion_disc(d_generated_C__sup, domain='real_G')) / 2
                        loss_G_disc = (loss_G_A2B_disc + loss_G_B2A_disc + loss_G_B2C_disc) / 3
                        loss_cycle_A = self.criterion_cycle(real_A, cycle_A)
                        loss_cycle_B = self.criterion_cycle(real_B, cycle_B)
                        loss_cycle = (cycle_loss_AtoB_ratio*loss_cycle_A + loss_cycle_B) / (cycle_loss_AtoB_ratio + 1)
                        loss_iden_A = self.criterion_cycle(real_A, generated_iden_A)
                        loss_iden_B = self.criterion_cycle(real_B, generated_iden_B)
                        loss_iden = (loss_iden_A + loss_iden_B) / 2
                        loss_CGAN = self.criterion_cycle(real_C, generated_C__sup)
                        loss_G = loss_G_disc + self.lambda_cycle_wrapper(self.lambda_cycle) * loss_cycle + self.lambda_identity_wrapper(self.lambda_identity) * loss_iden + self.lambda_cycle_wrapper(self.lambda_CGAN) * loss_CGAN
                        if clr_cyclegan:
                            loss_clr = (self.criterion_clr(e_generated_A, z_B) + self.criterion_clr(e_generated_B, z_A)) / 2
                            loss_G = loss_G + self.lambda_encode * loss_clr
                            if kldivloss_on_encoding:
                                if disable_kldivloss_on_encoding:
                                    loss_kldiv = 0
                                else:
                                    loss_kldiv = (self.KLDivcriterion(e_generated_A,z_B) + self.KLDivcriterion(e_generated_B,z_A)) / 2
                                if bicyclegan:
                                    loss_kldiv = loss_kldiv + (self.KLDivcriterion(e_real_A,z_A) + self.KLDivcriterion(e_real_B,z_B)) / 2
                                loss_G = loss_G + self.lambda_kldiv * loss_kldiv
                    if ii % log_periodicity_steps == 0:
                        G_A2B_lossvec.append(loss_G_A2B_disc.item())
                        G_B2A_lossvec.append(loss_G_B2A_disc.item())
                        G_B2C_lossvec.append(loss_G_B2C_disc.item())
                        CGAN_lossvec.append(loss_CGAN.item())
                        cycle_lossvec.append(loss_cycle.item())
                        iden_lossvec.append(loss_iden.item())
                        D_A_lossvec.append(loss_D_A.item())
                        D_B_lossvec.append(loss_D_B.item())
                        D_C_lossvec.append(loss_D_C.item())
                    scaler.scale(loss_G).backward()
                    if jj % n_subbatches == 0 and GD_algo == 'simultaneous':
                        scaler.step(self.D_optimizer)
                        scaler.update()
                        if correct_optimzero:
                            self.D_optimizer.zero_grad()
                    if jj % n_subbatches == 0:  # always update G
                        scaler.step(self.G_optimizer)
                        scaler.update()
                        if correct_optimzero:
                            self.G_optimizer.zero_grad()
            # logging 1/n
            if (ii % log_periodicity_steps) == 0 and self.rank == 0:
                time_per_step = time.time() - time_start_step
                lr_G_curr = get_lr(self.G_optimizer)
                lr_D_curr = get_lr(self.D_optimizer)
                print("[T] {}, {} Perc:{:.2f}% G_A2B:{:.4f} G_B2A:{:.4f} G_B2C:{:.4f} GAN:{:.4f} cycle_A:{:.4f} cycle_B:{:4f} cycle (w/ C):{:.4f} iden_A:{:.4f} iden_B:{:.4f} iden (w/ C):{:.4f} D_A:{:.4f} D_B:{:.4f} D_C:{:.4f} D_A_real:{:.4f} D_A_generated:{:.4f} D_B_real:{:.4f} D_B_generated:{:.4f} D_C_real:{:.4f} D_C_generated:{:.4f} lr_G_curr:{:.7f} lr_D_curr:{:.7f} Time:{}".format(
                    curr_iteration, curr_step, ii*100/len_trainsampler, loss_G_A2B_disc.item(), loss_G_B2A_disc.item(), loss_G_B2C_disc.item(), (self.lambda_cycle_wrapper(self.lambda_CGAN) * loss_CGAN).item(), loss_cycle_A.item(), loss_cycle_B.item(),
                    (self.lambda_cycle_wrapper(self.lambda_cycle) * loss_cycle).item(), loss_iden_A.item(), loss_iden_B.item(), (self.lambda_identity_wrapper(self.lambda_identity) * loss_iden).item(),
                    loss_D_A.item(), loss_D_B.item(), loss_D_C.item(), loss_D_A_real.item(), loss_D_A_generated.item(), loss_D_B_real.item(), loss_D_B_generated.item(), loss_D_C_real.item(), loss_D_C_generated.item(),
                    lr_G_curr, lr_D_curr, round(time_per_step)), file=open(log_training, 'a'))
                if learnable_cycle_loss:
                    print("learnable_cycle_loss AFTER act:{:.4f}".format(self.lambda_cycle_wrapper(self.lambda_cycle).item()), file=open(log_training, 'a'))
                if learnable_identity_loss:
                    print("learnable_identity_loss AFTER act:{:.4f}".format(self.lambda_identity_wrapper(self.lambda_identity).item()), file=open(log_training, 'a'))
                if clr_cyclegan:
                    print("clr_cyclegan:{:.4f}".format(loss_clr), file=open(log_training, 'a'))
                    if kldivloss_on_encoding:
                        print("kldivloss_on_encoding:{:.4f}".format(loss_kldiv), file=open(log_training, 'a'))
                time_start_step = time.time()
                if not disable_wandb:
                    dict_to_log = {'loss_G':loss_G, 'loss_D':loss_D, 'loss_G_A2B_disc':loss_G_A2B_disc, 'loss_G_B2A_disc':loss_G_B2A_disc, 'loss_CGAN':loss_CGAN, 'loss_cycle':loss_cycle, 'loss_cycle_A':loss_cycle_A, 'loss_cycle_B':loss_cycle_B, 'loss_iden':loss_iden,
                                    'loss_D_A_real':loss_D_A_real, 'loss_D_A_generated':loss_D_A_generated, 'loss_D_B_real':loss_D_B_real, 'loss_D_B_generated':loss_D_B_generated, 'loss_D_C_real':loss_D_C_real, 'loss_D_C_generated':loss_D_C_generated,
                                    'time_per_step':time_per_step}
                    if clr_cyclegan:
                        dict_to_log = mergeDicts(dict_to_log, {'clr':loss_clr})
                        if kldivloss_on_encoding:
                            dict_to_log = mergeDicts(dict_to_log, {'kldiv':loss_kldiv})
                    if learnable_cycle_loss:
                        dict_to_log = mergeDicts(dict_to_log, {'lambda_cycle AFTER act': self.lambda_cycle_wrapper(self.lambda_cycle).item()})
                    if learnable_identity_loss:
                        dict_to_log = mergeDicts(dict_to_log, {'lambda_identity AFTER act': self.lambda_identity_wrapper(self.lambda_identity).item()})
                    wandb.log(dict_to_log, step=curr_step)
            # logging 2/n
            if ii == (len_trainsampler - 2) and self.rank == 0:
                print_shell_cmd_output('nvidia-smi')
                print_shell_cmd_output(f'ps -Flww -p {os.getpid()}')
        if num_gpus > 1:
            dist.barrier()
        time_per_iteration = time.time() - time_start_iteration
        loss_G_A2B_mean = np.mean(G_A2B_lossvec)
        loss_G_A2B_std = np.std(G_A2B_lossvec)
        loss_G_B2A_mean = np.mean(G_B2A_lossvec)
        loss_G_B2A_std = np.std(G_B2A_lossvec)
        loss_G_B2C_mean = np.mean(G_B2C_lossvec)
        loss_G_B2C_std = np.std(G_B2C_lossvec)
        loss_CGAN_mean = np.mean(CGAN_lossvec)
        loss_CGAN_std = np.std(CGAN_lossvec)
        loss_cycle_mean = np.mean(cycle_lossvec)
        loss_cycle_std = np.std(cycle_lossvec)
        loss_iden_mean = np.mean(iden_lossvec)
        loss_iden_std = np.std(iden_lossvec)
        loss_D_A_mean = np.mean(D_A_lossvec)
        loss_D_A_std = np.std(D_A_lossvec)
        loss_D_B_mean = np.mean(D_B_lossvec)
        loss_D_B_std = np.std(D_B_lossvec)
        loss_D_C_mean = np.mean(D_C_lossvec)
        loss_D_C_std = np.std(D_C_lossvec)
        lr_G_curr = get_lr(self.G_optimizer)
        lr_D_curr = get_lr(self.D_optimizer)
        res = [round(_,4) for _ in [loss_G_A2B_mean, loss_G_A2B_std, loss_G_B2A_mean, loss_G_B2A_std, loss_G_B2C_mean, loss_G_B2C_std, loss_CGAN_mean, loss_CGAN_std, loss_cycle_mean, loss_cycle_std, loss_iden_mean, loss_iden_std,
                loss_D_A_mean, loss_D_A_std, loss_D_B_mean, loss_D_B_std, loss_D_C_mean, loss_D_C_std]]
        return *res, round(lr_G_curr,7), round(lr_D_curr,7), time_per_iteration

    def do_validate(self, curr_iteration):
        time_start_iteration = time.time()
        _ = self.G_A2B.train(mode=False)
        _ = self.G_B2A.train(mode=False)
        _ = self.G_B2C.train(mode=False)
        _ = self.D_A.train(mode=False)
        _ = self.D_B.train(mode=False)
        _ = self.D_C.train(mode=False)
        toggle_grad(self.G_A2B, False)
        toggle_grad(self.G_B2A, False)
        toggle_grad(self.G_B2C, False)
        toggle_grad(self.D_A, False)
        toggle_grad(self.D_B, False)
        toggle_grad(self.D_C, False)
        if clr_cyclegan:
            _ = self.E_A.train(mode=False)
            _ = self.E_B.train(mode=False)
            toggle_grad(self.E_A, False)
            toggle_grad(self.E_B, False)
        # update these vectors at logging frequency (minimize memory consumption)
        G_A2B_lossvec = [EPS, 2*EPS]
        G_B2A_lossvec = [EPS, 2*EPS]
        G_B2C_lossvec = [EPS, 2*EPS]
        CGAN_lossvec = [EPS, 2*EPS]
        cycle_lossvec = [EPS, 2*EPS]
        iden_lossvec = [EPS, 2*EPS]
        D_A_lossvec = [EPS, 2*EPS]
        D_B_lossvec = [EPS, 2*EPS]
        D_C_lossvec = [EPS, 2*EPS]
        if self.rank == 0:
            time_start_step = time.time()
        with torch.inference_mode():
            unpaired_traindataiterator = iter(unpaired_trainloader)     # 1/2
            for ii, (dataB,dataC) in enumerate(paired_trainloader):
                try:
                    dataA = next(unpaired_traindataiterator)
                except StopIteration:
                    unpaired_traindataiterator = iter(unpaired_trainloader)     # 2/2
                    dataA = next(unpaired_traindataiterator)
                # set learning rate ahead of training
                curr_step = int(ii + (curr_iteration-1)*len_valsampler)
                # load data
                dataA = dataA[0].to(device, non_blocking=True).float() / audio_scale
                dataB = dataB[0].to(device, non_blocking=True).float() / audio_scale
                dataC = dataC[0].to(device, non_blocking=True).float() / audio_scale
                real_A = dataA
                real_B = dataB
                real_C = dataB
                # B2A pre-processing
                real_A = self.do_B2A_preproc(real_A, real_B, real_C)
                assert real_A.shape == real_B.shape, f'{real_A.shape=} {real_B.shape=}'
                # discriminators
                with autocast(enabled=use_amp):
                    d_real_A = self.D_A(real_A)
                    z = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                    generated_A = self.G_B2A(real_B, z=z)
                    d_generated_A = self.D_A(generated_A)
                    d_real_B = self.D_B(real_B)
                    z = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                    generated_B = self.G_A2B(real_A, z=z)
                    d_generated_B = self.D_B(generated_B)
                    generated_C = self.G_B2C(generated_B)
                    d_generated_C = self.D_C(generated_C)
                    d_real_C = self.D_C(real_C)
                    generated_C__sup = self.G_B2C(real_B)
                    d_generated_C__sup = self.D_C(generated_C__sup)
                    if type_adv_loss == 'dcl':
                        loss_D_A = self.criterion_disc(d_real_A, d2=d_generated_A, domain='D')
                        loss_D_B = self.criterion_disc(d_real_B, d2=d_generated_B, domain='D')
                        loss_D_C = (self.criterion_disc(d_real_C, d2=d_generated_C, domain='D') + self.criterion_disc(d_real_C, d2=d_generated_C__sup, domain='D')) / 2
                        if identity_discriminator:
                            z_iden_B = std_noise*torch.randn(real_A.shape[0], dim_noise, device=device) if clr_cyclegan else None
                            generated_iden_A = self.G_B2A(real_A, z=z_iden_B)
                            d_generated_iden_A = self.D_A(generated_iden_A)
                            z_iden_A = std_noise*torch.randn(real_B.shape[0], dim_noise, device=device) if clr_cyclegan else None
                            generated_iden_B = self.G_A2B(real_B, z=z_iden_A)
                            d_generated_iden_B = self.D_B(generated_iden_B)
                            loss_D_A = (loss_D_A + self.criterion_disc(d_real_A, d2=d_generated_iden_A, domain='D')) / 2
                            loss_D_B = (loss_D_B + self.criterion_disc(d_real_B, d2=d_generated_iden_B, domain='D')) / 2
                    else:
                        loss_D_A_real = self.criterion_disc(d_real_A, domain='real_D')
                        loss_D_B_real = self.criterion_disc(d_real_B, domain='real_D')
                        loss_D_C_real = self.criterion_disc(d_real_C, domain='real_D')
                        loss_D_A_generated = self.criterion_disc(d_generated_A, domain='generated_D')
                        loss_D_B_generated = self.criterion_disc(d_generated_B, domain='generated_D')
                        loss_D_C_generated = (self.criterion_disc(d_generated_C, domain='generated_D') + self.criterion_disc(d_generated_C__sup, domain='generated_D')) / 2
                        if identity_discriminator:
                            z_iden_B = std_noise*torch.randn(real_A.shape[0], dim_noise, device=device) if clr_cyclegan else None
                            generated_iden_A = self.G_B2A(real_A, z=z_iden_B)
                            d_generated_iden_A = self.D_A(generated_iden_A)
                            loss_D_A_generated = (loss_D_A_generated + self.criterion_disc(d_generated_iden_A, domain='generated_D'))/2
                            z_iden_A = std_noise*torch.randn(real_B.shape[0], dim_noise, device=device) if clr_cyclegan else None
                            generated_iden_B = self.G_A2B(real_B, z=z_iden_A)
                            d_generated_iden_B = self.D_B(generated_iden_B)
                            loss_D_B_generated = (loss_D_B_generated + self.criterion_disc(d_generated_iden_B, domain='generated_D'))/2
                        loss_D_real = (loss_D_A_real + loss_D_B_real + loss_D_C_real) / 3   # these are just calculated for no reason
                        loss_D_generated = (loss_D_A_generated + loss_D_B_generated + loss_D_C_generated) / 3
                        loss_D_A = (loss_D_A_real + loss_D_A_generated) / 2
                        loss_D_B = (loss_D_B_real + loss_D_B_generated) / 2
                        loss_D_C = (loss_D_C_real + loss_D_C_generated) / 2
                    loss_D = (loss_D_A + loss_D_B + loss_D_C) / 3
                if ii % log_periodicity_steps == 0:
                    D_A_lossvec.append(loss_D_A.item())
                    D_B_lossvec.append(loss_D_B.item())
                    D_C_lossvec.append(loss_D_C.item())
                # generators
                with autocast(enabled=use_amp):
                    z_B = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                    generated_A = self.G_B2A(real_B, z=z_B)
                    if clr_cyclegan:
                        e_generated_A = self.E_A(generated_A)
                    d_generated_A = self.D_A(generated_A)
                    z_A = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                    generated_B = self.G_A2B(real_A, z=z_A)
                    if clr_cyclegan:
                        e_generated_B = self.E_B(generated_B)
                    d_generated_B = self.D_B(generated_B)
                    #
                    z_C = std_noise*torch.randn(real_C.shape[0],dim_noise,device=device) if clr_cyclegan else None
                    generated_C = self.G_B2C(generated_B, z=z_C)
#                    if clr_cyclegan:
#                        e_generated_B = self.E_B(generated_B)
                    d_generated_C = self.D_C(generated_C)
                    generated_C__sup = self.G_B2C(real_B, z=z_C)
                    d_generated_C__sup = self.D_C(generated_C__sup)
                    #
                    if bicyclegan:
                        if bicyclegan_ver == 'v1':
                            e_real_A = self.E_A(real_A)
                            e_real_B = self.E_B(real_B)
                        elif bicyclegan_ver == 'v2':
                            e_real_A = self.E_B(real_A)
                            e_real_B = self.E_A(real_B)
                    cycle_A = self.G_B2A(generated_B, z=e_real_A if bicyclegan else z_A)
                    cycle_B = self.G_A2B(generated_A, z=e_real_B if bicyclegan else z_B)    # use same noise
                    z = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                    iden_A = self.G_B2A(real_A, z=z)
                    z = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                    iden_B = self.G_A2B(real_B, z=z)

#                    loss_G_A2B_disc = self.criterion_disc(d_generated_B, domain='real')
#                    loss_G_B2A_disc = self.criterion_disc(d_generated_A, domain='real')
                    if type_adv_loss == 'dcl':
                        d_real_A = self.D_A(real_A)
                        d_real_B = self.D_B(real_B)
                        d_real_C = self.D_C(real_C)
                        loss_G_A2B_disc = self.criterion_disc(d_real_B, d2=d_generated_B, domain='G')
                        loss_G_B2A_disc = self.criterion_disc(d_real_A, d2=d_generated_A, domain='G')
                        loss_G_B2C_disc = (self.criterion_disc(d_real_C, d2=d_generated_C, domain='G') + self.criterion_disc(d_real_C, d2=d_generated_C__sup, domain='G')) / 2
                    else:
                        loss_G_A2B_disc = self.criterion_disc(d_generated_B, domain='real_G')
                        loss_G_B2A_disc = self.criterion_disc(d_generated_A, domain='real_G')
                        loss_G_B2C_disc = (self.criterion_disc(d_generated_C, domain='real_G') + self.criterion_disc(d_generated_C__sup, domain='real_G')) / 2
                    loss_G_disc = (loss_G_A2B_disc + loss_G_B2A_disc + loss_G_B2C_disc) / 3
                    loss_cycle_A = self.criterion_cycle(real_A, cycle_A)
                    loss_cycle_B = self.criterion_cycle(real_B, cycle_B)
                    loss_cycle = (cycle_loss_AtoB_ratio*loss_cycle_A + loss_cycle_B) / (cycle_loss_AtoB_ratio + 1)
                    loss_iden_A = self.criterion_cycle(real_A, iden_A)
                    loss_iden_B = self.criterion_cycle(real_B, iden_B)
                    loss_iden = (loss_iden_A + loss_iden_B) / 2
                    loss_CGAN = self.criterion_cycle(real_C, generated_C__sup)
                    loss_G = loss_G_disc + self.lambda_cycle_wrapper(self.lambda_cycle) * loss_cycle + self.lambda_identity_wrapper(self.lambda_identity) * loss_iden + self.lambda_cycle_wrapper(self.lambda_CGAN) * loss_CGAN
                    if clr_cyclegan:
                        loss_clr = (self.criterion_clr(e_generated_A, z_B) + self.criterion_clr(e_generated_B, z_A)) / 2
                        loss_G = loss_G + self.lambda_encode * loss_clr
                        if kldivloss_on_encoding:
                            if disable_kldivloss_on_encoding:
                                loss_kldiv = 0
                            else:
                                loss_kldiv = (self.KLDivcriterion(e_generated_A,z_B) + self.KLDivcriterion(e_generated_B,z_A)) / 2
                            if bicyclegan:
                                loss_kldiv = loss_kldiv + (self.KLDivcriterion(e_real_A,z_A) + self.KLDivcriterion(e_real_B,z_B)) / 2
                            loss_G = loss_G + self.lambda_kldiv * loss_kldiv
                if ii % log_periodicity_steps == 0:
                    G_A2B_lossvec.append(loss_G_A2B_disc.item())
                    G_B2A_lossvec.append(loss_G_B2A_disc.item())
                    G_B2C_lossvec.append(loss_G_B2C_disc.item())
                    CGAN_lossvec.append(loss_CGAN.item())
                    cycle_lossvec.append(loss_cycle.item())
                    iden_lossvec.append(loss_iden.item())
                    D_A_lossvec.append(loss_D_A.item())
                    D_B_lossvec.append(loss_D_B.item())
                    D_C_lossvec.append(loss_D_C.item())
            # logging 1/n
            if (ii % log_periodicity_steps) == 0 and self.rank == 0:
                time_per_step = time.time() - time_start_step
                lr_G_curr = get_lr(self.G_optimizer)
                lr_D_curr = get_lr(self.D_optimizer)
                print("[V] {}, {} Perc:{:.2f}% G_A2B:{:.4f} G_B2A:{:.4f} G_B2C:{:.4f} GAN:{:.4f} cycle_A:{:.4f} cycle_B:{:4f} cycle (w/ C):{:.4f} iden_A:{:.4f} iden_B:{:.4f} iden (w/ C):{:.4f} D_A:{:.4f} D_B:{:.4f} D_C:{:.4f} D_A_real:{:.4f} D_A_generated:{:.4f} D_B_real:{:.4f} D_B_generated:{:.4f} D_C_real:{:.4f} D_C_generated:{:.4f} lr_G_curr:{:.7f} lr_D_curr:{:.7f} Time:{}".format(
                    curr_iteration, curr_step, ii*100/len_valsampler, loss_G_A2B_disc.item(), loss_G_B2A_disc.item(), loss_G_B2C_disc.item(), (self.lambda_cycle_wrapper(self.lambda_CGAN) * loss_CGAN).item(), loss_cycle_A.item(), loss_cycle_B.item(),
                    loss_cycle_A.item(), loss_cycle_B.item(), (self.lambda_cycle_wrapper(self.lambda_cycle) * loss_cycle).item(),
                    loss_iden_A.item(), loss_iden_B.item(), (self.lambda_identity_wrapper(self.lambda_identity) * loss_iden).item(), loss_D_A.item(), loss_D_B.item(), loss_D_C.item(), loss_D_A_real.item(), loss_D_A_generated.item(),
                    loss_D_B_real.item(), loss_D_B_generated.item(), loss_D_C_real.item(), loss_D_C_generated.item(),
                    lr_G_curr, lr_D_curr, round(time_per_step)), file=open(log_training, 'a'))
                if clr_cyclegan:
                    print("clr_cyclegan:{:.4f}".format(loss_clr), file=open(log_training, 'a'))
                    if kldivloss_on_encoding:
                        print("kldivloss_on_encoding:{:.4f}".format(loss_kldiv), file=open(log_training, 'a'))
                time_start_step = time.time()
            # logging 2/n
            if ii == (len_valsampler - 2) and self.rank == 0:
                print_shell_cmd_output('nvidia-smi')
                print_shell_cmd_output(f'ps -Flww -p {os.getpid()}')
        if num_gpus > 1:
            dist.barrier()
        time_per_iteration = time.time() - time_start_iteration
        loss_G_A2B_mean = np.mean(G_A2B_lossvec)
        loss_G_A2B_std = np.std(G_A2B_lossvec)
        loss_G_B2A_mean = np.mean(G_B2A_lossvec)
        loss_G_B2A_std = np.std(G_B2A_lossvec)
        loss_G_B2C_mean = np.mean(G_B2C_lossvec)
        loss_G_B2C_std = np.std(G_B2C_lossvec)
        loss_cycle_mean = np.mean(cycle_lossvec)
        loss_CGAN_mean = np.mean(CGAN_lossvec)
        loss_CGAN_std = np.std(CGAN_lossvec)
        loss_cycle_std = np.std(cycle_lossvec)
        loss_iden_mean = np.mean(iden_lossvec)
        loss_iden_std = np.std(iden_lossvec)
        loss_D_A_mean = np.mean(D_A_lossvec)
        loss_D_A_std = np.std(D_A_lossvec)
        loss_D_B_mean = np.mean(D_B_lossvec)
        loss_D_B_std = np.std(D_B_lossvec)
        loss_D_C_mean = np.mean(D_C_lossvec)
        loss_D_C_std = np.std(D_C_lossvec)
        lr_G_curr = get_lr(self.G_optimizer)
        lr_D_curr = get_lr(self.D_optimizer)
        res = [round(_,4) for _ in [loss_G_A2B_mean, loss_G_A2B_std, loss_G_B2A_mean, loss_G_B2A_std, loss_G_B2C_mean, loss_G_B2C_std, loss_CGAN_mean, loss_CGAN_std, loss_cycle_mean, loss_cycle_std, loss_iden_mean, loss_iden_std,
                loss_D_A_mean, loss_D_A_std, loss_D_B_mean, loss_D_B_std, loss_D_C_mean, loss_D_C_std]]
        last_val_batch = {'real_A':real_A.cpu().numpy(), 'real_B':real_B.detach().cpu().numpy(), 'generated_A':generated_A.detach().cpu().numpy(),
                            'generated_B':generated_B.detach().cpu().numpy(), 'real_C':real_C.detach().cpu().numpy(), 'generated_C':generated_C.detach().cpu().numpy(), 'generated_C__sup':generated_C__sup.detach().cpu().numpy(),
                            'cycle_A':cycle_A.detach().cpu().numpy(), 'cycle_B':cycle_B.detach().cpu().numpy(),
                            'iden_A':iden_A.detach().cpu().numpy(), 'iden_B':iden_B.detach().cpu().numpy(),
                            'loss_G':loss_G.item(), 'loss_G_A2B_disc':loss_G_A2B_disc.item(), 'loss_G_B2A_disc':loss_G_B2A_disc.item(),
                            'loss_G_B2C_disc':loss_G_B2C_disc.item(), 'loss_CGAN':loss_CGAN.item(),
                            'loss_cycle':loss_cycle.item(), 'loss_iden':loss_iden.item(), 'loss_D':loss_D.item(), 'loss_D_A':loss_D_A.item(), 'loss_D_B':loss_D_B.item(), 'loss_D_C':loss_D_C.item()}
        return *res, round(lr_G_curr,7), round(lr_D_curr,7), time_per_iteration, last_val_batch

    def adjust_lr(self, curr_iteration, curr_step):
        if lr_warmup_epochs > 0 and curr_iteration <= lr_warmup_epochs:  # linear warmup
            assert curr_step <= len_trainsampler*lr_warmup_epochs, f'{lr_scheduler=} {curr_step=} {len_trainsampler=} {lr_warmup_epochs=}'
            new_lr_G = lr_min_G + curr_step*(lr_G-lr_min_G)/len_trainsampler
            new_lr_D = lr_min_D + curr_step*(lr_D-lr_min_D)/len_trainsampler
        elif lr_const_epochs > 0 and lr_warmup_epochs < curr_iteration <= lr_warmup_epochs + lr_const_epochs: # constant LR region after linear warmup
            assert curr_step <= len_trainsampler*(lr_warmup_epochs + lr_const_epochs), f'{lr_scheduler=} {curr_step=} {len_trainsampler=} {lr_warmup_epochs=} {lr_const_epochs=}'
            new_lr_G = lr_G
            new_lr_D = lr_D
        elif lr_scheduler == 'contLinearDecay':
            steps_done = len_trainsampler * (lr_warmup_epochs + lr_const_epochs)
            assert curr_step >= steps_done, f'{lr_scheduler=} {curr_step=} {steps_done=}'
            new_lr_G = max(lr_min_G, lr_G - (curr_step - steps_done)*(lr_G-lr_min_G)/(totalSteps_train - steps_done))
            new_lr_D = max(lr_min_D, lr_D - (curr_step - steps_done)*(lr_D-lr_min_D)/(totalSteps_train - steps_done))
        elif lr_scheduler == 'contCosineDecay':
            steps_done = len_trainsampler * (lr_warmup_epochs + lr_const_epochs)
            assert curr_step >= steps_done, f'{lr_scheduler=} {curr_step=} {steps_done=}'
            new_lr_G = (lr_G - lr_min_G)*math.cos((math.pi * (curr_step - steps_done))/(2 * (totalSteps_train - steps_done))) + lr_min_G
            new_lr_D = (lr_D - lr_min_D)*math.cos((math.pi * (curr_step - steps_done))/(2 * (totalSteps_train - steps_done))) + lr_min_D
            assert new_lr_G >= lr_min_G
            assert new_lr_D >= lr_min_D
        else:
            raise NotImplementedError(f'{lr_scheduler=}')
        # set new learning rates
        for gg in self.G_optimizer.param_groups:
            gg['lr'] = new_lr_G
        for gg in self.D_optimizer.param_groups:
            gg['lr'] = new_lr_D

    def save_model(self, curr_iteration, loss_mean_G_val, loss_mean_D_val, last_val_batch):
        if not self.rank == 0:
            return
        filepattern = os.path.join(dir_models, f'*.pt')
        states = glob.glob(filepattern)
        if len(states) != 0:
            fileToDelete = subprocess.run(f'ls -1tv {filepattern}', shell=True, check=True, stdout=subprocess.PIPE).stdout.decode("UTF-8").split()[-1]
            print(f'deleting: {fileToDelete}')
#            os.remove(fileToDelete)
        dict_to_save = {'G_A2B': self.G_A2B.state_dict(),
                        'G_B2A': self.G_B2A.state_dict(),
                        'G_B2C': self.G_B2C.state_dict(),
                        'D_A': self.D_A.state_dict(),
                        'D_B': self.D_B.state_dict(),
                        'D_C': self.D_C.state_dict(),
                        'G_optimizer': self.G_optimizer.state_dict(),
                        'D_optimizer': self.D_optimizer.state_dict(),
                        'loss_mean_G_val': loss_mean_G_val,
                        'loss_mean_D_val': loss_mean_D_val,
                        'last_val_batch': last_val_batch,
                        'lambda_CGAN': self.lambda_CGAN,
                        'lambda_cycle': self.lambda_cycle,
                        'lambda_identity': self.lambda_identity}
        if clr_cyclegan:
            dict_to_save = mergeDicts(dict_to_save, {'E_A':self.E_A.state_dict(), 'E_B':self.E_B.state_dict(), 'lambda_encode':self.lambda_encode})
            if kldivloss_on_encoding:
                dict_to_save = mergeDicts(dict_to_save, {'lambda_kldiv':self.lambda_kldiv})
        filename = os.path.join(dir_models, f'{curr_iteration}_{loss_mean_G_val}_{loss_mean_D_val}.pt')
        torch.save(dict_to_save, filename)

    def resume_model(self):
        filepattern = os.path.join(dir_models, f'*.pt')
        states = glob.glob(filepattern)
        if len(states) == 0:
            print(f"no match found : {filepattern} : training from beginning...")
            return 0
        else:
            filetoload = subprocess.run(f'ls -1tv {filepattern}', shell=True, check=True, stdout=subprocess.PIPE).stdout.decode("UTF-8").split()[-1]
            checkpoint = torch.load(filetoload, map_location=device_cpu)

            if num_gpus > 1 and not list(checkpoint['G_A2B'])[1].startswith('module.'):
                checkpoint['G_A2B'] = dict_rename(checkpoint['G_A2B'], remove=False)
            elif num_gpus <= 1 and list(checkpoint['G_A2B'])[1].startswith('module.'):
                checkpoint['G_A2B'] = dict_rename(checkpoint['G_A2B'], remove=True)
            print(list(checkpoint['G_A2B'])[1])
            self.G_A2B.load_state_dict(checkpoint['G_A2B'])

            if num_gpus > 1 and not list(checkpoint['D_A'])[1].startswith('module.'):
                checkpoint['D_A'] = dict_rename(checkpoint['D_A'], remove=False)
            elif num_gpus <= 1 and list(checkpoint['D_A'])[1].startswith('module.'):
                checkpoint['D_A'] = dict_rename(checkpoint['D_A'], remove=True)
            self.D_A.load_state_dict(checkpoint['D_A'])

            if num_gpus > 1 and not list(checkpoint['G_B2A'])[1].startswith('module.'):
                checkpoint['G_B2A'] = dict_rename(checkpoint['G_B2A'], remove=False)
            elif num_gpus <= 1 and list(checkpoint['G_B2A'])[1].startswith('module.'):
                checkpoint['G_B2A'] = dict_rename(checkpoint['G_B2A'], remove=True)
            self.G_B2A.load_state_dict(checkpoint['G_B2A'])

            if num_gpus > 1 and not list(checkpoint['D_B'])[1].startswith('module.'):
                checkpoint['D_B'] = dict_rename(checkpoint['D_B'], remove=False)
            elif num_gpus <= 1 and list(checkpoint['D_B'])[1].startswith('module.'):
                checkpoint['D_B'] = dict_rename(checkpoint['D_B'], remove=True)
            self.D_B.load_state_dict(checkpoint['D_B'])

            if num_gpus > 1 and not list(checkpoint['G_B2C'])[1].startswith('module.'):
                checkpoint['G_B2C'] = dict_rename(checkpoint['G_B2C'], remove=False)
            elif num_gpus <= 1 and list(checkpoint['G_B2C'])[1].startswith('module.'):
                checkpoint['G_B2C'] = dict_rename(checkpoint['G_B2C'], remove=True)
            self.G_B2C.load_state_dict(checkpoint['G_B2C'])

            if num_gpus > 1 and not list(checkpoint['D_C'])[1].startswith('module.'):
                checkpoint['D_C'] = dict_rename(checkpoint['D_C'], remove=False)
            elif num_gpus <= 1 and list(checkpoint['D_C'])[1].startswith('module.'):
                checkpoint['D_C'] = dict_rename(checkpoint['D_C'], remove=True)
            self.D_C.load_state_dict(checkpoint['D_C'])

            self.G_optimizer.load_state_dict(checkpoint['G_optimizer'])
            self.D_optimizer.load_state_dict(checkpoint['D_optimizer'])
            self.lambda_CGAN = checkpoint['lambda_CGAN']
            if learnable_cycle_loss:
                self.lambda_cycle.data = checkpoint['lambda_cycle'].data
            else:
                self.lambda_cycle = checkpoint['lambda_cycle']
            if learnable_identity_loss:
                self.lambda_identity.data = checkpoint['lambda_identity'].data
            else:
                self.lambda_identity = checkpoint['lambda_identity']
            if clr_cyclegan:
                self.E_A.load_state_dict(checkpoint['E_A'])
                self.E_B.load_state_dict(checkpoint['E_B'])
                self.lambda_encode = checkpoint['lambda_encode']
                if kldivloss_on_encoding:
                    self.lambda_kldiv = checkpoint['lambda_kldiv']
            last_state = int(filetoload.split('/')[-1].split('_')[0])
            print(f'resumed model with {filetoload=} {last_state=}')
            return last_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experimentID', type=str, default='1')
    parser.add_argument('--projectID', type=str, default='BWE-3')
    parser.add_argument('--Adatapath', type=str, default='./data')
    parser.add_argument('--Adataname', type=str, default='sre_alllangs')
    parser.add_argument('--Bdatapath', type=str, default='./data')
    parser.add_argument('--Bdataname', type=str, default='voxcelebcat_8k')
    parser.add_argument('--Cdatapath', type=str, default='./data')
    parser.add_argument('--Cdataname', type=str, default='voxcelebcat')
    parser.add_argument('--withSilenceTraining', type=lambda v: bool(distutils.util.strtobool(v)), default=True)
    parser.add_argument('--skip_origVADcheck', type=bool, default=True)
    parser.add_argument('--generation_style', type=str, default='wav2wav', choices=['wav2wav', 'feat2wav'])
    parser.add_argument('--class_generator', type=str, default='convtasnet')
    parser.add_argument('--class_discriminator', type=str, default='ParallelWaveGANDiscriminator')
    parser.add_argument('--pretrained_generator_path', type=str, default='')
    parser.add_argument('--lr_G', type=float, default=0.0002)
    parser.add_argument('--lr_D', type=float, default=0.0001)   # may wanna try doubling this
    parser.add_argument('--lr_min', type=float, default=1e-7)   # from old CycleGAN
    parser.add_argument('--lr_scheduler', type=str, default='contLinearDecay')
    parser.add_argument('--type_adv_loss', type=str, default='LSGAN', help='type of adversarial loss')
    parser.add_argument('--type_cycle_loss', type=str, default='l1', choices=['l1', 'l2', 'fm', 'mrstft'], help='type of supervised loss')
    parser.add_argument('--disable_detect_anamoly', action='store_true')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--sample_len_sec', type=float, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)   # lesser because there are multiple independent dataloaders
    parser.add_argument('--hrs_per_iter', type=float, default=50)
    parser.add_argument('--n_iterations', type=int, default=15, help="let's say iteration=epoch for this work")
    parser.add_argument('--fs', type=int, default=16000)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--device_training', type=str, default='cuda')
    parser.add_argument('--disable_pin_memory', action='store_true')
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--lambda_cycle', type=float, default=1)
    parser.add_argument('--lambda_identity', type=float, default=1)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--audio_scale', type=float, default=2**15-1)
    parser.add_argument('--D_update_periodicity', type=int, default=2)  # needs tuning
    parser.add_argument('--G_update_periodicity', type=int, default=1)  # needs tuning
    parser.add_argument('--GD_algo', type=str, default='alternative', choices=['alternative', 'simultaneous'])     # needs to tune: simulataneous should be significantly faster
    parser.add_argument('--log_periodicity_steps', type=int, default=12)
    parser.add_argument('--rstrip_key', type=str, default='-8k', help='what extra token is present in data A identifier')
    parser.add_argument('--rstrip_from2', action='store_true', help='whether the extra token is in data B identifier instead of A')
    parser.add_argument('--disable_dataloader_len_adjust', action='store_true')
    parser.add_argument('--separate_scaler_per_model', action='store_true', help='TODO')
    parser.add_argument('--pwg_disc_conv_channels', type=int, default=80)
    parser.add_argument('--LSGAN_target_offset', type=float, default=0)
    parser.add_argument('--class_optimizer', type=str, default='adam')
    parser.add_argument('--add_noise_to_G_input', action='store_true')
    parser.add_argument('--add_noise_to_D_input', action='store_true')
    parser.add_argument('--append_sigmoid_to_discriminator', action='store_true')
    parser.add_argument('--pwg_layers', type=int, default=10)
    parser.add_argument('--ctn_layer', type=int, default=8)
    parser.add_argument('--subbatch_size', type=int, default=0)
    parser.add_argument('--GaussianNoise_sigma', type=float, default=0.1)
    parser.add_argument('--metric_criterion', type=str, default='l1')
    parser.add_argument('--hifi_D_out_chs_multiplier', type=int, default=4)
    parser.add_argument('--hifi_D_channels', type=int, default=32)
    parser.add_argument('--hifi_D_scale_channels', type=int, default=16)
    parser.add_argument('--hifi_D_period_channels', type=int, default=4)
    parser.add_argument('--cycle_loss_AtoB_ratio', type=int, default=1)
    parser.add_argument('--clr_cyclegan', action='store_true')
    parser.add_argument('--dim_noise', type=int, default=64)
    parser.add_argument('--std_noise', type=float, default=0.01)
    parser.add_argument('--lambda_encode', type=float, default=1)
    parser.add_argument('--lambda_kldiv', type=float, default=1)
    parser.add_argument('--kldivloss_on_encoding', action='store_true')
    parser.add_argument('--bicyclegan', action='store_true')
    parser.add_argument('--identical_encoding_nws', action='store_true')
    parser.add_argument('--bicyclegan_ver', type=str, default='v1')
    parser.add_argument('--disable_kldivloss_on_encoding', action='store_true')
    parser.add_argument('--identity_discriminator', action='store_true')
    parser.add_argument('--ctn_kernel', type=int, default=3)
    parser.add_argument('--ctn_enc_dim', type=int, default=128)
    parser.add_argument('--ctn_feature_dim', type=int, default=128)
    parser.add_argument('--ctn_TCN_dilationFactor', type=int, default=2)
    parser.add_argument('--correct_optimzero', action='store_true')
    parser.add_argument('--lr_warmup_epochs', type=int, default=0)
    parser.add_argument('--lr_const_epochs', type=int, default=0)
    parser.add_argument('--do_last_iteration', action='store_true')
    parser.add_argument('--lr_min_autoscaling', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--learnable_identity_loss', action='store_true')
    parser.add_argument('--learnable_cycle_loss', action='store_true')
    parser.add_argument('--learnable_loss_wrapper_fxn', type=str, default='softplus')
    parser.add_argument('--learnable_loss_rand_init', action='store_true')
    parser.add_argument('--use_PowerSGD', action='store_true')
    parser.add_argument('--sigmoid_in_disc_loss_calc', action='store_true')
    parser.add_argument('--topk_in_DCL', action='store_true')
    parser.add_argument('--topk_in_DCL_smallest', action='store_true')
    parser.add_argument('--topk_in_DCL_perc', type=float, default=50)
    parser.add_argument('--type_loss_clr', type=str, default='l2')
    parser.add_argument('--adjust_dataloader_len_up', action='store_true')
    parser.add_argument('--p_B2A', type=float, default=0)
    parser.add_argument('--B2A_preproc_dir', type=str, default='')
    parser.add_argument('--B2A_keepA', action='store_true')
    parser.add_argument('--B2A_preproc_domain', type=str, default='B', choices=['C', 'B'])
    parser.add_argument('--B2A_removeC', action='store_true')
    parser.add_argument('--lambda_CGAN', type=float, default=1)
    parser.add_argument('--skip_do_validate', action='store_true')
    #
    args = parser.parse_args()
    dict_args = vars(args)
    print(dict_args)
    #
    experimentID = args.experimentID
    projectID = args.projectID
    Adatapath = args.Adatapath
    Adataname = args.Adataname
    Bdatapath = args.Bdatapath
    Bdataname = args.Bdataname
    Cdatapath = args.Cdatapath
    Cdataname = args.Cdataname
    withSilenceTraining = args.withSilenceTraining
    skip_origVADcheck = args.skip_origVADcheck
    generation_style = args.generation_style
    class_generator = args.class_generator
    class_discriminator = args.class_discriminator
    pretrained_generator_path = args.pretrained_generator_path
    lr_G = args.lr_G
    lr_D = args.lr_D
    lr_min = args.lr_min
    lr_scheduler = args.lr_scheduler
    type_adv_loss = args.type_adv_loss
    type_cycle_loss = args.type_cycle_loss
    disable_detect_anamoly = args.disable_detect_anamoly
    num_gpus = args.num_gpus
    disable_wandb = args.disable_wandb
    sample_len_sec = args.sample_len_sec
    batch_size = args.batch_size
    num_workers = args.num_workers
    hrs_per_iter = args.hrs_per_iter
    n_iterations = args.n_iterations
    fs = args.fs
    use_amp = args.use_amp
    device_training = args.device_training
    disable_pin_memory = args.disable_pin_memory
    prefetch_factor = args.prefetch_factor
    lambda_cycle = args.lambda_cycle
    lambda_identity = args.lambda_identity
    adam_beta2 = args.adam_beta2
    audio_scale = args.audio_scale
    D_update_periodicity = args.D_update_periodicity
    G_update_periodicity = args.G_update_periodicity
    GD_algo = args.GD_algo
    log_periodicity_steps = args.log_periodicity_steps
    rstrip_key = args.rstrip_key
    rstrip_from2 = args.rstrip_from2
    disable_dataloader_len_adjust = args.disable_dataloader_len_adjust
    separate_scaler_per_model = args.separate_scaler_per_model
    pwg_disc_conv_channels = args.pwg_disc_conv_channels
    LSGAN_target_offset = args.LSGAN_target_offset
    class_optimizer = args.class_optimizer
    add_noise_to_G_input = args.add_noise_to_G_input
    add_noise_to_D_input = args.add_noise_to_D_input
    append_sigmoid_to_discriminator = args.append_sigmoid_to_discriminator
    pwg_layers = args.pwg_layers
    ctn_layer = args.ctn_layer
    subbatch_size = args.subbatch_size
    GaussianNoise_sigma = args.GaussianNoise_sigma
    metric_criterion = args.metric_criterion
    hifi_D_out_chs_multiplier = args.hifi_D_out_chs_multiplier
    hifi_D_channels = args.hifi_D_channels
    hifi_D_scale_channels = args.hifi_D_scale_channels
    hifi_D_period_channels = args.hifi_D_period_channels
    cycle_loss_AtoB_ratio = args.cycle_loss_AtoB_ratio
    clr_cyclegan = args.clr_cyclegan
    dim_noise = args.dim_noise
    std_noise = args.std_noise
    lambda_encode = args.lambda_encode
    lambda_kldiv = args.lambda_kldiv
    kldivloss_on_encoding = args.kldivloss_on_encoding
    bicyclegan = args.bicyclegan
    identical_encoding_nws = args.identical_encoding_nws
    bicyclegan_ver = args.bicyclegan_ver
    disable_kldivloss_on_encoding = args.disable_kldivloss_on_encoding
    identity_discriminator = args.identity_discriminator
    ctn_kernel = args.ctn_kernel
    ctn_enc_dim = args.ctn_enc_dim
    ctn_feature_dim = args.ctn_feature_dim
    ctn_TCN_dilationFactor = args.ctn_TCN_dilationFactor
    correct_optimzero = args.correct_optimzero
    lr_warmup_epochs = args.lr_warmup_epochs
    lr_const_epochs = args.lr_const_epochs
    do_last_iteration = args.do_last_iteration
    lr_min_autoscaling = args.lr_min_autoscaling
    local_rank = args.local_rank
    learnable_identity_loss = args.learnable_identity_loss
    learnable_cycle_loss = args.learnable_cycle_loss
    learnable_loss_wrapper_fxn = args.learnable_loss_wrapper_fxn
    learnable_loss_rand_init = args.learnable_loss_rand_init
    use_PowerSGD = args.use_PowerSGD
    sigmoid_in_disc_loss_calc = args.sigmoid_in_disc_loss_calc
    topk_in_DCL = args.topk_in_DCL
    topk_in_DCL_smallest = args.topk_in_DCL_smallest
    topk_in_DCL_perc = args.topk_in_DCL_perc
    type_loss_clr = args.type_loss_clr
    adjust_dataloader_len_up = args.adjust_dataloader_len_up
    p_B2A = args.p_B2A
    B2A_preproc_dir = args.B2A_preproc_dir
    B2A_keepA = args.B2A_keepA
    B2A_preproc_domain = args.B2A_preproc_domain
    B2A_removeC = args.B2A_removeC
    lambda_CGAN = args.lambda_CGAN
    skip_do_validate = args.skip_do_validate
    #
    dir_models = os.path.join('models', projectID, experimentID)
    mkdir_safe(dir_models)
    gpu_id = local_rank
    #
    args_file = os.path.join(dir_models, 'args.yaml')
    write_yaml(dict_args, args_file, overwrite=False)

    # some checks on args
    assert experimentID and int(experimentID)
    assert log_periodicity_steps % D_update_periodicity == 0, 'otherwise logging on disk and cloud (wandb) will be problematic'
    assert log_periodicity_steps % G_update_periodicity == 0
    assert log_periodicity_steps == 1 or log_periodicity_steps % 2 == 0, 'for validity with subbatch (gradient accumulation)'
    assert D_update_periodicity > 0
    assert G_update_periodicity > 0
    assert subbatch_size == 0 or (batch_size % subbatch_size == 0), f'{batch_size=} {subbatch_size=}; wont be able to partition'
    assert not (kldivloss_on_encoding and not clr_cyclegan)
    assert not (bicyclegan and not clr_cyclegan)
    assert 0 <= p_B2A <= 1, f'{p_B2A=}. If ==1, then use train_cycleGAN_parallel_with_target.py'
    assert not (0 < p_B2A <= 1 and not B2A_preproc_dir), f'{B2A_preproc_dir=} is needed since {p_B2A=} > 0'
    assert not (p_B2A == 0 and B2A_preproc_dir), f'{B2A_preproc_dir=} should not be provided'

    # post modification of args
    if withSilenceTraining:
        datapath_token = '_proc_audio'
    else:
        datapath_token = '_proc_audio_no_sil'   # by default, we train without silence (i.e. silence is removed via VAD)
    Adatapath = os.path.join(Adatapath, Adataname)
    Bdatapath = os.path.join(Bdatapath, Bdataname)
    Cdatapath = os.path.join(Cdatapath, Cdataname)
    AorigVAD = os.path.join(Adatapath, 'vad.scp')
    BorigVAD = os.path.join(Bdatapath, 'vad.scp')
    CorigVAD = os.path.join(Cdatapath, 'vad.scp')
    if not skip_origVADcheck:
        assert not (not withSilenceTraining and getcol(AorigVAD, n=2) != getcol(BorigVAD, n=2)), 'source VAD mismatch so there might be alignment problem when training with silence frames'
    Adatapath = Adatapath + datapath_token
    Bdatapath = Bdatapath + datapath_token
    Cdatapath = Cdatapath + datapath_token
    assert dir_exists_and_notempty(Adatapath)
    assert dir_exists_and_notempty(Bdatapath)
    assert dir_exists_and_notempty(Cdatapath)
    if subbatch_size == 0:
        n_subbatches = 1
        subbatch_size = batch_size
    else:
        n_subbatches = batch_size // subbatch_size
    num_workers = min(num_workers, subbatch_size)
    assert num_workers <= 4
    if GD_algo == 'simultaneous':
        assert G_update_periodicity == D_update_periodicity == 1, f'{G_update_periodicity =} {D_update_periodicity =}'
    if p_B2A == 0:  # over-write B2A_preproc_dir if provided
        B2A_preproc_dir = ''

    # global settings
    torch.autograd.set_detect_anomaly(not disable_detect_anamoly)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.backends.cudnn.benchmark = True
    np.seterr(all='raise')  # should protect against some naninf issues
    device_cpu = torch.device('cpu')
    is_cuda_available = torch.cuda.is_available()
    print(f'{is_cuda_available=}')
    if num_gpus >= 1 and device_training in ['gpu', 'cuda']:
        assert is_cuda_available
        ddp_args = {'num_gpus': num_gpus, 'node_id': 0, 'num_nodes': 1, 'master_addr': 'localhost', 'master_port': '1238'}
        device, rank, world_size = ddp.ddp_init(gpu_id, **ddp_args)
#        device = open_device(num_gpus=num_gpus)
    else:
        if is_cuda_available:
            warn(f'CUDA should not be available ideally when training on CPU')
        device = device_cpu
        disable_pin_memory = True
        disable_wandb = True
        assert not use_amp
    set_randomseed(seed=0)
    multiprocessing.set_start_method('forkserver')

    # misc defs
    log_training = os.path.join(dir_models, 'log_training.txt')
    log_progress = os.path.join(dir_models, 'log_progress.txt')
    scaler = GradScaler(enabled=use_amp)
    if clr_cyclegan:
        G_support_noise = True
    else:
        G_support_noise = False
    lr_min_G = lr_min
    if lr_min_autoscaling:
        lr_min_D = lr_min*lr_D/lr_G
    else:
        lr_min_D = lr_min
    learnable_loss_wrapper_fxn_dict = {'softplus': nn.Softplus(), 'relu': nn.ReLU(), 'identity': nn.Identity()}

    # datasets
    ADargs = {'return_key': False, 'return_class': False, 'min_chunk_length': sample_len_sec, 'max_chunk_length': sample_len_sec, 'aug_cfg': None}
    AD2args = ADargs | {'rstrip_from2': rstrip_from2}

    Atraindatascp = os.path.join(Adatapath, 'lists_xvec/train.scp')
    Btraindatascp = os.path.join(Bdatapath, 'lists_xvec/train.scp')
    Ctraindatascp = os.path.join(Cdatapath, 'lists_xvec/train.scp')
    paired_traindata = AD2(os.path.join(Bdatapath, 'wav.scp'), os.path.join(Cdatapath, 'wav.scp'), Btraindatascp, Ctraindatascp,
                    time_durs_file=os.path.join(Bdatapath, 'utt2dur'), time_durs_file2=os.path.join(Cdatapath, 'utt2dur'), rstrip_key=rstrip_key, **AD2args)
    unpaired_traindata = AD(os.path.join(Adatapath, 'wav.scp'), Atraindatascp, time_durs_file=os.path.join(Adatapath, 'utt2dur'), **ADargs)

    Avaldatascp = os.path.join(Adatapath, 'lists_xvec/val.scp')
    Bvaldatascp = os.path.join(Bdatapath, 'lists_xvec/val.scp')
    Cvaldatascp = os.path.join(Cdatapath, 'lists_xvec/val.scp')
    paired_valdata = AD2(os.path.join(Bdatapath, 'wav.scp'), os.path.join(Cdatapath, 'wav.scp'), Bvaldatascp, Cvaldatascp,
                    time_durs_file=os.path.join(Bdatapath, 'utt2dur'), time_durs_file2=os.path.join(Cdatapath, 'utt2dur'), rstrip_key=rstrip_key, **AD2args)
    unpaired_valdata = AD(os.path.join(Adatapath, 'wav.scp'), Avaldatascp, time_durs_file=os.path.join(Adatapath, 'utt2dur'), **ADargs)

    # samplers
    Samplerargs_train = {'batch_size': batch_size, 'var_batch_size': False, 'iters_per_epoch': 1.0, 'num_egs_per_class': 1, 'num_egs_per_utt': 1}
    Samplerargs_val = {'batch_size': subbatch_size, 'var_batch_size': False, 'iters_per_epoch': 1.0, 'num_egs_per_class': 1, 'num_egs_per_utt': 1}

    paired_trainsampler = Sampler(paired_traindata, **Samplerargs_train)
    unpaired_trainsampler = Sampler(unpaired_traindata, **Samplerargs_train)
    paired_valsampler = Sampler(paired_valdata, **Samplerargs_val)
    unpaired_valsampler = Sampler(unpaired_valdata, **Samplerargs_val)

    if adjust_dataloader_len_up:
        dataloader_len_adjust_fxn = max
    else:
        dataloader_len_adjust_fxn = min
    if disable_dataloader_len_adjust:
        len_trainsampler = dataloader_len_adjust_fxn(paired_trainsampler._len, unpaired_trainsampler._len)
        len_valsampler = dataloader_len_adjust_fxn(paired_valsampler._len, unpaired_valsampler._len)
        paired_trainsampler._len = len_trainsampler
        unpaired_trainsampler._len = len_trainsampler
        paired_valsampler._len = len_valsampler
        unpaired_valsampler._len = len_valsampler
    else:   # same length of both samplers is important
        len_trainsampler = math.floor(3600*hrs_per_iter/(batch_size*sample_len_sec))
        paired_trainsampler._len = len_trainsampler
        unpaired_trainsampler._len = len_trainsampler
        len_valsampler = dataloader_len_adjust_fxn(math.floor(len_trainsampler/10), paired_valsampler._len, unpaired_valsampler._len)
        paired_valsampler._len = len_valsampler
        unpaired_valsampler._len = len_valsampler
    print(f'{len_trainsampler=} {len_valsampler=}')
    assert len_trainsampler > 0
    assert len_valsampler > 0
    totalSteps_train = n_iterations*len_trainsampler    # total means all training
    totalSteps_val = n_iterations*len_valsampler

    # dataloaders
    loaderArgs = {'num_workers': num_workers, 'pin_memory': not disable_pin_memory, 'prefetch_factor': prefetch_factor}

    paired_trainloader = torch.utils.data.DataLoader(paired_traindata, batch_sampler=paired_trainsampler, **loaderArgs)
    paired_valloader = torch.utils.data.DataLoader(paired_valdata, batch_sampler=paired_valsampler, **loaderArgs)
    unpaired_trainloader = torch.utils.data.DataLoader(unpaired_traindata, batch_sampler=unpaired_trainsampler, **loaderArgs)
    unpaired_valloader = torch.utils.data.DataLoader(unpaired_valdata, batch_sampler=unpaired_valsampler, **loaderArgs)

    #
    bwe = BWEtrainer_CycleGAN()
    bwe.do_train()
