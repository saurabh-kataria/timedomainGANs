#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''unpaired learning
TODO:
    1. add backuping of log file
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
#import bitsandbytes as bnb
import multiprocessing
import multiprocessing as mp

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
import fairseq
from fairseq.dataclass import FairseqDataclass
from examples.data2vec.models.data2vec2 import Data2VecMultiModel, Data2VecMultiConfig
from examples.data2vec.data.modality import Modality
sys.path.append('/home/hltcoe/skataria/unilm/wavlm')
from WavLM import WavLM, WavLMConfig
sys.path.append('./RawNet/python/RawNet3')
from models_rw.RawNet3 import RawNet3
from models_rw.RawNetBasicBlock import Bottle2neck
from UniSpeech.downstreams.speaker_verification.models.ecapa_tdnn import ECAPA_TDNN_SMALL


#class SSLFeatureExtractor(nn.Module):
#    def __init__(self, class_ssl='wav2vec2', normalize_audio=False, w2v2_ver='v2'):
#        super().__init__()
#        self.normalize_audio = normalize_audio
#        if class_ssl == 'wav2vec2':
#            self.w2v2_ver = w2v2_ver
#            cp = '/exp/skataria/segan2/vox/models_pretrained/wav2vec_small.pt'
#            model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp])
#            model = model[0]
#            model = model.eval()
#            self.model = model
#        else:
#            raise NotImplementedError(f'{class_ssl=}')
#    def forward2(self, x):
#        x = self.model(x, features_only=True, mask=False)['x']
#        if self.w2v2_ver == 'v1':
#            x = self.model.final_proj(x)
#        return x
#    def forward(self, x):
##        raise Exception(f'{x.shape=} {x.max()=} {x.min()=}')
#        if len(x.shape) == 3:
#            x = x.squeeze(1)
#            assert len(x.shape) == 2
#        if self.normalize_audio:
#            x = x / (2**15 - 1)
#        x_mu = x.mean(1, keepdim=True).expand_as(x)
#        x_std = x.std(1, keepdim=True).expand_as(x)
#        x = (x - x_mu) / (x_std + EPS)
#        with torch.no_grad():
##            x = self.model(x)['x_final_proj']
#            x = self.forward2(x)
#        return x#.clone()


class SSLFeatureExtractor(nn.Module):
    def __init__(self, class_ssl='wav2vec2', normalize_audio=False, w2v2_ver='v2', film_wav2vec2_mpath='models_pretrained/wav2vec_small.pt', preproc_SSL=False,
                film_ssl_weightedsum=False, film_ssl_wsum_learnable=False, film_ssl_layer=0):
        super().__init__()
        self.normalize_audio = normalize_audio
        self.class_ssl = class_ssl
        self.film_wav2vec2_mpath = film_wav2vec2_mpath
        self.preproc_SSL = preproc_SSL
        self.film_ssl_weightedsum = film_ssl_weightedsum
        self.film_ssl_wsum_learnable = film_ssl_wsum_learnable
        self.film_ssl_layer = film_ssl_layer
        if self.class_ssl == 'wav2vec2':
            self.w2v2_ver = w2v2_ver
            model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([film_wav2vec2_mpath])
            model = model[0]
        elif self.class_ssl == 'data2vec2':
            self.mod = Modality.AUDIO
            ckpt = torch.load(film_data2vec2_mpath, map_location=torch.device('cpu'))
            if 'multirun' in film_data2vec2_mpath:
                pass
            else:
                try:
                    del ckpt['model']['modality_encoders.AUDIO.alibi_scale']
                except Exception as e:
                    print(f'{e=}')
            model = Data2VecMultiModel(Data2VecMultiConfig(FairseqDataclass(ckpt['cfg'])), [self.mod])
            model.load_state_dict(ckpt['model'], strict=True)
        elif self.class_ssl == 'wavlm':
            checkpoint = torch.load(film_wavlm_mpath, map_location=torch.device('cpu'))
            cfg = WavLMConfig(checkpoint['cfg'])
            model = WavLM(cfg)
            model.load_state_dict(checkpoint['model'])
        elif self.class_ssl == 'hubert':
            model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([film_hubert_mpath])
            model = model[0]
        elif self.class_ssl == 'xvec':
            aux_state_dict = torch.load(aux_path, map_location=torch.device('cpu'))
            try:
                aux_state_dict = aux_state_dict['model_state_dict']
            except:
                aux_state_dict = aux_state_dict['nw_emb']
            if list(aux_state_dict)[1].startswith('module.'):
                aux_state_dict = dict_rename(aux_state_dict)
            model = XVec(**mergeDicts(xvec_args, {'num_classes': aux_state_dict['classif_net.output.kernel'].shape[1]}))
            model.load_state_dict(aux_state_dict)
        elif self.class_ssl == 'wavlm_asv':
            model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large')
            state_dict = torch.load('models_pretrained/wavlm_large_finetune.pth', map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict['model'], strict=False)
        else:
            raise NotImplementedError(f'{self.class_ssl=}')
        model = model.eval()
        self.model = model
        if self.preproc_SSL:
            self.model_preproc_SSL = get_preproc_SSL().to(device)

    def forward2(self, x):
        if self.class_ssl == 'wav2vec2':
            if self.film_ssl_weightedsum or self.film_ssl_wsum_learnable:
                x = self.model(x, features_only=True, mask=None)['layer_results']
                x = torch.stack([x[ii][0].transpose(0,1) for ii in range(len(x))], axis=-1)
                if self.film_ssl_wsum_learnable: return x
                else: x = torch.mean(x, axis=-1)
            else:
                x = self.model(x, features_only=True, mask=None, layer=self.film_ssl_layer if self.film_ssl_layer > 0 else None)['x']
                if self.w2v2_ver == 'v1':
                    x = self.model.final_proj(x)
        elif self.class_ssl == 'data2vec2':
            x = self.model(x, mask=None, features_only=True, mode=self.mod)['x']
        elif self.class_ssl == 'wavlm':
            if self.film_ssl_weightedsum or self.film_ssl_wsum_learnable:
                x = self.model.extract_features(x, ret_layer_results=True, output_layer=-1)
                x = x[0][1] # extra step
                x = torch.stack([x[ii][0].transpose(0,1) for ii in range(len(x))], axis=-1)
                if self.film_ssl_wsum_learnable: return x
                else: x = torch.mean(x, axis=-1)
            else:
                x = self.model.extract_features(x)[0]
        elif self.class_ssl == 'hubert':
            x = self.model(x, mask=None, features_only=True)['x']
        elif self.class_ssl == 'xvec':
            if film_ssl_emb_feats:
                x = self.model.extract_embed(feat_extractor(x))
            else:
                x = get_intermediate_activations(self.model, x, n=4, use_lastDFLactivation=True)[0]
                x = x.reshape(x.shape[0], -1, x.shape[-1]).transpose(1,2)
        elif self.class_ssl == 'wavlm_asv':
            if film_ssl_emb_feats:
                x = self.model(x)
            else:
                x = self.model.get_feat(x).transpose(1,2)
        else:
            raise NotImplementedError(f'{self.class_ssl=}')
        return x

    def forward(self, x):
        with autocast(enabled=False) and torch.no_grad():
            if len(x.shape) == 4:
                assert x.shape[1] == 1, f'{x.shape=} {x.max()=} {x.min()=}'
                x = x.squeeze(1)
            if len(x.shape) == 3:
                assert x.shape[1] == 1, f'{x.shape=} {x.max()=} {x.min()=}'
                x = x.squeeze(1)
            assert len(x.shape) == 2, f'{x.shape=} {x.max()=} {x.min()=}'
    #        raise Exception(f'{x.shape=} {x.max()=} {x.min()=}')
            if self.preproc_SSL:
                x = do_preproc_SSL(self.model_preproc_SSL, x)
            if self.normalize_audio:
                x = x / (2**15 - 1)
            if self.class_ssl not in ['xvec']:
                x_mu = x.mean(1, keepdim=True).expand_as(x)
                x_std = x.std(1, keepdim=True).expand_as(x)
                x = (x - x_mu) / (x_std + EPS)
#            with torch.no_grad():
#            x = self.model(x)['x_final_proj']
            x = self.forward2(x)
        return x#.clone()


#def get_generator(rank=0):
#    if class_generator == 'convtasnet':
#        G = TasNet(num_spk=1, layer=ctn_layer, enc_dim=ctn_enc_dim, stack=ctn_stack, kernel=ctn_kernel, win=1, TCN_dilationFactor=TCN_dilationFactor, feature_dim=ctn_feature_dim, masks_type='mul', audio_scale=audio_scale, masking_nonlinearity='sigmoid', support_noise=G_support_noise, dim_noise=dim_noise, std_noise=std_noise,
#            film_do=film_do, film_ver=film_ver, film_d_embed=film_d_embed, film_d_embed_interim=film_d_embed_interim, film_type_pooling=film_type_pooling)
#    elif class_generator == 'demucs':
#        G = Demucs(causal=False, hidden=15, device=device)
#    else:
#        raise NotImplementedError(f'{class_generator=}')
#    out = [G]
#    if add_noise_to_G_input:
#        out = [GaussianNoise(sigma=GaussianNoise_sigma)] + out
#    if len(out) > 1:
#        G = nn.Sequential(*out)
#    else:
#        G = out[0]
#    if rank == 0:
#        print(G)
#        print(f'{get_ntrainableparams_nn(G)=}')
#    return G

def get_generator(rank=0):
    if class_generator == 'convtasnet':
        G = TasNet(num_spk=1, layer=ctn_layer, enc_dim=ctn_enc_dim, feature_dim=ctn_feature_dim, stack=1, kernel=3, win=1, TCN_dilationFactor=TCN_dilationFactor, masks_type='mul',
                   audio_scale=audio_scale, masking_nonlinearity='sigmoid', film_do=film_do, film_ver=film_ver, film_d_embed=film_d_embed, film_d_embed_interim=film_d_embed_interim,
                   film_type_pooling=film_type_pooling, film_ssl_wsum_learnable=film_ssl_wsum_learnable, film_ssl_nlayers=film_ssl_nlayers, film_ssl_wsum_actfxn=film_ssl_wsum_actfxn,
                    film_alpha=film_alpha)
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
        D = ParallelWaveGANDiscriminator(conv_channels=pwg_disc_conv_channels, layers=pwg_layers,
                film_do=film_D_do, film_ver=film_D_ver, film_d_embed=film_D_d_embed, film_d_embed_interim=film_D_d_embed_interim, film_type_pooling=film_D_type_pooling,
                film_ssl_wsum_learnable=film_ssl_wsum_learnable, film_ssl_nlayers=film_ssl_nlayers, film_ssl_wsum_actfxn=film_ssl_wsum_actfxn,
                film_alpha=film_alpha)
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
        out = out + [nn.Sigmoid()]
    if len(out) > 1:
        D = nn.Sequential(*out)
    else:
        D = out[0]
    if rank == 0:
        print(D)
        print(f'{get_ntrainableparams_nn(D)=}')
    return D


def get_latent_discriminator(rank=0):
    if class_latent_disc == 'custom1':
        D = nn.Sequential(nn.Linear(dim_noise, dim_LD_interim), nn.BatchNorm1d(dim_LD_interim), nn.ReLU(), nn.Linear(dim_LD_interim, dim_LD_interim), nn.BatchNorm1d(dim_LD_interim), nn.ReLU(), nn.Linear(dim_LD_interim, dim_LD_interim))
    elif class_latent_disc == 'ParallelWaveGANDiscriminator':
        D = ParallelWaveGANDiscriminator(conv_channels=pwg_disc_conv_channels, layers=pwg_layers)
    elif class_latent_disc == 'ResidualParallelWaveGANDiscriminator':
        D = ResidualParallelWaveGANDiscriminator(layers=pwg_layers, stacks=pwg_layers//10)
    elif class_latent_disc == 'MelGANDiscriminator':
        D = MelGANDiscriminator()
    elif class_latent_disc == 'MelGANMultiScaleDiscriminator':
        D = MelGANMultiScaleDiscriminator()
    elif class_latent_disc == 'HiFiGANPeriodDiscriminator':
        D = HiFiGANPeriodDiscriminator(out_chs_multiplier=hifi_D_out_chs_multiplier, channels=hifi_D_channels)
    elif class_latent_disc == 'HiFiGANMultiPeriodDiscriminator':
        discriminator_params = HiFiGANMultiPeriodDiscriminator().discriminator_params
        discriminator_params['channels'] = hifi_D_channels
        discriminator_params['film_do'] = film_D_do
        discriminator_params['film_ver'] = film_D_ver
        discriminator_params['film_d_embed'] = film_D_d_embed
        discriminator_params['film_d_embed_interim'] = film_D_d_embed_interim
        discriminator_params['film_type_pooling'] = film_D_type_pooling
        D = HiFiGANMultiPeriodDiscriminator(discriminator_params=discriminator_params)
    elif class_latent_disc == 'HiFiGANScaleDiscriminator':
        D = HiFiGANScaleDiscriminator(channels=hifi_D_channels)
    elif class_latent_disc == 'HiFiGANMultiScaleDiscriminator':
        discriminator_params = HiFiGANMultiScaleDiscriminator().discriminator_params
        discriminator_params['channels'] = hifi_D_channels
        D = HiFiGANMultiScaleDiscriminator(discriminator_params=discriminator_params)
    elif class_latent_disc == 'HiFiGANMultiScaleMultiPeriodDiscriminator':
        scale_discriminator_params = HiFiGANMultiScaleMultiPeriodDiscriminator().scale_discriminator_params
        scale_discriminator_params['channels'] = hifi_D_scale_channels
        period_discriminator_params = HiFiGANMultiScaleMultiPeriodDiscriminator().period_discriminator_params
        period_discriminator_params['channels'] = hifi_D_period_channels
        D = HiFiGANMultiScaleMultiPeriodDiscriminator(scale_discriminator_params=scale_discriminator_params,period_discriminator_params=period_discriminator_params)
    elif class_latent_disc == 'StyleMelGANDiscriminator':
        D = StyleMelGANDiscriminator()
    else:
        raise NotImplementedError(f'{class_latent_disc=}')
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


#def get_criterion():
#    if metric_criterion == 'l1':
#        c = nn.L1Loss()
#    elif metric_criterion == 'l2':
#        c = nn.MSELoss()
#    elif metric_criterion == 'mrstft':
#        c = MultiResolutionSTFTLoss(factor_sc=0.5,factor_mag=0.5)
#    else:
#        raise NotImplementedError(f'{metric_criterion=}')
#    return c


def get_criterion():
    'get list of loss fxn criterions for supervised loss calculation'
    assert type_sup_loss
    c = []
    for curr_type_sup_loss in type_sup_loss.split(','):
        if curr_type_sup_loss == 'l1':
            c.append(nn.L1Loss())
        elif curr_type_sup_loss == 'l2':
            c.append(nn.MSELoss())
        elif curr_type_sup_loss == 'mrstft':
            c.append(MultiResolutionSTFTLoss(factor_sc=0.5, factor_mag=0.5))
        elif curr_type_sup_loss in ['fm', 'afm']:
            c.append(FMloss)
        else:
            raise NotImplementedError(f'{curr_type_sup_loss=}')
        try:
            c[-1] = c[-1].to(device)
        except Exception as e:
            print(e)
    return c


def get_criterion_base():
    if metric_criterion == 'l1':
        c = nn.L1Loss()
    elif metric_criterion == 'l2':
        c = nn.MSELoss()
    else:
        raise NotImplementedError(f'{metric_criterion=}')
    return c


def FMloss(x, y, model=None, metric=None, intermediate_extraction=False):
    assert model is not None
    if intermediate_extraction:
        a_x = get_intermediate_activations(model, x)
        a_y = get_intermediate_activations(model, y)
    else:
        a_x = model(x)
        a_y = model(y)
    assert isinstance(a_x, list), 'atleast len 1 list of activations expected'
    return torch_calc_error(a_x, a_y, metric)


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


def get_auxModel(aux_path, afm_class):
#    assert class_cycleCriterion or class_identityCriterion or use_SemanticConsistencyLoss or use_CycleSemanticConsistencyLoss
    if afm_class == 'xvec':
        assert aux_path
        if aux_path.split('/')[-1].endswith('.pth'):
            aux_path_full = aux_path
        else:
            assert os.path.isdir(aux_path)
            filepattern = os.path.join(aux_path, f'*.pth')
            states = glob.glob(filepattern)
            assert len(states) > 0
            aux_path_full = subprocess.run(f'ls -1tv {filepattern}', shell=True, check=True, stdout=subprocess.PIPE).stdout.decode("UTF-8").split()[-1]
        aux_state_dict = torch.load(aux_path_full, map_location=device_cpu)
        try:
            aux_state_dict = aux_state_dict['model_state_dict']
        except:
            aux_state_dict = aux_state_dict['nw_emb']
        if list(aux_state_dict)[1].startswith('module.'):
            aux_state_dict = dict_rename(aux_state_dict)
        aux_num_classes = aux_state_dict['classif_net.output.kernel'].shape[1]
        auxModel = XVec(**mergeDicts(xvec_args, {'num_classes': aux_num_classes}))
        auxModel.load_state_dict(aux_state_dict)
    elif afm_class == 'rawnet3':
        auxModel = RawNet3(Bottle2neck, model_scale=8, context=True, summed=True, encoder_type="ECA", nOut=256, out_bn=False, sinc_stride=10, log_sinc=True, norm_sinc="mean", grad_mult=1)
        aux_state_dict = torch.load(rawnet_path, map_location=torch.device('cpu'))
        try: aux_state_dict = aux_state_dict['model']
        except: pass
        if list(aux_state_dict)[1].startswith('__S__.'):
            aux_state_dict = dict_rename(aux_state_dict, token='__S__.')
        auxModel.load_state_dict(aux_state_dict, strict=False)  # this load without final linear layer
    else:
        raise Exception(f'{afm_class=}')
    auxModel.eval()
#    _ = auxModel.train(mode=False)
#    freeze_nn(auxModel)     # neural network hence becomes a fixed function; its paramaters are not trainable now
    toggle_grad(auxModel, False)
    print(f'LOADED: {aux_path}')
    return auxModel


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


def get_intermediate_activations(m, x, n=4, use_lastDFLactivation=False, use_firstandlastDFLactivation=False):
    'x: B,T'
    if len(x.shape) == 3:
        x = x.squeeze(1)
        assert len(x.shape) == 2
    if afm_class == 'xvec':
        output = []
        x = feat_extractor(x)   # N,E,S
    ##    x = x.transpose(1,2)    # Hyperion code uses N,E,S format while mine uses N,S,E
    ##    x = preproc(x)
        x = x.unsqueeze(1)
        if m.in_norm:   # hyperion specific
            x = m.in_bn(x)
        m = m.encoder_net
        #1
        x = m.in_block(x)
        output.append(x)
        if n==1 and not use_lastDFLactivation:
            return output
        #2
        x = m.layer1(x)
        output.append(x)
        if n==2 and not use_lastDFLactivation:
            return output
        #3
        x = m.layer2(x)
        output.append(x)
        if n==3 and not use_lastDFLactivation:
            return output
        #4
        x = m.layer3(x)
        output.append(x)
        if n==4 and not use_lastDFLactivation:
            return output
        x = m.layer4(x)
        output.append(x)
    elif afm_class == 'rawnet3':
        x = x / (2**15 - 1)
        output = m(x, n=n)
    else:
        raise NotImplementedError(f'{afm_class=}')
    if use_lastDFLactivation:
        return [output[-1]]
    elif use_firstandlastDFLactivation:
        assert len(output) > 2, f'{len(output)=}'
        return [output[0], output[-1]]
    else:
        #print([_.shape for _ in output]) ; raise Exception
        return output


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

        self.D_A = get_discriminator(rank=self.rank).to(device)
        self.D_A = model_for_ddp(self.D_A)

        self.D_B = get_discriminator(rank=self.rank).to(device)
        self.D_B = model_for_ddp(self.D_B)

        if clr_cyclegan:
            self.E_A = Emo_Raw_TDNN(dim_noise=dim_noise).to(device)
            self.E_A = model_for_ddp(self.E_A)
            if identical_encoding_nws:
                self.E_B = self.E_A
            else:
                self.E_B = Emo_Raw_TDNN(dim_noise=dim_noise).to(device)
                self.E_B = model_for_ddp(self.E_B)
            if type_loss_clr == 'disc':
                self.D_E = get_latent_discriminator(rank=self.rank).to(device)
                self.D_E = model_for_ddp(self.D_E)

        if sigmoid_in_disc_loss_calc:
            self.disc_output_masker = nn.Sigmoid()
        else:
            self.disc_output_masker = nn.Identity()
        self.disc_output_masker = self.disc_output_masker.to(device)

        if B2A_preproc_dir:
            self.B2A_preproc = get_B2A_preproc().to(device)

        # criterions
        self.criterion_base = get_criterion_base().to(device)
        self.criterion = get_criterion()
        if use_criterion_identity:
            self.criterion_iden = self.criterion_base
        else:
            self.criterion_iden = self.criterion_cycle
        try:
            self.criterion = self.criterion.to(device)
        except Exception as e:
            print(e)
        self.MSE_criterion = nn.MSELoss().to(device)
        self.MAE_criterion = nn.L1Loss().to(device)
        self.KLDivcriterion = nn.KLDivLoss(reduction='batchmean').to(device)
        if 'afm' in type_sup_loss.split(','):
            self.auxmodel = get_auxModel(aux_path, afm_class)
            self.auxmodel = self.auxmodel.to(device)

        # optimizer
        G_params = list(self.G_A2B.parameters()) + list(self.G_B2A.parameters())
        D_params = list(self.D_A.parameters()) + list(self.D_B.parameters())
        if clr_cyclegan:
            G_params = G_params + list(self.E_A.parameters()) + list(self.E_B.parameters())
            if type_loss_clr == 'disc':
                D_params = D_params + list(self.D_E.parameters())
        self.G_optimizer = self.create_optimizer(G_params, lr_G)
        self.D_optimizer = self.create_optimizer(D_params, lr_D)

        # scheduler

        # constants
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

        #
        if film_do:
            self.sslfeat_extractor = SSLFeatureExtractor(class_ssl=film_ssl_class, film_wav2vec2_mpath=film_wav2vec2_mpath, preproc_SSL=preproc_SSL,
                film_ssl_weightedsum=film_ssl_weightedsum, film_ssl_wsum_learnable=film_ssl_wsum_learnable).to(device)
#            self.sslfeat_extractor = model_for_ddp(self.sslfeat_extractor)

        # resume model
        self.latest_iteration = self.resume_model()     # how many iterations done already (0:= training not started yet)
        if self.rank == 0:
            if self.latest_iteration == 0:
                backup_if_exists(log_training)
                backup_if_exists(log_progress)
                print('iteration', 'G_A2B [T]', 'G_B2A [T]', 'D_A [T]', 'D_B [T]', 'cycle [T]', 'iden [T]', 't [T]',
                                   'G_A2B [V]', 'G_B2A [V]', 'D_A [V]', 'D_B [V]', 'cycle [V]', 'iden [V]', 't [V]', 'lr_G', 'lr_D', sep=',', file=open(log_progress, 'a'))    # T:train, V:validation
            else:
                print(f'RESUMING @ {getcurrtimestamp()}', file=open(log_training, 'a'))
                print(f'RESUMING @ {getcurrtimestamp()}', file=open(log_progress, 'a'))

    def create_optimizer(self, params, lr):
        if class_optimizer == 'adam':
            optimizer_base = optim.Adam
        elif class_optimizer == 'lion':
            optimizer_base = Lion
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

    def criterion_clr(self, gt, pred, domain=None):
        assert domain in ['G', 'D', None]   # whether it is real_G || real_D || generated_D is determined later manually
        if type_loss_clr == 'l1':
            loss = self.MAE_criterion(gt, pred)
        elif type_loss_clr == 'l2':
            loss = self.MSE_criterion(gt, pred)
        elif type_loss_clr == 'disc':
            assert domain
            d = self.D_E(gt)
            d2 = self.D_E(pred)
            if type_adv_loss == 'dcl':
                loss = self.criterion_disc(d, d2=d2, domain=domain)
            else:
                if domain == 'G':
                    loss = (self.criterion_disc(d, domain='real_G') + self.criterion_disc(d2, domain='real_G')) / 2
                elif domain == 'D':
                    loss = (self.criterion_disc(d, domain='real_D') + self.criterion_disc(d2, domain='generated_D')) / 2
                else:
                    raise NotImplementedError(f'{domain=}')
        else:
            raise NotImplementedError(f'{type_loss_clr=}')
        return loss

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

#    def criterion_cycle(self, x, y):
#        if len(x.shape) == 2 and len(y.shape) == 3:
#            y = y.squeeze(1)
#        elif len(x.shape) == 3 and len(y.shape) == 2:
#            x = x.squeeze(1)
#        else:
#            assert len(x.shape) == len(y.shape)
#        return self.criterion(x, y)

    def criterion_cycle(self, x, y):    # criterion_sup
        if len(x.shape) == 2 and len(y.shape) == 3:
#            x = x.unsqueeze(1)
            y = y.squeeze(1)
        elif len(x.shape) == 3 and len(y.shape) == 2:
#            y = y.unsqueeze(1)
            x = x.squeeze(1)
        else:
            assert len(x.shape) == len(y.shape)
        loss = 0
        for idx,curr_type_sup_loss in enumerate(type_sup_loss.split(',')):
            sup_metric = self.criterion_base
            intermediate_extraction = False
            df_model = None
            if curr_type_sup_loss == 'fm':
                df_model = self.D
            elif curr_type_sup_loss == 'afm':
                df_model = self.auxmodel
                intermediate_extraction=True
            else:
                loss = loss + self.criterion[idx](x, y)*weights_sup_criterion[idx]
                continue
            loss = loss + self.criterion[idx](x, y, model=df_model, metric=sup_metric, intermediate_extraction=intermediate_extraction)*weights_sup_criterion[idx]
        if average_sup_loss:
            loss = loss / len(type_sup_loss.split(','))
        return loss

    def do_B2A_preproc(self, dataA, dataB):
        # order-preserving pre-processing
        if 0 < p_B2A <= 1:
            B = dataA.shape[0]
            set_idx_1 = list(*np.where((np.random.random(B) <= p_B2A) == True))
            if len(set_idx_1) > 0:
#                set_idx_2 = [_ for _ in range(B) if _ not in set_idx_1]
#                assert len(set_idx_2) > 0
                if B2A_preproc_domain == 'A':
                    data2adapt = dataA
                elif B2A_preproc_domain == 'B':
                    data2adapt = dataB
                dataB2A = torch.index_select(data2adapt, 0, torch.tensor(set_idx_1, dtype=torch.int64, device=data2adapt.device))
                with torch.no_grad():
                    dataB2A = self.B2A_preproc(dataB2A).squeeze(1)
#                dataA_subset = torch.index_select(dataA, 0, set_idx_2) # does not refer to same memory
                if B2A_keepA:
                    dataA = torch.cat((dataA, dataB2A))
                    set_idx_3 = random.choices(range(dataA.shape[0]), k=B)
                    dataA = torch.index_select(dataA, 0, torch.tensor(set_idx_3, dtype=torch.int64, device=dataA.device))
                else:
                    for ii, index in enumerate(set_idx_1):
                        dataA[index] = dataB2A[ii]
        return dataA

    def do_train(self):
        for iteration in range(self.latest_iteration+1, n_iterations+int(do_last_iteration)):
            loss_G_A2B_mean, loss_G_A2B_std, loss_G_B2A_mean, loss_G_B2A_std, loss_cycle_mean, loss_cycle_std, loss_iden_mean, loss_iden_std, loss_D_A_mean, loss_D_A_std, loss_D_B_mean, loss_D_B_std, \
                lr_G_curr, lr_D_curr, time_per_iteration = self.do_train_single_epoch(iteration)
            if self.rank == 0:
                print(iteration, f'{loss_G_A2B_mean} ({loss_G_A2B_std})', f'{loss_G_B2A_mean} ({loss_G_B2A_std})', f'{loss_cycle_mean} ({loss_cycle_std})', f'{loss_iden_mean} ({loss_iden_std})',
                    f'{loss_D_A_mean} ({loss_D_A_std})', f'{loss_D_B_mean} ({loss_D_B_std})',
                    round(time_per_iteration), sep=',', end=',', file=open(log_progress, 'a')) # std in brackets this time
            if not skip_do_validate:
                loss_G_A2B_mean, loss_G_A2B_std, loss_G_B2A_mean, loss_G_B2A_std, loss_cycle_mean, loss_cycle_std, loss_iden_mean, loss_iden_std, loss_D_A_mean, loss_D_A_std, loss_D_B_mean, loss_D_B_std, \
                    lr_G_curr_after, lr_D_curr_after, time_per_iteration, last_val_batch = self.do_validate(iteration)
                assert lr_G_curr == lr_G_curr_after, f'{lr_G_curr=} {lr_G_curr_after=}'
                assert lr_D_curr == lr_D_curr_after, f'{lr_D_curr=} {lr_D_curr_after=}'
                if self.rank == 0:
                    print(iteration, f'{loss_G_A2B_mean} ({loss_G_A2B_std})', f'{loss_G_B2A_mean} ({loss_G_B2A_std})', f'{loss_cycle_mean} ({loss_cycle_std})', f'{loss_iden_mean} ({loss_iden_std})',
                        f'{loss_D_A_mean} ({loss_D_A_std})', f'{loss_D_B_mean} ({loss_D_B_std})',
                        round(time_per_iteration), lr_G_curr, lr_D_curr, sep=',', file=open(log_progress, 'a')) # std in brackets this time
                loss_G_mean = round((loss_G_A2B_mean + loss_G_B2A_mean)/2, 4)
                loss_D_mean = round((loss_D_A_mean + loss_D_B_mean)/2, 4)
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
        _ = self.D_A.train(mode=True)
        _ = self.D_B.train(mode=True)
        if clr_cyclegan:
            _ = self.E_A.train(mode=True)
            _ = self.E_B.train(mode=True)
            toggle_grad(self.E_A, True)
            toggle_grad(self.E_B, True)
            if type_loss_clr == 'disc':
                _ = self.D_E.train(mode=True)
                toggle_grad(self.D_E, True)
        # update these vectors at logging frequency (minimize memory consumption)
        G_A2B_lossvec = [EPS, 2*EPS]
        G_B2A_lossvec = [EPS, 2*EPS]
        cycle_lossvec = [EPS, 2*EPS]
        iden_lossvec = [EPS, 2*EPS]
        D_A_lossvec = [EPS, 2*EPS]
        D_B_lossvec = [EPS, 2*EPS]
        if self.rank == 0:
            time_start_step = time.time()
        Atraindataiterator = iter(Atrainloader)
        for ii, dataB in enumerate(Btrainloader):
            try:
                dataA = next(Atraindataiterator)
            except StopIteration:
                Atraindataiterator = iter(Atrainloader)
                dataA = next(Atraindataiterator)
            # set learning rate ahead of training
            curr_step = int(ii + (curr_iteration-1)*len_trainsampler)
            self.adjust_lr(curr_iteration, curr_step)
            # load data
            dataA = dataA[0].to(device, non_blocking=True).float() / audio_scale
            dataB = dataB[0].to(device, non_blocking=True).float() / audio_scale
            # B2A pre-processing
            dataA = self.do_B2A_preproc(dataA, dataB)
            if n_subbatches == 1:
                data = (dataA,), (dataB,)
            else:
                data = torch.tensor_split(dataA, n_subbatches, 0), torch.tensor_split(dataB, n_subbatches, 0)
            self.G_optimizer.zero_grad()
            self.D_optimizer.zero_grad()
            for jj, (real_A,real_B) in enumerate(zip(*data)):
                assert real_A.shape == real_B.shape, f'{real_A.shape=} {real_B.shape=}'
                # update discriminators
                if ii % D_update_periodicity == 0:
                    toggle_grad(self.G_A2B, False or GD_algo == 'simultaneous')
                    toggle_grad(self.G_B2A, False or GD_algo == 'simultaneous')
                    toggle_grad(self.D_A, True)
                    toggle_grad(self.D_B, True)
                    if clr_cyclegan and type_loss_clr == 'disc':
                        toggle_grad(self.D_E, True)
                    with autocast(enabled=use_amp):
                        d_real_A = self.D_A(real_A, e=self.sslfeat_extractor(real_A) if film_D_do else None)
                        z_B = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                        if film_do:
                            with autocast(enabled=False):
                                ssf_real_B = self.sslfeat_extractor(real_B)
                        generated_A = self.G_B2A(real_B, z=z_B, e=ssf_real_B if film_do else -1)
                        d_generated_A = self.D_A(generated_A, e=self.sslfeat_extractor(real_B) if film_D_do else None)
                        d_real_B = self.D_B(real_B, e=self.sslfeat_extractor(real_B) if film_D_do else None)
                        z_A = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                        if film_do:
                            with autocast(enabled=False):
                                ssf_real_A = self.sslfeat_extractor(real_A)
                        generated_B = self.G_A2B(real_A, z=z_A, e=ssf_real_A if film_do else -1)
                        d_generated_B = self.D_B(generated_B, e=self.sslfeat_extractor(real_A) if film_D_do else None)
                        if clr_cyclegan and type_loss_clr == 'disc':
                            e_generated_A = self.E_A(generated_A)
                            e_generated_B = self.E_B(generated_B)
                        if type_adv_loss == 'dcl':
                            loss_D_A_real = EPS*torch.tensor(1).to(device)
                            loss_D_B_real = EPS*torch.tensor(1).to(device)
                            loss_D_A_generated = EPS*torch.tensor(1).to(device)
                            loss_D_B_generated = EPS*torch.tensor(1).to(device)
                            loss_D_A = self.criterion_disc(d_real_A, d2=d_generated_A, domain='D')
                            loss_D_B = self.criterion_disc(d_real_B, d2=d_generated_B, domain='D')
                            if identity_discriminator:
                                z_iden_B = std_noise*torch.randn(real_A.shape[0], dim_noise, device=device) if clr_cyclegan else None
                                if film_do:
                                    with autocast(enabled=False):
                                        ssf_real_A = self.sslfeat_extractor(real_A)
                                generated_iden_A = self.G_B2A(real_A, z=z_iden_B, e=ssf_real_A if film_do else -1)
                                d_generated_iden_A = self.D_A(generated_iden_A, e=self.sslfeat_extractor(real_A) if film_D_do else None)
                                z_iden_A = std_noise*torch.randn(real_B.shape[0], dim_noise, device=device) if clr_cyclegan else None
                                if film_do:
                                    with autocast(enabled=False):
                                        ssf_real_B = self.sslfeat_extractor(real_B)
                                generated_iden_B = self.G_A2B(real_B, z=z_iden_A, e=ssf_real_B if film_do else -1)
                                d_generated_iden_B = self.D_B(generated_iden_B, e=self.sslfeat_extractor(real_B) if film_D_do else None)
                                loss_D_A = (loss_D_A + self.criterion_disc(d_real_A, d2=d_generated_iden_A, domain='D')) / 2
                                loss_D_B = (loss_D_B + self.criterion_disc(d_real_B, d2=d_generated_iden_B, domain='D')) / 2
                        else:
                            loss_D_A_real = self.criterion_disc(d_real_A, domain='real_D')
                            loss_D_B_real = self.criterion_disc(d_real_B, domain='real_D')
                            loss_D_A_generated = self.criterion_disc(d_generated_A, domain='generated_D')
                            loss_D_B_generated = self.criterion_disc(d_generated_B, domain='generated_D')
                            if identity_discriminator:
                                z_iden_B = std_noise*torch.randn(real_A.shape[0], dim_noise, device=device) if clr_cyclegan else None
                                if film_do:
                                    with autocast(enabled=False):
                                        ssf_real_A = self.sslfeat_extractor(real_A)
                                generated_iden_A = self.G_B2A(real_A, z=z_iden_B, e=ssf_real_A if film_do else -1)
                                d_generated_iden_A = self.D_A(generated_iden_A, e=self.sslfeat_extractor(real_A) if film_D_do else None)
                                loss_D_A_generated = (loss_D_A_generated + self.criterion_disc(d_generated_iden_A, domain='generated_D'))/2
                                z_iden_A = std_noise*torch.randn(real_B.shape[0], dim_noise, device=device) if clr_cyclegan else None
                                if film_do:
                                    with autocast(enabled=False):
                                        ssf_real_B = self.sslfeat_extractor(real_B)
                                generated_iden_B = self.G_A2B(real_B, z=z_iden_A, e=ssf_real_B if film_do else -1)
                                d_generated_iden_B = self.D_B(generated_iden_B, e=self.sslfeat_extractor(real_B) if film_D_do else None)
                                loss_D_B_generated = (loss_D_B_generated + self.criterion_disc(d_generated_iden_B, domain='generated_D'))/2
                            loss_D_real = (loss_D_A_real + loss_D_B_real) / 2   # these are just calculated for no reason
                            loss_D_generated = (loss_D_A_generated + loss_D_B_generated) / 2
                            loss_D_A = (loss_D_A_real + loss_D_A_generated) / 2
                            loss_D_B = (loss_D_B_real + loss_D_B_generated) / 2
                        loss_D = (loss_D_A + loss_D_B) / 2
                        if clr_cyclegan and type_loss_clr == 'disc':
                            loss_D_clr = (self.criterion_clr(z_B, e_generated_A, domain='D') + self.criterion_clr(z_A, e_generated_B, domain='D')) / 2
                            loss_D = (loss_D + loss_D_clr) / 2
                    if ii % log_periodicity_steps == 0:
                        D_A_lossvec.append(loss_D_A.item())
                        D_B_lossvec.append(loss_D_B.item())
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
                    toggle_grad(self.D_A, False or GD_algo == 'simultaneous')
                    toggle_grad(self.D_B, False or GD_algo == 'simultaneous')
                    if clr_cyclegan and type_loss_clr == 'disc':
                        toggle_grad(self.D_E, False)
                    with autocast(enabled=use_amp):
                        if GD_algo == 'alternative':
                            z_B = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                            if film_do:
                                with autocast(enabled=False):
                                    ssf_real_B = self.sslfeat_extractor(real_B)
                            generated_A = self.G_B2A(real_B, z=z_B, e=ssf_real_B if film_do else -1)
                        else:   # generated_A exists, just detach it
                            generated_A = generated_A.detach()
                        if clr_cyclegan:
                            e_generated_A = self.E_A(generated_A)
                        d_generated_A = self.D_A(generated_A, e=self.sslfeat_extractor(real_B) if film_D_do else None)
                        if GD_algo == 'alternative':
                            z_A = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                            if film_do:
                                with autocast(enabled=False):
                                    ssf_real_A = self.sslfeat_extractor(real_A)
                            generated_B = self.G_A2B(real_A, z=z_A, e=ssf_real_A if film_do else -1)
                        else:   # similar to generated_A, generated_B exists
                            generated_B = generated_B.detach()
                        if clr_cyclegan:
                            e_generated_B = self.E_B(generated_B)
                        d_generated_B = self.D_B(generated_B, e=self.sslfeat_extractor(real_A) if film_D_do else None)
                        if bicyclegan:
                            if bicyclegan_ver == 'v1':
                                e_real_A = self.E_A(real_A)
                                e_real_B = self.E_B(real_B)
                            elif bicyclegan_ver == 'v2':
                                e_real_A = self.E_B(real_A)
                                e_real_B = self.E_A(real_B)
                        if film_do:
                            with autocast(enabled=False):
                                ssf_generated_B = self.sslfeat_extractor(generated_B)
                        cycle_A = self.G_B2A(generated_B, z=e_real_A if bicyclegan else z_A, e=ssf_generated_B if film_do else -1)
                        if film_do:
                            with autocast(enabled=False):
                                ssf_generated_A = self.sslfeat_extractor(generated_A)
                        cycle_B = self.G_A2B(generated_A, z=e_real_B if bicyclegan else z_B, e=ssf_generated_A if film_do else -1)    # use same noise
                        if GD_algo == 'alternative' or (GD_algo == 'simultaneous' and not identity_discriminator):
                            z_iden_B = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                            if film_do:
                                with autocast(enabled=False):
                                    ssf_real_A = self.sslfeat_extractor(real_A)
                            generated_iden_A = self.G_B2A(real_A, z=z_iden_B, e=ssf_real_A if film_do else -1)
                            z_iden_A = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                            if film_do:
                                with autocast(enabled=False):
                                    ssf_real_B = self.sslfeat_extractor(real_B)
                            generated_iden_B = self.G_A2B(real_B, z=z_iden_A, e=ssf_real_B if film_do else -1)

                        if type_adv_loss == 'dcl':
                            d_real_A = self.D_A(real_A, e=self.sslfeat_extractor(real_A) if film_D_do else None)
                            d_real_B = self.D_B(real_B, e=self.sslfeat_extractor(real_B) if film_D_do else None)
                            loss_G_A2B_disc = self.criterion_disc(d_real_B, d2=d_generated_B, domain='G')
                            loss_G_B2A_disc = self.criterion_disc(d_real_A, d2=d_generated_A, domain='G')
                        else:
                            loss_G_A2B_disc = self.criterion_disc(d_generated_B, domain='real_G')
                            loss_G_B2A_disc = self.criterion_disc(d_generated_A, domain='real_G')
                        loss_G_disc = (loss_G_A2B_disc + loss_G_B2A_disc) / 2
                        loss_cycle_A = self.criterion_cycle(real_A, cycle_A)
                        loss_cycle_B = self.criterion_cycle(real_B, cycle_B)
                        loss_cycle = (cycle_loss_AtoB_ratio*loss_cycle_A + loss_cycle_B) / (cycle_loss_AtoB_ratio + 1)
                        loss_iden_A = self.criterion_iden(real_A, generated_iden_A)
                        loss_iden_B = self.criterion_iden(real_B, generated_iden_B)
                        loss_iden = (loss_iden_A + loss_iden_B) / 2
                        loss_G = loss_G_disc + self.lambda_cycle_wrapper(self.lambda_cycle) * loss_cycle + self.lambda_identity_wrapper(self.lambda_identity) * loss_iden
                        if clr_cyclegan:
                            loss_clr = (self.criterion_clr(z_B, e_generated_A, domain='G') + self.criterion_clr(z_A, e_generated_B, domain='G')) / 2
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
                        cycle_lossvec.append(loss_cycle.item())
                        iden_lossvec.append(loss_iden.item())
                        D_A_lossvec.append(loss_D_A.item())
                        D_B_lossvec.append(loss_D_B.item())
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
                print("[T] {}, {} Perc:{:.2f}% G_A2B:{:.4f} G_B2A:{:.4f} cycle_A:{:.4f} cycle_B:{:4f} cycle (w/ C):{:.4f} iden_A:{:.4f} iden_B:{:.4f} iden (w/ C):{:.4f} D_A:{:.4f} D_B:{:.4f} D_A_real:{:.4f} D_A_generated:{:.4f} D_B_real:{:.4f} D_B_generated:{:.4f} lr_G_curr:{:.7f} lr_D_curr:{:.7f} Time:{}".format(
                    curr_iteration, curr_step, ii*100/len_trainsampler, loss_G_A2B_disc.item(), loss_G_B2A_disc.item(), loss_cycle_A.item(), loss_cycle_B.item(), (self.lambda_cycle_wrapper(self.lambda_cycle) * loss_cycle).item(),
                    loss_iden_A.item(), loss_iden_B.item(), (self.lambda_identity_wrapper(self.lambda_identity) * loss_iden).item(), loss_D_A.item(), loss_D_B.item(), loss_D_A_real.item(), loss_D_A_generated.item(),
                    loss_D_B_real.item(), loss_D_B_generated.item(),
                    lr_G_curr, lr_D_curr, round(time_per_step)), file=open(log_training, 'a'))
                if learnable_cycle_loss:
                    print("learnable_cycle_loss AFTER act:{:.4f}".format(self.lambda_cycle_wrapper(self.lambda_cycle).item()), file=open(log_training, 'a'))
                if learnable_identity_loss:
                    print("learnable_identity_loss AFTER act:{:.4f}".format(self.lambda_identity_wrapper(self.lambda_identity).item()), file=open(log_training, 'a'))
                if clr_cyclegan:
                    print("clr_cyclegan:{:.4f}".format(loss_clr), file=open(log_training, 'a'))
                    if kldivloss_on_encoding:
                        print("kldivloss_on_encoding:{:.4f}".format(loss_kldiv), file=open(log_training, 'a'))
                    if type_loss_clr == 'disc':
                        print("loss_D_clr:{:.4f}".format(loss_D_clr), file=open(log_training, 'a'))
                time_start_step = time.time()
                if not disable_wandb:
                    dict_to_log = {'loss_G':loss_G, 'loss_D':loss_D, 'loss_G_A2B_disc':loss_G_A2B_disc, 'loss_G_B2A_disc':loss_G_B2A_disc, 'loss_cycle':loss_cycle, 'loss_cycle_A':loss_cycle_A, 'loss_cycle_B':loss_cycle_B, 'loss_iden':loss_iden,
                                    'loss_D_A_real':loss_D_A_real, 'loss_D_A_generated':loss_D_A_generated, 'loss_D_B_real':loss_D_B_real, 'loss_D_B_generated':loss_D_B_generated,
                                    'time_per_step':time_per_step}
                    if clr_cyclegan:
                        dict_to_log = mergeDicts(dict_to_log, {'clr':loss_clr})
                        if kldivloss_on_encoding:
                            dict_to_log = mergeDicts(dict_to_log, {'kldiv':loss_kldiv})
                        if type_loss_clr == 'disc':
                            dict_to_log = mergeDicts(dict_to_log, {'loss_D_clr': loss_D_clr})
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
        loss_cycle_mean = np.mean(cycle_lossvec)
        loss_cycle_std = np.std(cycle_lossvec)
        loss_iden_mean = np.mean(iden_lossvec)
        loss_iden_std = np.std(iden_lossvec)
        loss_D_A_mean = np.mean(D_A_lossvec)
        loss_D_A_std = np.std(D_A_lossvec)
        loss_D_B_mean = np.mean(D_B_lossvec)
        loss_D_B_std = np.std(D_B_lossvec)
        lr_G_curr = get_lr(self.G_optimizer)
        lr_D_curr = get_lr(self.D_optimizer)
        res = [round(_,4) for _ in [loss_G_A2B_mean, loss_G_A2B_std, loss_G_B2A_mean, loss_G_B2A_std, loss_cycle_mean, loss_cycle_std, loss_iden_mean, loss_iden_std,
                loss_D_A_mean, loss_D_A_std, loss_D_B_mean, loss_D_B_std]]
        return *res, round(lr_G_curr,7), round(lr_D_curr,7), time_per_iteration

    def do_validate(self, curr_iteration):
        time_start_iteration = time.time()
        _ = self.G_A2B.train(mode=False)
        _ = self.G_B2A.train(mode=False)
        _ = self.D_A.train(mode=False)
        _ = self.D_B.train(mode=False)
        toggle_grad(self.G_A2B, False)
        toggle_grad(self.G_B2A, False)
        toggle_grad(self.D_A, False)
        toggle_grad(self.D_B, False)
        if clr_cyclegan:
            _ = self.E_A.train(mode=False)
            _ = self.E_B.train(mode=False)
            toggle_grad(self.E_A, False)
            toggle_grad(self.E_B, False)
            if type_loss_clr == 'disc':
                _ = self.D_E.train(mode=False)
                toggle_grad(self.D_E, False)
        # update these vectors at logging frequency (minimize memory consumption)
        G_A2B_lossvec = [EPS, 2*EPS]
        G_B2A_lossvec = [EPS, 2*EPS]
        cycle_lossvec = [EPS, 2*EPS]
        iden_lossvec = [EPS, 2*EPS]
        D_A_lossvec = [EPS, 2*EPS]
        D_B_lossvec = [EPS, 2*EPS]
        if self.rank == 0:
            time_start_step = time.time()
        with torch.inference_mode():
            Avaldataiterator = iter(Avalloader)
            for ii, dataB in enumerate(Bvalloader):
                try:
                    dataA = next(Avaldataiterator)
                except StopIteration:
                    Avaldataiterator = iter(Avalloader)
                    dataA = next(Avaldataiterator)
                # set learning rate ahead of training
                curr_step = int(ii + (curr_iteration-1)*len_valsampler)
                # load data
                dataA = dataA[0].to(device, non_blocking=True).float() / audio_scale
                dataB = dataB[0].to(device, non_blocking=True).float() / audio_scale
                real_A = dataA
                real_B = dataB
                # B2A pre-processing
                real_A = self.do_B2A_preproc(real_A, real_B)
                assert real_A.shape == real_B.shape, f'{real_A.shape=} {real_B.shape=}'
                # discriminators
                with autocast(enabled=use_amp):
                    d_real_A = self.D_A(real_A, e=self.sslfeat_extractor(real_A) if film_D_do else None)
                    z_B = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                    if film_do:
                        with autocast(enabled=False):
                            ssf_real_B = self.sslfeat_extractor(real_B)
                    generated_A = self.G_B2A(real_B, z=z_B, e=ssf_real_B if film_do else -1)
                    d_generated_A = self.D_A(generated_A, e=self.sslfeat_extractor(real_B) if film_D_do else None)
                    d_real_B = self.D_B(real_B, e=self.sslfeat_extractor(real_B) if film_D_do else None)
                    z_A = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                    if film_do:
                        with autocast(enabled=False):
                            ssf_real_A = self.sslfeat_extractor(real_A)
                    generated_B = self.G_A2B(real_A, z=z_A, e=ssf_real_A if film_do else -1)
                    d_generated_B = self.D_B(generated_B, e=self.sslfeat_extractor(real_A) if film_D_do else None)
                    if clr_cyclegan and type_loss_clr == 'disc':
                        e_generated_A = self.E_A(generated_A)
                        e_generated_B = self.E_B(generated_B)
                    if type_adv_loss == 'dcl':
                        loss_D_A = self.criterion_disc(d_real_A, d2=d_generated_A, domain='D')
                        loss_D_B = self.criterion_disc(d_real_B, d2=d_generated_B, domain='D')
                        if identity_discriminator:
                            z_iden_B = std_noise*torch.randn(real_A.shape[0], dim_noise, device=device) if clr_cyclegan else None
                            if film_do:
                                with autocast(enabled=False):
                                    ssf_real_A = self.sslfeat_extractor(real_A)
                            generated_iden_A = self.G_B2A(real_A, z=z_iden_B, e=ssf_real_A if film_do else -1)
                            d_generated_iden_A = self.D_A(generated_iden_A, e=self.sslfeat_extractor(real_A) if film_D_do else None)
                            z_iden_A = std_noise*torch.randn(real_B.shape[0], dim_noise, device=device) if clr_cyclegan else None
                            if film_do:
                                with autocast(enabled=False):
                                    ssf_real_B = self.sslfeat_extractor(real_B)
                            generated_iden_B = self.G_A2B(real_B, z=z_iden_A, e=ssf_real_B if film_do else -1)
                            d_generated_iden_B = self.D_B(generated_iden_B, e=self.sslfeat_extractor(real_B) if film_D_do else None)
                            loss_D_A = (loss_D_A + self.criterion_disc(d_real_A, d2=d_generated_iden_A, domain='D')) / 2
                            loss_D_B = (loss_D_B + self.criterion_disc(d_real_B, d2=d_generated_iden_B, domain='D')) / 2
                    else:
                        loss_D_A_real = self.criterion_disc(d_real_A, domain='real_D')
                        loss_D_B_real = self.criterion_disc(d_real_B, domain='real_D')
                        loss_D_A_generated = self.criterion_disc(d_generated_A, domain='generated_D')
                        loss_D_B_generated = self.criterion_disc(d_generated_B, domain='generated_D')
                        if identity_discriminator:
                            z_iden_B = std_noise*torch.randn(real_A.shape[0], dim_noise, device=device) if clr_cyclegan else None
                            if film_do:
                                with autocast(enabled=False):
                                    ssf_real_A = self.sslfeat_extractor(real_A)
                            generated_iden_A = self.G_B2A(real_A, z=z_iden_B, e=ssf_real_A if film_do else -1)
                            d_generated_iden_A = self.D_A(generated_iden_A, e=self.sslfeat_extractor(real_A) if film_D_do else None)
                            loss_D_A_generated = (loss_D_A_generated + self.criterion_disc(d_generated_iden_A, domain='generated_D'))/2
                            z_iden_A = std_noise*torch.randn(real_B.shape[0], dim_noise, device=device) if clr_cyclegan else None
                            if film_do:
                                with autocast(enabled=False):
                                    ssf_real_B = self.sslfeat_extractor(real_B)
                            generated_iden_B = self.G_A2B(real_B, z=z_iden_A, e=ssf_real_B if film_do else -1)
                            d_generated_iden_B = self.D_B(generated_iden_B, e=self.sslfeat_extractor(real_B) if film_D_do else None)
                            loss_D_B_generated = (loss_D_B_generated + self.criterion_disc(d_generated_iden_B, domain='generated_D'))/2
                        loss_D_real = (loss_D_A_real + loss_D_B_real) / 2   # these are just calculated for no reason
                        loss_D_generated = (loss_D_A_generated + loss_D_B_generated) / 2
                        loss_D_A = (loss_D_A_real + loss_D_A_generated) / 2
                        loss_D_B = (loss_D_B_real + loss_D_B_generated) / 2
                    loss_D = (loss_D_A + loss_D_B) / 2
                    if clr_cyclegan and type_loss_clr == 'disc':
                        loss_D_clr = (self.criterion_clr(z_B, e_generated_A, domain='D') + self.criterion_clr(z_A, e_generated_B, domain='D')) / 2
                        loss_D = (loss_D + loss_D_clr) / 2
                if ii % log_periodicity_steps == 0:
                    D_A_lossvec.append(loss_D_A.item())
                    D_B_lossvec.append(loss_D_B.item())
                # generators
                with autocast(enabled=use_amp):
                    z_B = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                    if film_do:
                        with autocast(enabled=False):
                            ssf_real_B = self.sslfeat_extractor(real_B)
                    generated_A = self.G_B2A(real_B, z=z_B, e=ssf_real_B if film_do else -1)
                    if clr_cyclegan:
                        e_generated_A = self.E_A(generated_A)
                    d_generated_A = self.D_A(generated_A, e=self.sslfeat_extractor(real_B) if film_D_do else None)
                    z_A = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                    if film_do:
                        with autocast(enabled=False):
                            ssf_real_A = self.sslfeat_extractor(real_A)
                    generated_B = self.G_A2B(real_A, z=z_A, e=ssf_real_A if film_do else -1)
                    if clr_cyclegan:
                        e_generated_B = self.E_B(generated_B)
                    d_generated_B = self.D_B(generated_B, e=self.sslfeat_extractor(real_A) if film_D_do else None)
                    if bicyclegan:
                        if bicyclegan_ver == 'v1':
                            e_real_A = self.E_A(real_A)
                            e_real_B = self.E_B(real_B)
                        elif bicyclegan_ver == 'v2':
                            e_real_A = self.E_B(real_A)
                            e_real_B = self.E_A(real_B)
                    if film_do:
                        with autocast(enabled=False):
                            ssf_generated_B = self.sslfeat_extractor(generated_B)
                    cycle_A = self.G_B2A(generated_B, z=e_real_A if bicyclegan else z_A, e=ssf_generated_B if film_do else -1)
                    if film_do:
                        with autocast(enabled=False):
                            ssf_generated_A = self.sslfeat_extractor(generated_A)
                    cycle_B = self.G_A2B(generated_A, z=e_real_B if bicyclegan else z_B, e=ssf_generated_A if film_do else -1)    # use same noise
                    z = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                    if film_do:
                        with autocast(enabled=False):
                            ssf_real_A = self.sslfeat_extractor(real_A)
                    iden_A = self.G_B2A(real_A, z=z, e=ssf_real_A if film_do else -1)
                    z = std_noise*torch.randn(real_B.shape[0],dim_noise,device=device) if clr_cyclegan else None
                    if film_do:
                        with autocast(enabled=False):
                            ssf_real_B = self.sslfeat_extractor(real_B)
                    iden_B = self.G_A2B(real_B, z=z, e=ssf_real_B if film_do else -1)

#                    loss_G_A2B_disc = self.criterion_disc(d_generated_B, domain='real')
#                    loss_G_B2A_disc = self.criterion_disc(d_generated_A, domain='real')
                    if type_adv_loss == 'dcl':
                        d_real_A = self.D_A(real_A, e=self.sslfeat_extractor(real_A) if film_D_do else None)
                        d_real_B = self.D_B(real_B, e=self.sslfeat_extractor(real_B) if film_D_do else None)
                        loss_G_A2B_disc = self.criterion_disc(d_real_B, d2=d_generated_B, domain='G')
                        loss_G_B2A_disc = self.criterion_disc(d_real_A, d2=d_generated_A, domain='G')
                    else:
                        loss_G_A2B_disc = self.criterion_disc(d_generated_B, domain='real_G')
                        loss_G_B2A_disc = self.criterion_disc(d_generated_A, domain='real_G')
                    loss_G_disc = (loss_G_A2B_disc + loss_G_B2A_disc) / 2
                    loss_cycle_A = self.criterion_cycle(real_A, cycle_A)
                    loss_cycle_B = self.criterion_cycle(real_B, cycle_B)
                    loss_cycle = (cycle_loss_AtoB_ratio*loss_cycle_A + loss_cycle_B) / (cycle_loss_AtoB_ratio + 1)
                    loss_iden_A = self.criterion_iden(real_A, iden_A)
                    loss_iden_B = self.criterion_iden(real_B, iden_B)
                    loss_iden = (loss_iden_A + loss_iden_B) / 2
                    loss_G = loss_G_disc + self.lambda_cycle_wrapper(self.lambda_cycle) * loss_cycle + self.lambda_identity_wrapper(self.lambda_identity) * loss_iden
                    if clr_cyclegan:
                        loss_clr = (self.criterion_clr(z_B, e_generated_A, domain='G') + self.criterion_clr(z_A, e_generated_B, domain='G')) / 2
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
                    cycle_lossvec.append(loss_cycle.item())
                    iden_lossvec.append(loss_iden.item())
                    D_A_lossvec.append(loss_D_A.item())
                    D_B_lossvec.append(loss_D_B.item())
            # logging 1/n
            if (ii % log_periodicity_steps) == 0 and self.rank == 0:
                time_per_step = time.time() - time_start_step
                lr_G_curr = get_lr(self.G_optimizer)
                lr_D_curr = get_lr(self.D_optimizer)
                print("[V] {}, {} Perc:{:.2f}% G_A2B:{:.4f} G_B2A:{:.4f} cycle_A:{:.4f} cycle_B:{:4f} cycle (w/ C):{:.4f} iden_A:{:.4f} iden_B:{:.4f} iden (w/ C):{:.4f} D_A:{:.4f} D_B:{:.4f} D_A_real:{:.4f} D_A_generated:{:.4f} D_B_real:{:.4f} D_B_generated:{:.4f} lr_G_curr:{:.7f} lr_D_curr:{:.7f} Time:{}".format(
                    curr_iteration, curr_step, ii*100/len_valsampler, loss_G_A2B_disc.item(), loss_G_B2A_disc.item(), loss_cycle_A.item(), loss_cycle_B.item(), (self.lambda_cycle_wrapper(self.lambda_cycle) * loss_cycle).item(),
                    loss_iden_A.item(), loss_iden_B.item(), (self.lambda_identity_wrapper(self.lambda_identity) * loss_iden).item(), loss_D_A.item(), loss_D_B.item(), loss_D_A_real.item(), loss_D_A_generated.item(),
                    loss_D_B_real.item(), loss_D_B_generated.item(),
                    lr_G_curr, lr_D_curr, round(time_per_step)), file=open(log_training, 'a'))
                if clr_cyclegan:
                    print("clr_cyclegan:{:.4f}".format(loss_clr), file=open(log_training, 'a'))
                    if kldivloss_on_encoding:
                        print("kldivloss_on_encoding:{:.4f}".format(loss_kldiv), file=open(log_training, 'a'))
                    if type_loss_clr == 'disc':
                        print("loss_D_clr:{:.4f}".format(loss_D_clr), file=open(log_training, 'a'))
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
        loss_cycle_mean = np.mean(cycle_lossvec)
        loss_cycle_std = np.std(cycle_lossvec)
        loss_iden_mean = np.mean(iden_lossvec)
        loss_iden_std = np.std(iden_lossvec)
        loss_D_A_mean = np.mean(D_A_lossvec)
        loss_D_A_std = np.std(D_A_lossvec)
        loss_D_B_mean = np.mean(D_B_lossvec)
        loss_D_B_std = np.std(D_B_lossvec)
        lr_G_curr = get_lr(self.G_optimizer)
        lr_D_curr = get_lr(self.D_optimizer)
        res = [round(_,4) for _ in [loss_G_A2B_mean, loss_G_A2B_std, loss_G_B2A_mean, loss_G_B2A_std, loss_cycle_mean, loss_cycle_std, loss_iden_mean, loss_iden_std,
                loss_D_A_mean, loss_D_A_std, loss_D_B_mean, loss_D_B_std]]
        last_val_batch = {'real_A':real_A.cpu().numpy(), 'real_B':real_B.detach().cpu().numpy(), 'generated_A':generated_A.detach().cpu().numpy(),
                            'generated_B':generated_B.detach().cpu().numpy(), 'cycle_A':cycle_A.detach().cpu().numpy(), 'cycle_B':cycle_B.detach().cpu().numpy(),
                            'iden_A':iden_A.detach().cpu().numpy(), 'iden_B':iden_B.detach().cpu().numpy(),
                            'loss_G':loss_G.item(), 'loss_G_A2B_disc':loss_G_A2B_disc.item(), 'loss_G_B2A_disc':loss_G_B2A_disc.item(),
                            'loss_cycle':loss_cycle.item(), 'loss_iden':loss_iden.item(), 'loss_D':loss_D.item(), 'loss_D_A':loss_D_A.item(), 'loss_D_B':loss_D_B.item()}
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
#            os.remove(fileToDelete)    # we will delete later
        dict_to_save = {'G_A2B': self.G_A2B.state_dict(),
                        'G_B2A': self.G_B2A.state_dict(),
                        'D_A': self.D_A.state_dict(),
                        'D_B': self.D_B.state_dict(),
                        'G_optimizer': self.G_optimizer.state_dict(),
                        'D_optimizer': self.D_optimizer.state_dict(),
                        'loss_mean_G_val': loss_mean_G_val,
                        'loss_mean_D_val': loss_mean_D_val,
                        'last_val_batch': last_val_batch,
                        'lambda_cycle': self.lambda_cycle,
                        'lambda_identity': self.lambda_identity}
        if clr_cyclegan:
            dict_to_save = mergeDicts(dict_to_save, {'E_A':self.E_A.state_dict(), 'E_B':self.E_B.state_dict(), 'lambda_encode':self.lambda_encode})
            if kldivloss_on_encoding:
                dict_to_save = mergeDicts(dict_to_save, {'lambda_kldiv':self.lambda_kldiv})
            if type_loss_clr == 'disc':
                dict_to_save = mergeDicts(dict_to_save, {'D_E': self.D_E.state_dict()})
        filename = os.path.join(dir_models, f'{curr_iteration}_{loss_mean_G_val}_{loss_mean_D_val}.pt')
        torch.save(dict_to_save, filename)
        if len(states) != 0:
            os.remove(fileToDelete)

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

            self.G_optimizer.load_state_dict(checkpoint['G_optimizer'])
            self.D_optimizer.load_state_dict(checkpoint['D_optimizer'])
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
    parser.add_argument('--Adataname', type=str, default='voxcelebcat_8k')
    parser.add_argument('--Bdatapath', type=str, default='./data')
    parser.add_argument('--Bdataname', type=str, default='voxcelebcat')
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
    parser.add_argument('--ctn_stack', type=int, default=1)
    parser.add_argument('--ctn_kernel', type=int, default=3)
    parser.add_argument('--ctn_enc_dim', type=int, default=128)
    parser.add_argument('--ctn_feature_dim', type=int, default=128)
    parser.add_argument('--ctn_TCN_dilationFactor', type=int, default=0)
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
    parser.add_argument('--type_loss_clr', type=str, default='l2', choices=['l1', 'l2', 'disc'])
    parser.add_argument('--adjust_dataloader_len_up', action='store_true')
    parser.add_argument('--p_B2A', type=float, default=0)
    parser.add_argument('--B2A_preproc_dir', type=str, default='')
    parser.add_argument('--B2A_keepA', action='store_true')
    parser.add_argument('--dim_LD_interim', type=int, default=128)
    parser.add_argument('--class_latent_disc', type=str, default='custom1')
    parser.add_argument('--B2A_preproc_domain', type=str, default='B', choices=['A', 'B'])
    parser.add_argument('--skip_do_validate', action='store_true')
    parser.add_argument('--TCN_dilationFactor', type=int, default=2)
    parser.add_argument('--film_do', action='store_true')
    parser.add_argument('--film_ver', type=str, default='v1')
    parser.add_argument('--film_d_embed', type=int, default=256)
    parser.add_argument('--film_d_embed_interim', type=int, default=0)
    parser.add_argument('--film_type_pooling', type=str, default='mean')
    parser.add_argument('--film_D_do', action='store_true')
    parser.add_argument('--film_D_ver', type=str, default='v1')
    parser.add_argument('--film_D_d_embed', type=int, default=256)
    parser.add_argument('--film_D_d_embed_interim', type=int, default=0)
    parser.add_argument('--film_D_type_pooling', type=str, default='mean')
    parser.add_argument('--film_ssl_layer', type=int, default=0)
    parser.add_argument('--film_ssl_class', type=str, default='wav2vec2')
    parser.add_argument('--film_wav2vec2_mpath', type=str, default='models_pretrained/wav2vec_small.pt')
    parser.add_argument('--film_wavlm_mpath', type=str, default='models_pretrained/WavLM-Base.pt')
    parser.add_argument('--film_hubert_mpath', type=str, default='models_pretrained/hubert_base_ls960.pt')
    parser.add_argument('--film_ssl_weightedsum', action='store_true')
    parser.add_argument('--film_ssl_wsum_learnable', action='store_true')
    parser.add_argument('--film_ssl_nlayers', type=int, default=0)
    parser.add_argument('--film_ssl_wsum_actfxn', type=str, default='identity')
    parser.add_argument('--preproc_SSL', action='store_true')
    parser.add_argument('--preproc_SSL_dir', type=str, default='')
    parser.add_argument('--afm_class', type=str, default='xvec')
    parser.add_argument('--film_alpha', type=float, default=1, help='hypp: importance value for filming')
    parser.add_argument('--film_ssl_emb_feats', action='store_true')
    parser.add_argument('--rawnet_path', type=str, default='RawNet/python/RawNet3/models_rw/weights/model.pt')
    parser.add_argument('--film_data2vec2_mpath', type=str, default='models_pretrained/d2v2_base_libri.pt')
    parser.add_argument('--type_sup_loss', type=str, default='l1', help='type of supervised loss; can be multiple CSV')
    parser.add_argument('--weights_sup_criterion', type=str, default='1')
    parser.add_argument('--aux_path', type=str, default='')
    parser.add_argument('--average_sup_loss', action='store_true')
    parser.add_argument('--use_criterion_identity', action='store_true')
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
    ctn_stack = args.ctn_stack
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
    dim_LD_interim = args.dim_LD_interim
    class_latent_disc = args.class_latent_disc
    B2A_preproc_domain = args.B2A_preproc_domain
    skip_do_validate = args.skip_do_validate
    film_do = args.film_do
    film_ver = args.film_ver
    film_d_embed = args.film_d_embed
    film_d_embed_interim = args.film_d_embed_interim
    film_type_pooling = args.film_type_pooling
    TCN_dilationFactor = args.TCN_dilationFactor
    film_D_do = args.film_D_do
    film_D_ver = args.film_D_ver
    film_D_d_embed = args.film_D_d_embed
    film_D_d_embed_interim = args.film_D_d_embed_interim
    film_D_type_pooling = args.film_D_type_pooling
    film_ssl_layer = args.film_ssl_layer
    film_ssl_class = args.film_ssl_class
    film_wav2vec2_mpath = args.film_wav2vec2_mpath
    film_wavlm_mpath = args.film_wavlm_mpath
    film_hubert_mpath = args.film_hubert_mpath
    film_ssl_weightedsum = args.film_ssl_weightedsum
    film_ssl_wsum_learnable = args.film_ssl_wsum_learnable
    film_ssl_nlayers = args.film_ssl_nlayers
    film_ssl_wsum_actfxn = args.film_ssl_wsum_actfxn
    preproc_SSL = args.preproc_SSL
    preproc_SSL_dir = args.preproc_SSL_dir
    afm_class = args.afm_class
    film_alpha = args.film_alpha
    film_ssl_emb_feats = args.film_ssl_emb_feats
    rawnet_path = args.rawnet_path
    film_data2vec2_mpath = args.film_data2vec2_mpath
    type_sup_loss = args.type_sup_loss
    weights_sup_criterion = args.weights_sup_criterion
    aux_path = args.aux_path
    average_sup_loss = args.average_sup_loss
    use_criterion_identity = args.use_criterion_identity
    #
    dir_models = os.path.join('models', projectID, experimentID)
    mkdir_safe(dir_models)
    try:
        gpu_id = int(os.environ["LOCAL_RANK"])
    except:
        gpu_id = 0
#    gpu_id = local_rank
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
    assert ctn_TCN_dilationFactor == 0, f'ctn_TCN_dilationFactor is DEPRECATED'
    if film_D_do:   # the only discriminator archs that are modified to handle FILM (for CycleGAN)
        assert class_discriminator in ['HiFiGANMultiPeriodDiscriminator']

    # post modification of args
    if withSilenceTraining:
        datapath_token = '_proc_audio'
    else:
        datapath_token = '_proc_audio_no_sil'   # by default, we train without silence (i.e. silence is removed via VAD)
    if len(type_sup_loss.split(',')) == 1:
        assert len(weights_sup_criterion.split(',')) == 1
        weights_sup_criterion = [float(weights_sup_criterion)]
    else:
        if len(weights_sup_criterion.split(',')) == len(type_sup_loss.split(',')):
            weights_sup_criterion = [float(_) for _ in weights_sup_criterion.split(',')]
        elif len(weights_sup_criterion.split(',')) == 1:
            weights_sup_criterion = [float(weights_sup_criterion) for _ in len(type_sup_loss.split(','))]
        else:
            raise Exception(f'{weights_sup_criterion=} {type_sup_loss=}')
    print(f'{weights_sup_criterion=}')


    Adatapath = os.path.join(Adatapath, Adataname)
    Bdatapath = os.path.join(Bdatapath, Bdataname)
    AorigVAD = os.path.join(Adatapath, 'vad.scp')
    BorigVAD = os.path.join(Bdatapath, 'vad.scp')
    if not skip_origVADcheck:
        assert not (not withSilenceTraining and getcol(AorigVAD, n=2) != getcol(BorigVAD, n=2)), 'source VAD mismatch so there might be alignment problem when training with silence frames'
    Adatapath = Adatapath + datapath_token
    Bdatapath = Bdatapath + datapath_token
    assert dir_exists_and_notempty(Adatapath)
    assert dir_exists_and_notempty(Bdatapath)
    if subbatch_size == 0:
        n_subbatches = 1
        subbatch_size = batch_size
    else:
        n_subbatches = batch_size // subbatch_size
#    num_workers = min(num_workers, subbatch_size)
#    assert num_workers <= 4
    if GD_algo == 'simultaneous':
        assert G_update_periodicity == D_update_periodicity == 1, f'{G_update_periodicity =} {D_update_periodicity =}'
    if p_B2A == 0:  # over-write B2A_preproc_dir if provided
        B2A_preproc_dir = ''
    if afm_class == 'rawnet3':
        assert subbatch_size != 1

    # global settings
    torch.autograd.set_detect_anomaly(not disable_detect_anamoly)
#    torch.autograd.profiler.profile(False)
#    torch.autograd.profiler.emit_nvtx(False)
#    torch.backends.cudnn.benchmark = True
#    try:
#        mp.set_start_method('spawn')
#    except RuntimeError:
#        pass
#    multiprocessing.set_start_method('forkserver')
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
#    multiprocessing.set_start_method('forkserver')

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

    Atraindatascp = os.path.join(Adatapath, 'lists_xvec/train.scp')
    Atraindata = AD(os.path.join(Adatapath, 'wav.scp'), Atraindatascp, time_durs_file=os.path.join(Adatapath, 'utt2dur'), **ADargs)
    Btraindatascp = os.path.join(Bdatapath, 'lists_xvec/train.scp')
    Btraindata = AD(os.path.join(Bdatapath, 'wav.scp'), Btraindatascp, time_durs_file=os.path.join(Bdatapath, 'utt2dur'), **ADargs)

    Avaldatascp = os.path.join(Adatapath, 'lists_xvec/val.scp')
    Avaldata = AD(os.path.join(Adatapath, 'wav.scp'), Avaldatascp, time_durs_file=os.path.join(Adatapath, 'utt2dur'), **ADargs)
    Bvaldatascp = os.path.join(Bdatapath, 'lists_xvec/val.scp')
    Bvaldata = AD(os.path.join(Bdatapath, 'wav.scp'), Bvaldatascp, time_durs_file=os.path.join(Bdatapath, 'utt2dur'), **ADargs)

    # samplers
    Samplerargs_train = {'batch_size': batch_size, 'var_batch_size': False, 'iters_per_epoch': 1.0, 'num_egs_per_class': 1, 'num_egs_per_utt': 1}
    Samplerargs_val = {'batch_size': subbatch_size, 'var_batch_size': False, 'iters_per_epoch': 1.0, 'num_egs_per_class': 1, 'num_egs_per_utt': 1}

    Atrainsampler = Sampler(Atraindata, **Samplerargs_train)
    Avalsampler = Sampler(Avaldata, **Samplerargs_val)
    Btrainsampler = Sampler(Btraindata, **Samplerargs_train)
    Bvalsampler = Sampler(Bvaldata, **Samplerargs_val)

    if adjust_dataloader_len_up:
        dataloader_len_adjust_fxn = max
    else:
        dataloader_len_adjust_fxn = min
    if disable_dataloader_len_adjust:
        len_trainsampler = dataloader_len_adjust_fxn(Atrainsampler._len, Btrainsampler._len)
        len_valsampler = dataloader_len_adjust_fxn(Avalsampler._len, Bvalsampler._len)
        Atrainsampler._len = len_trainsampler
        Btrainsampler._len = len_trainsampler
        Avalsampler._len = len_valsampler
        Bvalsampler._len = len_valsampler
    else:   # same length of both samplers is important
        len_trainsampler = math.floor(3600*hrs_per_iter/(batch_size*sample_len_sec))
        Atrainsampler._len = len_trainsampler
        Btrainsampler._len = len_trainsampler
        len_valsampler = dataloader_len_adjust_fxn(math.floor(len_trainsampler/10), Avalsampler._len, Bvalsampler._len)
        Avalsampler._len = len_valsampler
        Bvalsampler._len = len_valsampler
    print(f'{len_trainsampler=} {len_valsampler=}')
    assert len_trainsampler > 0
    assert len_valsampler > 0
    totalSteps_train = n_iterations*len_trainsampler    # total means all training
    totalSteps_val = n_iterations*len_valsampler

    # dataloaders
    loaderArgs = {'num_workers': num_workers, 'pin_memory': not disable_pin_memory, 'prefetch_factor': prefetch_factor}

    Atrainloader = torch.utils.data.DataLoader(Atraindata, batch_sampler=Atrainsampler, **loaderArgs)
    Avalloader = torch.utils.data.DataLoader(Avaldata, batch_sampler=Avalsampler, **loaderArgs)
    Btrainloader = torch.utils.data.DataLoader(Btraindata, batch_sampler=Btrainsampler, **loaderArgs)
    Bvalloader = torch.utils.data.DataLoader(Bvaldata, batch_sampler=Bvalsampler, **loaderArgs)

    #
    bwe = BWEtrainer_CycleGAN()
    bwe.do_train()
