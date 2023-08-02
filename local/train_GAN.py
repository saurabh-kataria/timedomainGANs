#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''band-width expansion via GANs (conditional/supervised)
    NOTES:
        1. "generated" = "fake"
        2. A = narrowband domain (16k for now, may have 8k later); B = wideband domain (16k)
        3. "discriminator" = "critic"
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

from hyperion.torch.data import AudioDataset2 as AD2
from hyperion.torch.data import ClassWeightedSeqSampler as Sampler
from hyperion.torch.utils import open_device
from hyperion.torch.utils import ddp, TorchDDP
from hyperion.torch.layers import AudioFeatsFactory as AFF
from hyperion.torch.layers import MeanVarianceNorm as MVN
from hyperion.torch.models import ResNetXVector as XVec

from denoiser.demucs import Demucs
from denoiser.stft_loss import MultiResolutionSTFTLoss
#from vits.models import Generator as vitsGenerator
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


def get_expander_from_dir(modelDir):
    'take model directory path and get BWE model'
    args_file = os.path.join(modelDir, 'args.yaml')
    assert file_exists_and_notempty(args_file)
    args = read_yaml(args_file)
#    G = ...
#    load latest file
    return


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
        aux_num_classes = aux_state_dict['classif_net.output.kernel'].shape[1]  # 14274
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


def do_preproc_SSL(m, x):
    assert len(x.shape) == 2, f'{x.shape=}'
    x = x / (2**15 - 1)
    with torch.inference_mode():
        x = m(x)
    if len(x.shape) == 3:
        x = x.squeeze(1)
    assert len(x.shape) == 2
    x = x * (2**15 - 1)
    return x


def get_preproc_SSL():
    m = TasNet(num_spk=1, layer=8, enc_dim=128, stack=1, kernel=3, win=1, TCN_dilationFactor=2)
    filepattern = os.path.join(preproc_SSL_dir, f'*.pt')
    states = glob.glob(filepattern)
    assert len(states) > 0
    aux_path_full = subprocess.run(f'ls -1tv {filepattern}', shell=True, check=True, stdout=subprocess.PIPE).stdout.decode("UTF-8").split()[-1]
    state_dict = torch.load(aux_path_full, map_location=device_cpu)['G']
    for key in list(state_dict):
        if key.startswith('module.'):
            state_dict['.'.join(key.split('.')[1:])] = state_dict.pop(key)
    m.load_state_dict(state_dict)
    m.eval()
    freeze_nn(m)
    return m


class BWEtrainer_GAN:
    def __init__(self):
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
        self.G = get_generator(rank=self.rank).to(device)
        self.G = model_for_ddp(self.G)
        self.D = get_discriminator(rank=self.rank).to(device)
        self.D = model_for_ddp(self.D)
        if B2A_preproc_dir:
            self.B2A_preproc = get_B2A_preproc().to(device)

        # criterions
        self.criterion_base = get_criterion_base().to(device)
        self.criterion = get_criterion()
        try:
            self.criterion = self.criterion.to(device)
        except Exception as e:
            print(e)
        self.MSE_criterion = nn.MSELoss().to(device)
        if 'afm' in type_sup_loss.split(','):
            self.auxmodel = get_auxModel(aux_path, afm_class)
            self.auxmodel = self.auxmodel.to(device)
        if sigmoid_in_disc_loss_calc:
            self.disc_output_masker = nn.Sigmoid()
        else:
            self.disc_output_masker = nn.Identity()
        self.disc_output_masker = self.disc_output_masker.to(device)

        # optimizer
        G_params = list(self.G.parameters())
        D_params = list(self.D.parameters())
        self.G_optimizer = self.create_optimizer(G_params, lr_G)
        self.D_optimizer = self.create_optimizer(D_params, lr_D)

        # scheduler

        # constants
        self.lambda_sup = lambda_sup

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
                print('iteration', 'loss_G_mean (std) TR', 'loss_G_sup_mean (std) TR', 'loss_D_mean (std) TR', 't TR', 'lr_G', 'lr_D',\
                                   'loss_G_mean (std) VAL', 'loss_G_sup_mean (std) VAL', 'loss_D_mean (std) VAL', 't VAL', sep=',', file=open(log_progress, 'a'))
            else:
                print(f'RESUMING @ {getcurrtimestamp()}', file=open(log_training, 'a'))
                print(f'RESUMING @ {getcurrtimestamp()}', file=open(log_progress, 'a'))

    def create_optimizer(self, params, lr):
        if class_optimizer == 'adam':
            optimizer_base = optim.Adam
        elif class_optimizer == 'ranger':
            optimizer_base = Ranger
        else:
            raise NotImplementedError(f'{class_optimizer=}')
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

    def criterion_sup(self, x, y):
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
            set_idx_1 = list(*np.where((np.random.random(B) < p_B2A) == True))
            if len(set_idx_1) > 0:
#                set_idx_2 = [_ for _ in range(B) if _ not in set_idx_1]
#                assert len(set_idx_2) > 0
                if B2A_preproc_domain == 'A':
                    data2adapt = dataA
                elif B2A_preproc_domain == 'B':
                    data2adapt = dataB
                dataB2A = torch.index_select(data2adapt, 0, torch.tensor(set_idx_1, dtype=torch.int64, device=data2adapt.device))
                with torch.no_grad():
                    dataB2A = self.B2A_preproc(dataB2A)
#                dataA_subset = torch.index_select(dataA, 0, set_idx_2) # does not refer to same memory
                for ii, index in enumerate(set_idx_1):
                    dataA[index] = dataB2A[ii]
        return dataA

    def do_train(self):
        for iteration in range(self.latest_iteration+1, n_iterations):
            loss_mean_G, loss_std_G, loss_mean_G_sup, loss_std_G_sup, loss_mean_D, loss_std_D, lr_G_curr, lr_D_curr, time_per_iteration = self.do_train_single_epoch(iteration)
            if self.rank == 0:
                print(iteration, f'{loss_mean_G} ({loss_std_G})', f'{loss_mean_G_sup} ({loss_std_G_sup})', f'{loss_mean_D} ({loss_std_D})', round(time_per_iteration), lr_G_curr, lr_D_curr, sep=',', end=',', file=open(log_progress, 'a')) # std in brackets this time
            if not skip_do_validate:
                loss_mean_G, loss_std_G, loss_mean_G_sup, loss_std_G_sup, loss_mean_D, loss_std_D, lr_G_curr, lr_D_curr, time_per_iteration, last_val_batch = self.do_validate(iteration)
                if self.rank == 0:
                    print(iteration, f'{loss_mean_G} ({loss_std_G})', f'{loss_mean_G_sup} ({loss_std_G_sup})', f'{loss_mean_D} ({loss_std_D})', round(time_per_iteration), sep=',', file=open(log_progress, 'a')) # std in brackets this time
            else:
                if self.rank == 0:
                    print('\n', file=open(log_progress, 'a'))
                loss_mean_G = loss_mean_D = last_val_batch = 0
            self.save_model(iteration, loss_mean_G, loss_mean_D, last_val_batch)
        #
        if not disable_wandb and self.rank == 0:
            self.wandb_run.finish()
        ddp.ddp_cleanup()

    def do_train_single_epoch(self, curr_iteration):
        time_start_iteration = time.time()
        _ = self.G.train(mode=True)
        _ = self.D.train(mode=True)
        G_lossvec = [EPS, 2*EPS]
        G_lossvec_sup = [EPS, 2*EPS]
        D_lossvec = [EPS, 2*EPS]
        if self.rank == 0:
            time_start_step = time.time()
        for ii, (dataA,dataB) in enumerate(trainloader):
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
            self.D_optimizer.zero_grad()
            self.G_optimizer.zero_grad()
            for jj, (real_A,real_B) in enumerate(zip(*data)):
                # update discriminator
                if ii % D_update_periodicity == 0:
                    toggle_grad(self.G, False)
                    toggle_grad(self.D, True)
                    with autocast(enabled=use_amp):
                        d_real_B = self.D(real_B, e = self.sslfeat_extractor(real_A) if film_D_do else None)
                        if film_do:
                            with autocast(enabled=False):
                                ssf_real_A = self.sslfeat_extractor(real_A)
                        generated_B = self.G(real_A, e=ssf_real_A if film_do else -1)
                        d_generated_B = self.D(generated_B, e = self.sslfeat_extractor(real_A) if film_D_do else None)
                        if lambda_identity > 0 and identity_discriminator:
                            if film_do:
                                with autocast(enabled=False):
                                    ssf_real_B = self.sslfeat_extractor(real_B)
                            generated_iden_B =  self.G(real_B, e=ssf_real_B if film_do else -1)
                            d_generated_iden_B = self.D(generated_iden_B, e = self.sslfeat_extractor(real_B) if film_D_do else None)
                        if type_adv_loss == 'dcl':
                            loss_D_real = EPS*torch.tensor(1).to(device)
                            loss_D_generated = EPS*torch.tensor(1).to(device)
                            loss_D = self.criterion_disc(d_real_B, d2=d_generated_B, domain='D')
                            if lambda_identity > 0 and identity_discriminator:
                                loss_D = loss_D + self.criterion_disc(d_real_B, d2=d_generated_iden_B, domain='D')
                        else:
                            loss_D_real = self.criterion_disc(d_real_B, domain='real_D')
                            loss_D_generated = self.criterion_disc(d_generated_B, domain='generated_D')
                            loss_D = (loss_D_real + loss_D_generated) / 2
                            if lambda_identity > 0 and identity_discriminator:
                                loss_D = (loss_D + self.criterion_disc(d_generated_iden_B, domain='generated_D'))/2
                    D_lossvec.append(loss_D.item())
                    scaler.scale(loss_D).backward()
                    if jj % n_subbatches == 0:
                        if type_adv_loss == 'wgan':
                            nn.utils.clip_grad_norm_(self.D.parameters(), wgan_clip_grad_value, error_if_nonfinite=True)
                        scaler.step(self.D_optimizer)
                        scaler.update()
                        if correct_optimzero:
                            self.D_optimizer.zero_grad()
                # update generator
                if ii % G_update_periodicity == 0:
                    toggle_grad(self.G, True)
                    toggle_grad(self.D, False)
                    with autocast(enabled=use_amp):
                        if film_do:
                            with autocast(enabled=False):
                                ssf_real_A = self.sslfeat_extractor(real_A)
                        generated_B = self.G(real_A, e=ssf_real_A if film_do else -1)
                        d_generated_B = self.D(generated_B, e = self.sslfeat_extractor(real_A) if film_D_do else None)
                        if type_adv_loss == 'dcl':
                            d_real_B = self.D(real_B, e = self.sslfeat_extractor(real_A) if film_D_do else None)
                            loss_G_disc = self.criterion_disc(d_real_B, d2=d_generated_B, domain='G')
                        else:
                            loss_G_disc = self.criterion_disc(d_generated_B, domain='real_G')     # inversion
                        try:
                            loss_G_sup = self.criterion_sup(generated_B, real_B)
                        except Exception as e:
                            raise Exception(f'{e=} {generated_B.shape=} {real_B.shape=}')
                        loss_G = loss_G_disc + self.lambda_sup * loss_G_sup
                        if lambda_identity > 0:
                            if film_do:
                                with autocast(enabled=False):
                                    ssf_real_B = self.sslfeat_extractor(real_B)
                            generated_iden_B =  self.G(real_B, e=ssf_real_B if film_do else -1)
                            loss_G_iden = self.criterion_sup(generated_iden_B, real_B)
                            loss_G = loss_G + lambda_identity * loss_G_iden
                    G_lossvec.append(loss_G.item())
                    G_lossvec_sup.append(loss_G_sup.item())
                    scaler.scale(loss_G).backward()
                    if jj % n_subbatches == 0:
                        scaler.step(self.G_optimizer)
                        scaler.update()
                        if correct_optimzero:
                            self.G_optimizer.zero_grad()
            # logging 1/n
            if (ii % log_periodicity_steps) == 0 and self.rank == 0:
                time_per_step = time.time() - time_start_step
                lr_G_curr = get_lr(self.G_optimizer)
                lr_D_curr = get_lr(self.D_optimizer)
                print("[TRAIN] Iter:{} Step:{} Perc:{:.2f}% G Loss:{:.4f} D Loss:{:.4f} loss_G_disc:{:.4f} loss_G_sup (w/o const.):{:.4f} loss_G_sup (w/ const.):{:.4f} loss_D_real:{:.4f} loss_D_generated:{:.4f} lr_G_curr:{:.7f} lr_D_curr:{:.7f} Time:{}".format(
                    curr_iteration,curr_step, ii*100/len_trainsampler, loss_G.item(), loss_D.item(), loss_G_disc.item(), loss_G_sup.item(), loss_G_sup.item()*self.lambda_sup, loss_D_real.item(), loss_D_generated.item(),
                    lr_G_curr, lr_D_curr, round(time_per_step)), file=open(log_training, 'a'))
                if lambda_identity > 0:
                    print("loss_G_iden:{:.4f}".format((loss_G_iden*lambda_identity).item()), file=open(log_training, 'a'))
                time_start_step = time.time()
                if not disable_wandb:
                    dict_to_log = {'loss_G':loss_G, 'loss_D':loss_D, 'loss_G_disc':loss_G_disc, 'loss_G_sup':loss_G_sup*self.lambda_sup, 'loss_D_real':loss_D_real, 'loss_D_generated':loss_D_generated}
                    if lambda_identity > 0:
                        dict_to_log['loss_G_iden'] = loss_G_iden*lambda_identity
                    wandb.log(dict_to_log, step=curr_step)
            # logging 2/n
            if ii == (len_trainsampler - 2) and self.rank == 0:
                print_shell_cmd_output('nvidia-smi')
                print_shell_cmd_output(f'ps -Flww -p {os.getpid()}')
        time_per_iteration = time.time() - time_start_iteration
        loss_mean_G = np.mean(G_lossvec)
        loss_std_G = np.std(G_lossvec)
        loss_mean_G_sup = np.mean(G_lossvec_sup)
        loss_std_G_sup = np.std(G_lossvec_sup)
        loss_mean_D = np.mean(D_lossvec)
        loss_std_D = np.std(D_lossvec)
        lr_G_curr = get_lr(self.G_optimizer)
        lr_D_curr = get_lr(self.D_optimizer)
        res = [round(_,4) for _ in [loss_mean_G, loss_std_G, loss_mean_G_sup, loss_std_G_sup, loss_mean_D, loss_std_D]]
        return *res, round(lr_G_curr,7), round(lr_D_curr,7), time_per_iteration

    def do_validate(self, curr_iteration):
        time_start_iteration = time.time()
        _ = self.G.train(mode=False)
        _ = self.D.train(mode=False)
        G_lossvec = [EPS, 2*EPS]
        G_lossvec_sup = [EPS, 2*EPS]
        D_lossvec = [EPS, 2*EPS]
        if self.rank == 0:
            time_start_step = time.time()
        with torch.inference_mode():
            for ii, (dataA,dataB) in enumerate(valloader):
                # load data
                real_A = dataA[0].to(device, non_blocking=True).float() / audio_scale
                real_B = dataB[0].to(device, non_blocking=True).float() / audio_scale
                # B2A pre-processing
                real_A = self.do_B2A_preproc(real_A, real_B)
                # set learning rate ahead of training
                curr_step = int(ii + (curr_iteration-1)*len_valsampler)
                # discriminator
                with autocast(enabled=use_amp):
                    d_real_B = self.D(real_B, e = self.sslfeat_extractor(real_A) if film_D_do else None)
                    if film_do:
                        with autocast(enabled=False):
                            ssf_real_A = self.sslfeat_extractor(real_A)
                    generated_B = self.G(real_A, e=ssf_real_A if film_do else -1)
                    d_generated_B = self.D(generated_B, e = self.sslfeat_extractor(real_A) if film_D_do else None)
                    if lambda_identity > 0:# and identity_discriminator:
                        if film_do:
                            with autocast(enabled=False):
                                ssf_real_B = self.sslfeat_extractor(real_B)
                        generated_iden_B =  self.G(real_B, e=ssf_real_B if film_do else -1)
                        d_generated_iden_B = self.D(generated_iden_B, e = self.sslfeat_extractor(real_B) if film_D_do else None)
                    if type_adv_loss == 'dcl':
                        loss_D_real = EPS*torch.tensor(1).to(device)
                        loss_D_generated = EPS*torch.tensor(1).to(device)
                        loss_D = self.criterion_disc(d_real_B, d2=d_generated_B, domain='D')
                        if lambda_identity > 0 and identity_discriminator:
                            loss_D = loss_D + self.criterion_disc(d_real_B, d2=d_generated_iden_B, domain='D')
                    else:
                        loss_D_real = self.criterion_disc(d_real_B, domain='real_D')
                        loss_D_generated = self.criterion_disc(d_generated_B, domain='generated_D')
                        loss_D = (loss_D_real + loss_D_generated) / 2
                        if lambda_identity > 0 and identity_discriminator:
                            loss_D = (loss_D + self.criterion_disc(d_generated_iden_B, domain='generated_D'))/2
                D_lossvec.append(loss_D.item())
                # generator
                with autocast(enabled=use_amp):
#                    generated_B = self.G(real_A)   # redundant calculation
#                    d_generated_B = self.D(generated_B, e = self.sslfeat_extractor(real_A) if film_D_do else None)
                    if type_adv_loss == 'dcl':
#                        d_real_B = self.D(real_B, e = self.sslfeat_extractor(real_A) if film_D_do else None)
                        loss_G_disc = self.criterion_disc(d_real_B, d2=d_generated_B, domain='G')
                    else:
                        loss_G_disc = self.criterion_disc(d_generated_B, domain='real_G')     # inversion
                    loss_G_sup = self.criterion_sup(generated_B, real_B)
                    loss_G = loss_G_disc + self.lambda_sup * loss_G_sup
                    if lambda_identity > 0:
#                        generated_iden_B =  self.G(real_B)
                        loss_G_iden = self.criterion_sup(generated_iden_B, real_B)
                        loss_G = loss_G + lambda_identity * loss_G_iden
                G_lossvec.append(loss_G.item())
                G_lossvec_sup.append(loss_G_sup.item())
                # logging 1/n
                if (ii % log_periodicity_steps) == 0 and self.rank == 0:
                    time_per_step = time.time() - time_start_step
                    lr_G_curr = get_lr(self.G_optimizer)
                    lr_D_curr = get_lr(self.D_optimizer)
                    print("[VAL] Iter:{} Step:{} Perc:{:.2f}% G Loss:{:.4f} D Loss:{:.4f} loss_G_disc:{:.4f} loss_G_sup (w/o const.):{:.4f} loss_G_sup (w/ const.):{:.4f} loss_D_real:{:.4f} loss_D_generated:{:.4f} lr_G_curr:{:.7f} lr_D_curr:{:.7f} Time:{}".format(
                        curr_iteration, curr_step, ii*100/len_valsampler, loss_G.item(), loss_D.item(), loss_G_disc.item(), loss_G_sup.item(), loss_G_sup.item()*self.lambda_sup, loss_D_real.item(), loss_D_generated.item(),
                        lr_G_curr, lr_D_curr, round(time_per_step)), file=open(log_training, 'a'))
                    if lambda_identity > 0:
                        print("loss_G_iden:{:.4f}".format((loss_G_iden*lambda_identity).item()), file=open(log_training, 'a'))
                    time_start_step = time.time()
                # logging 2/n
                if ii == (len_valsampler - 2) and self.rank == 0:
                    print_shell_cmd_output('nvidia-smi')
                    print_shell_cmd_output(f'ps -Flww -p {os.getpid()}')
        time_per_iteration = time.time() - time_start_iteration
        loss_mean_G = np.mean(G_lossvec)
        loss_std_G = np.std(G_lossvec)
        loss_mean_G_sup = np.mean(G_lossvec_sup)
        loss_std_G_sup = np.std(G_lossvec_sup)
        loss_mean_D = np.mean(D_lossvec)
        loss_std_D = np.std(D_lossvec)
        lr_G_curr = get_lr(self.G_optimizer)
        lr_D_curr = get_lr(self.D_optimizer)
        res = [round(_,4) for _ in [loss_mean_G, loss_std_G, loss_mean_G_sup, loss_std_G_sup, loss_mean_D, loss_std_D]]
        last_val_batch = {'real_A':real_A.detach().cpu().numpy(), 'real_B':real_B.detach().cpu().numpy(), 'generated_B':generated_B.detach().cpu().numpy(),
                            'loss_D_real':loss_D_real.item(), 'loss_D_generated':loss_D_generated.item(), 'loss_G_disc':loss_G_disc.item(), 'loss_G_sup':loss_G_sup.item(), 'loss_G':loss_G.item()}
        return *res, round(lr_G_curr,7), round(lr_D_curr,7), time_per_iteration, last_val_batch

    def adjust_lr(self, curr_iteration, curr_step):
        if lr_scheduler == 'contLinearDecay':
            new_lr_G = max(lr_min, lr_G - curr_step*(lr_G-lr_min)/totalSteps_train)
            new_lr_D = max(lr_min, lr_D - curr_step*(lr_D-lr_min)/totalSteps_train)
            for gg in self.G_optimizer.param_groups:
                gg['lr'] = new_lr_G
            for gg in self.D_optimizer.param_groups:
                gg['lr'] = new_lr_D
        else:
            raise NotImplementedError(f'{lr_scheduler=}')

    def save_model(self, curr_iteration, loss_mean_G_val, loss_mean_D_val, last_val_batch):
        if not self.rank == 0:
            return
        filepattern = os.path.join(dir_models, f'*.pt')
        states = glob.glob(filepattern)
        if len(states) != 0:
            fileToDelete = subprocess.run(f'ls -1tv {filepattern}', shell=True, check=True, stdout=subprocess.PIPE).stdout.decode("UTF-8").split()[-1]
            print(f'deleting: {fileToDelete}')
#            os.remove(fileToDelete)
        dict_to_save = {'G': self.G.state_dict(),
                        'D': self.D.state_dict(),
                        'G_optimizer': self.G_optimizer.state_dict(),
                        'D_optimizer': self.D_optimizer.state_dict(),
                        'loss_mean_G_val': loss_mean_G_val,
                        'loss_mean_D_val': loss_mean_D_val,
                        'last_val_batch': last_val_batch}
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
            checkpoint = torch.load(filetoload, map_location=device)
            self.G.load_state_dict(checkpoint['G'])
            self.D.load_state_dict(checkpoint['D'])
            self.G_optimizer.load_state_dict(checkpoint['G_optimizer'])
            self.D_optimizer.load_state_dict(checkpoint['D_optimizer'])
            last_state = int(filetoload.split('/')[-1].split('_')[0])
            print(f'resumed model with {filetoload=} {last_state=}')
            return last_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experimentID', type=str, default='1')
    parser.add_argument('--projectID', type=str, default='BWE-2')
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
    parser.add_argument('--type_sup_loss', type=str, default='l1', help='type of supervised loss; can be multiple CSV')
    parser.add_argument('--disable_detect_anamoly', action='store_true')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--sample_len_sec', type=float, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--hrs_per_iter', type=float, default=50)
    parser.add_argument('--n_iterations', type=int, default=15, help="let's say iteration=epoch for this work")
    parser.add_argument('--fs', type=int, default=16000)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--device_training', type=str, default='cuda')
    parser.add_argument('--disable_pin_memory', action='store_true')
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--lambda_sup', type=float, default=1)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--audio_scale', type=float, default=2**15-1)
    parser.add_argument('--D_update_periodicity', type=int, default=2)  # needs tuning
    parser.add_argument('--G_update_periodicity', type=int, default=1)  # needs tuning
    parser.add_argument('--minimax_GD_type', type=str, default='alternative', choices=['alternative', 'simulataneous'])     # needs to tune: simulataneous should be significantly faster
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
    parser.add_argument('--ctn_enc_dim', type=int, default=128)
    parser.add_argument('--ctn_feature_dim', type=int, default=128)
    parser.add_argument('--subbatch_size', type=int, default=0)
    parser.add_argument('--GaussianNoise_sigma', type=float, default=0.1)
    parser.add_argument('--metric_criterion', type=str, default='l1')
    parser.add_argument('--hifi_D_out_chs_multiplier', type=int, default=4)
    parser.add_argument('--hifi_D_channels', type=int, default=32)
    parser.add_argument('--hifi_D_scale_channels', type=int, default=16)
    parser.add_argument('--hifi_D_period_channels', type=int, default=4)
    parser.add_argument('--frame_length', type=int, default=25)
    parser.add_argument('--frame_shift', type=int, default=10)
    parser.add_argument('--aux_path', type=str, default='')
    parser.add_argument('--sigmoid_in_disc_loss_calc', action='store_true')
    parser.add_argument('--wgan_clip_grad_value', type=float, default=0.01)
    parser.add_argument('--weights_sup_criterion', type=str, default='1')
    parser.add_argument('--lambda_identity', type=float, default=0)
    parser.add_argument('--identity_discriminator', action='store_true')
    parser.add_argument('--correct_optimzero', action='store_true')
    parser.add_argument('--average_sup_loss', action='store_true')
    parser.add_argument('--use_PowerSGD', action='store_true')
    parser.add_argument('--skip_do_validate', action='store_true')
    parser.add_argument('--topk_in_DCL', action='store_true')
    parser.add_argument('--topk_in_DCL_smallest', action='store_true')
    parser.add_argument('--topk_in_DCL_perc', type=float, default=50)
    parser.add_argument('--p_B2A', type=float, default=0)
    parser.add_argument('--B2A_preproc_dir', type=str, default='')
    parser.add_argument('--B2A_preproc_domain', type=str, default='B', choices=['A', 'B'])
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
    type_sup_loss = args.type_sup_loss
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
    lambda_sup = args.lambda_sup
    adam_beta2 = args.adam_beta2
    audio_scale = args.audio_scale
    D_update_periodicity = args.D_update_periodicity
    G_update_periodicity = args.G_update_periodicity
    minimax_GD_type = args.minimax_GD_type
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
    ctn_enc_dim = args.ctn_enc_dim
    ctn_feature_dim = args.ctn_feature_dim
    subbatch_size = args.subbatch_size
    GaussianNoise_sigma = args.GaussianNoise_sigma
    metric_criterion = args.metric_criterion
    hifi_D_out_chs_multiplier = args.hifi_D_out_chs_multiplier
    hifi_D_channels = args.hifi_D_channels
    hifi_D_scale_channels = args.hifi_D_scale_channels
    hifi_D_period_channels = args.hifi_D_period_channels
    frame_length = args.frame_length
    frame_shift = args.frame_shift
    aux_path = args.aux_path
    sigmoid_in_disc_loss_calc = args.sigmoid_in_disc_loss_calc
    wgan_clip_grad_value = args.wgan_clip_grad_value
    weights_sup_criterion = args.weights_sup_criterion
    lambda_identity = args.lambda_identity
    identity_discriminator = args.identity_discriminator
    correct_optimzero = args.correct_optimzero
    average_sup_loss = args.average_sup_loss
    use_PowerSGD = args.use_PowerSGD
    skip_do_validate = args.skip_do_validate
    topk_in_DCL = args.topk_in_DCL
    topk_in_DCL_smallest = args.topk_in_DCL_smallest
    topk_in_DCL_perc = args.topk_in_DCL_perc
    p_B2A = args.p_B2A
    B2A_preproc_dir = args.B2A_preproc_dir
    B2A_preproc_domain = args.B2A_preproc_domain
    TCN_dilationFactor = args.TCN_dilationFactor
    film_do = args.film_do
    film_ver = args.film_ver
    film_d_embed = args.film_d_embed
    film_d_embed_interim = args.film_d_embed_interim
    film_type_pooling = args.film_type_pooling
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
    #
    dir_models = os.path.join('models', projectID, experimentID)
    mkdir_safe(dir_models)
    try:
        gpu_id = int(os.environ["LOCAL_RANK"])
    except:
        gpu_id = 0
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
    assert len(weights_sup_criterion.split(',')) in [1, len(type_sup_loss.split(','))], f'{weights_sup_criterion=} {type_sup_loss=}'
    assert not (identity_discriminator and lambda_identity == 0), 'cannot have identity discriminator w/o identity loss'
    assert 0 <= p_B2A <= 1
    assert not (B2A_preproc_domain == 'B' and p_B2A == 1), f'{p_B2A=} {B2A_preproc_domain=}. If ==1, then use train_GAN_with_target.py'
    assert not (0 < p_B2A <= 1 and not B2A_preproc_dir), f'{B2A_preproc_dir=} is needed since {p_B2A=} > 0'
    assert not (p_B2A == 0 and B2A_preproc_dir), f'{B2A_preproc_dir=} should not be provided'
    if film_D_do:   # the only discriminator archs that are modified to handle FILM
        assert class_discriminator in ['ParallelWaveGANDiscriminator']
    assert not (film_ssl_weightedsum and film_ssl_wsum_learnable), 'you cant have non-learnable and learnable weighting method'
    if preproc_SSL:
        assert preproc_SSL_dir and os.path.isdir(preproc_SSL_dir), f'please specify preproc_SSL_dir {preproc_SSL_dir=}'
    if film_ssl_emb_feats:
        assert film_ssl_class in ['xvec', 'wavlm_asv']
    if film_ssl_emb_feats:
        warn('SETTING film_type_pooling = film_D_type_pooling = "none"')
        film_type_pooling = film_D_type_pooling = 'none'

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
    if p_B2A == 0:  # over-write B2A_preproc_dir if provided
        B2A_preproc_dir = ''

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
    if afm_class == 'rawnet3':
        assert subbatch_size != 1
    assert num_workers <= subbatch_size

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

    # misc defs
    log_training = os.path.join(dir_models, 'log_training.txt')
    log_progress = os.path.join(dir_models, 'log_progress.txt')
    scaler = GradScaler(enabled=use_amp)

    # film and SSL stuff
    dict_film_ssl_nlayers = {'wav2vec2': 12}    # 'wavlm'
    try:
        film_ssl_nlayers = dict_film_ssl_nlayers[film_ssl_class]
    except Exception as e:
        print(e)
        film_ssl_nlayers = 0

    # feature extractors and preprocessors
    high_freq_dict = {8000: 3700, 16000: 7600}
    dim_feats_dict = {8000: 64, 16000: 80}
    mvncontext = math.floor((25/2 + 10*150 - frame_length/2)/frame_shift)   # (25,10)=>150; (32,16)=>93     # one-side context length calculation
    mvn = MVN(left_context=mvncontext, right_context=mvncontext)
    feat_extractor_base = AFF.create('logfb', sample_frequency=fs, high_freq=high_freq_dict[fs], use_energy=False, frame_length=frame_length, frame_shift=frame_shift, num_filters=dim_feats_dict[fs])
    feat_extractor = FeatExtractor(feat_extractor_base, mvn=mvn)
    feat_extractor = feat_extractor.to(device)
    xvec_args = {'hid_act': 'relu6', 'loss_type': 'arc-softmax', 'cos_scale': 30.0, 'margin': 0.3, 'margin_warmup_epochs': 20, 'in_feats': 80, 'resnet_type': 'lresnet34', 'in_channels': 1, 'base_channels': 64, 'in_kernel_size': 3, 'in_norm': False, 'embed_dim': 256}

    # datasets
    ADargs = {'return_key': False, 'return_class': False, 'min_chunk_length': sample_len_sec, 'max_chunk_length': sample_len_sec, 'aug_cfg': None, 'rstrip_from2': rstrip_from2}

    Atraindatascp = os.path.join(Adatapath, 'lists_xvec/train.scp')
    Btraindatascp = os.path.join(Bdatapath, 'lists_xvec/train.scp')
    traindata = AD2(os.path.join(Adatapath, 'wav.scp'), os.path.join(Bdatapath, 'wav.scp'), Atraindatascp, Btraindatascp,
                    time_durs_file=os.path.join(Adatapath, 'utt2dur'), time_durs_file2=os.path.join(Bdatapath, 'utt2dur'), rstrip_key=rstrip_key, **ADargs)

    Avaldatascp = os.path.join(Adatapath, 'lists_xvec/val.scp')
    Bvaldatascp = os.path.join(Bdatapath, 'lists_xvec/val.scp')
    valdata = AD2(os.path.join(Adatapath, 'wav.scp'), os.path.join(Bdatapath, 'wav.scp'), Avaldatascp, Bvaldatascp,
                    time_durs_file=os.path.join(Adatapath, 'utt2dur'), time_durs_file2=os.path.join(Bdatapath, 'utt2dur'), rstrip_key=rstrip_key, **ADargs)

    # samplers
    Samplerargs_train = {'batch_size': batch_size, 'var_batch_size': False, 'iters_per_epoch': 1.0, 'num_egs_per_class': 1, 'num_egs_per_utt': 1}
    Samplerargs_val = {'batch_size': subbatch_size, 'var_batch_size': False, 'iters_per_epoch': 1.0, 'num_egs_per_class': 1, 'num_egs_per_utt': 1}

    trainsampler = Sampler(traindata, **Samplerargs_train)
    valsampler = Sampler(valdata, **Samplerargs_val)

    if disable_dataloader_len_adjust:
        len_trainsampler = trainsampler._len
        len_valsampler = valsampler._len
    else:
        len_trainsampler = math.floor(3600*hrs_per_iter/(batch_size*sample_len_sec))
        trainsampler._len = len_trainsampler
        len_valsampler = min(math.floor(len_trainsampler*batch_size/(10*subbatch_size)), valsampler._len)
        valsampler._len = len_valsampler
    print(f'{len_trainsampler=} {len_valsampler=}')
    assert len_trainsampler > 0
    assert len_valsampler > 0
    totalSteps_train = n_iterations*len_trainsampler    # total means all training
    totalSteps_val = n_iterations*len_valsampler

    # dataloaders
    loaderArgs = {'num_workers': num_workers, 'pin_memory': not disable_pin_memory, 'prefetch_factor': prefetch_factor}

    trainloader = torch.utils.data.DataLoader(traindata, batch_sampler=trainsampler, **loaderArgs)
    valloader = torch.utils.data.DataLoader(valdata, batch_sampler=valsampler, **loaderArgs)

    #
    bwe = BWEtrainer_GAN()
    bwe.do_train()
