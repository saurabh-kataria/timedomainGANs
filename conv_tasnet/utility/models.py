import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from hyperion.torch.layers.custom import Film


class cLN(nn.Module):
    def __init__(self, dimension, eps = 1e-8, trainable=True):
        super(cLN, self).__init__()
        
        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step
        
        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)
        
        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T
        
        entry_cnt = np.arange(channel, channel*(time_step+1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)
        
        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T
        
        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)
        
        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())
    
def repackage_hidden(h):
    """
    Wraps hidden states in new Variables, to detach them from their history.
    """

    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

class MultiRNN(nn.Module):
    """
    Container module for multiple stacked RNN layers.
    
    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape 
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state. The corresponding output should 
                    have shape (batch, seq_len, hidden_size).
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, num_layers=1, bidirectional=False):
        super(MultiRNN, self).__init__()

        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, num_layers, dropout=dropout, 
                                         batch_first=True, bidirectional=bidirectional)
        
        

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_direction = int(bidirectional) + 1

    def forward(self, input):
        hidden = self.init_hidden(input.size(0))
        self.rnn.flatten_parameters()
        return self.rnn(input, hidden)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.num_layers*self.num_direction, batch_size, self.hidden_size).zero_()),
                    Variable(weight.new(self.num_layers*self.num_direction, batch_size, self.hidden_size).zero_()))
        else:
            return Variable(weight.new(self.num_layers*self.num_direction, batch_size, self.hidden_size).zero_())
        
        
class FCLayer(nn.Module):
    """
    Container module for a fully-connected layer.
    
    args:
        input_size: int, dimension of the input feature. The input should have shape 
                    (batch, input_size).
        hidden_size: int, dimension of the output. The corresponding output should 
                    have shape (batch, hidden_size).
        nonlinearity: string, the nonlinearity applied to the transformation. Default is None.
    """
    
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity=None):
        super(FCLayer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.FC = nn.Linear(self.input_size, self.hidden_size, bias=bias)
        if nonlinearity:
            self.nonlinearity = getattr(F, nonlinearity)
        else:
            self.nonlinearity = None
            
        self.init_hidden()
    
    def forward(self, input):
        if self.nonlinearity is not None:
            return self.nonlinearity(self.FC(input))
        else:
            return self.FC(input)
              
    def init_hidden(self):
        initrange = 1. / np.sqrt(self.input_size * self.hidden_size)
        self.FC.weight.data.uniform_(-initrange, initrange)
        if self.bias:
            self.FC.bias.data.fill_(0)
            
            
class DepthConv1d(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, causal=False,
                film_do=False, film_ver='v1', film_d_embed=256, film_d_embed_interim=0, film_type_pooling='mean',
                film_ssl_wsum_learnable=False, film_ssl_nlayers=0, film_ssl_wsum_actfxn='identity', return_film_weights=False,
                film_alpha=1,
                ):
        super(DepthConv1d, self).__init__()
        
        self.return_film_weights = return_film_weights
        self.causal = causal
        self.skip = skip
        
        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding
        self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel, dilation=dilation,
          groups=hidden_channel,
          padding=self.padding)
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if self.causal:
            self.reg1 = cLN(hidden_channel, eps=1e-08)
            self.reg2 = cLN(hidden_channel, eps=1e-08)
        else:
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        
        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.film_do = film_do
        self.film_ver = film_ver
        self.film_d_embed = film_d_embed
        if self.film_do:
            self.film1 = Film(film_d_embed, hidden_channel, d_embed_interim=film_d_embed_interim, type_pooling=film_type_pooling,
                              film_ssl_wsum_learnable=film_ssl_wsum_learnable, film_ssl_nlayers=film_ssl_nlayers, film_ssl_wsum_actfxn=film_ssl_wsum_actfxn,
                              return_film_weights=return_film_weights, film_alpha=film_alpha)

    def forward(self, input, e=None, film_weights={'gamma':[], 'beta':[]}):
#        output = self.reg1(self.nonlinearity1(self.conv1d(input)))
        output = self.conv1d(input)
        output = self.nonlinearity1(output)
        output = self.reg1(output)
        if self.causal:
#            output = self.reg2(self.nonlinearity2(self.dconv1d(output)[:,:,:-self.padding]))
            output = self.dconv1d(output)[:,:,:-self.padding]
            output = self.nonlinearity2(output)
            if self.film_do:
                if self.return_film_weights:
                    output, film_weights_i = self.film1(output, e)
                    film_weights['gamma'].append(film_weights_i[0]); film_weights['beta'].append(film_weights_i[1])
                else:
                    output = self.film1(output, e)
            output = self.reg2(output)
        else:
#            output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
            output = self.dconv1d(output)
            output = self.nonlinearity2(output)
            if self.film_do:
                if self.return_film_weights:
                    output, film_weights_i = self.film1(output, e)
                    film_weights['gamma'].append(film_weights_i[0]); film_weights['beta'].append(film_weights_i[1])
                else:
                    output = self.film1(output, e)
            output = self.reg2(output)
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip, film_weights
        else:
            return residual


class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim,
                 layer, stack, kernel=3, skip=True, 
                 causal=False, dilated=True, dilationFactor=2,
                 film_do=False, film_ver='v1', film_d_embed=256, film_d_embed_interim=0, film_type_pooling='mean',
                 film_ssl_wsum_learnable=False, film_ssl_nlayers=0, film_ssl_wsum_actfxn='identity', return_film_weights=False,
                 film_alpha=1,
                ):
        super(TCN, self).__init__()
        
        # input is a sequence of features of shape (B, N, L)
        
        # normalization
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)

        self.BN = nn.Conv1d(input_dim, BN_dim, 1)
        
        # TCN for feature extraction
        self.receptive_field = 0
        self.dilated = dilated
        self.dilationFactor = dilationFactor
#        print(f'TCN: {film_ssl_wsum_learnable=}')
        
        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                if self.dilated:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=self.dilationFactor**i, padding=self.dilationFactor**i, skip=skip, causal=causal,
                                    film_do=film_do, film_ver=film_ver, film_d_embed=film_d_embed, film_d_embed_interim=film_d_embed_interim, film_type_pooling=film_type_pooling,
                                    film_ssl_wsum_learnable=film_ssl_wsum_learnable, film_ssl_nlayers=film_ssl_nlayers, film_ssl_wsum_actfxn=film_ssl_wsum_actfxn, return_film_weights=return_film_weights,
                                    film_alpha=film_alpha)) 
                else:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal,
                                    film_do=film_do, film_ver=film_ver, film_d_embed=film_d_embed, film_d_embed_interim=film_d_embed_interim, film_type_pooling=film_type_pooling,
                                    film_ssl_wsum_learnable=film_ssl_wsum_learnable, film_ssl_nlayers=film_ssl_nlayers, film_ssl_wsum_actfxn=film_ssl_wsum_actfxn, return_film_weights=return_film_weights,
                                    film_alpha=film_alpha)) 
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * self.dilationFactor**i
                    else:
                        self.receptive_field += (kernel - 1)
                    
        print("Receptive field: {:3d} frames.".format(self.receptive_field))
        
        # output layer
        
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(BN_dim, output_dim, 1)
                                   )
        
        self.skip = skip
        
    def forward(self, input, e=None, film_weights={'gamma':[], 'beta':[]}):
        
        # input shape: (B, N, L)
        
        # normalization
        output = self.BN(self.LN(input))
        
        # pass to TCN
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                residual, skip, film_weights = self.TCN[i](output, e=e, film_weights=film_weights)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output) #, film_weights=film_weights)
                output = output + residual
            
        # output layer
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)
        
        return output, film_weights
