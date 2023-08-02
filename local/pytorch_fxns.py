from functools import partial
import numpy
import copy

import torch
import torch.nn as nn
from torch.optim import Optimizer

from local.supp_fxns import *


class Lion(Optimizer):
  r"""Implements Lion algorithm."""

  def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
    """Initialize the hyperparameters.
    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
      lr (float, optional): learning rate (default: 1e-4)
      betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.99))
      weight_decay (float, optional): weight decay coefficient (default: 0)
    """

    if not 0.0 <= lr:
      raise ValueError('Invalid learning rate: {}'.format(lr))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
    defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
    super().__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.
    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    Returns:
      the loss.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        # Perform stepweight decay
        p.data.mul_(1 - group['lr'] * group['weight_decay'])

        grad = p.grad
        state = self.state[p]
        # State initialization
        if len(state) == 0:
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(p)

        exp_avg = state['exp_avg']
        beta1, beta2 = group['betas']

        # Weight update
        update = exp_avg * beta1 + grad * (1 - beta1)
        p.add_(torch.sign(update), alpha=-group['lr'])
        # Decay the momentum running average coefficient
        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

    return loss


def make_broadcastable(v, X):
    """Returns a view of `v` that can be broadcast with `X`.

    If `v` is a one-dimensional tensor [N] and `X` is a tensor of shape
    `[N, ..., ]`, returns a view of v with singleton dimensions appended.

    Example:
        `v` is a tensor of shape `[10]` and `X` is a tensor of shape `[10, 3, 3]`.
        We want to multiply each `[3, 3]` element of `X` by the corresponding
        element of `v` to get a matrix `Y` of shape `[10, 3, 3]` such that
        `Y[i, a, b] = v[i] * X[i, a, b]`.

        `w = make_broadcastable(v, X)` gives a `w` of shape `[10, 1, 1]`,
        and we can now broadcast `Y = w * X`.
    """
    broadcasting_shape = (-1, *[1 for _ in X.shape[1:]])
    return v.reshape(broadcasting_shape)


class DP_SGD(Optimizer):
    """Differentially Private SGD.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        max_norm (float, optional): maximum norm of the individual gradient,
            to which they will be clipped if exceeded (default: 0.01)
        stddev (float, optional): standard deviation of the added noise
            (default: 1.0)
    """

    def __init__(self, params, lr=0.1, max_norm=0.01, stddev=2.0):
        self.lr = lr
        self.max_norm = max_norm
        self.stddev = stddev
        super().__init__(params, dict())

    def step(self):
        """Performs a single optimization step.

        The function expects the gradients to have been computed by BackPACK
        and the parameters to have a ``batch_l2`` and ``grad_batch`` attribute.
        """
        l2_norms_all_params_list = []
        for group in self.param_groups:
            for p in group["params"]:
                l2_norms_all_params_list.append(p.batch_l2)

        l2_norms_all_params = torch.stack(l2_norms_all_params_list)
        total_norms = torch.sqrt(torch.sum(l2_norms_all_params, dim=0))
        scaling_factors = torch.clamp_max(total_norms / self.max_norm, 1.0)

        for group in self.param_groups:
            for p in group["params"]:
                clipped_grads = p.grad_batch * make_broadcastable(
                    scaling_factors, p.grad_batch
                )
                clipped_grad = torch.sum(clipped_grads, dim=0)

                noise_magnitude = self.stddev * self.max_norm
                noise = torch.randn_like(clipped_grad) * noise_magnitude

                perturbed_update = clipped_grad + noise

                p.data.add_(-self.lr * perturbed_update)


class FeatExtractor(nn.Module):
    def __init__(self, feat_extractor, mvn=None):
        super().__init__()
        self.feat_extractor = feat_extractor
        self.mvn = mvn

    def forward(self, x):
        f = self.feat_extractor(x)
        if self.mvn is not None:
            f = self.mvn(f)
        f = f.transpose(1,2).contiguous()
        return f


def torch_to_numpy(x, recursive=False):
    'handles upto list of list (so not really recursive)'
    if recursive:
        assert not isinstance(x, (tuple,set))
        if isinstance(x, list):
            if isinstance(x[0], list):  # list of list
                assert not isinstance(x[0][0], list)
                return [[inner.clone().detach().cpu().numpy() for inner in outer] for outer in x]
            else:
                return [copy.deepcopy(x[ii].detach().cpu().numpy()) for ii in range(len(x))]
    else:
        return x.detach().cpu().numpy()


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x 


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def modify_requires_grad_nn(model, set_grad=None):
    """
        change the requires_grad flag of all trainable params in nn.Module
    """
    assert set_grad in [True, False]
    for param in model.parameters():
        param.requires_grad = set_grad

freeze_nn = partial(modify_requires_grad_nn, set_grad=False)
unfreeze_nn = partial(modify_requires_grad_nn, set_grad=True)   # DOES NOT PUT MODEL TO TRAINING MODE, WHICH MAY BE REQUIRED


class SumOfCriterions(nn.Module):
    def __init__(self, criterions, weights=None):
        super().__init__()
        self.criterions = criterions
        assert isinstance(criterions, (tuple, list))
        self.N = len(criterions)
        self.weights = weights
        if self.weights is None:
            self.weights = [1 for _ in range(self.N)]
    def forward(self, x, y):
        loss = 0
        for ii, criterion in enumerate(self.criterions):
            loss = loss + self.weights[ii] * criterion(x, y)
        return loss


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def set_lr(optimizer, lr_in):
    for g in optimizer.param_groups:
        g['lr'] = lr_in


def torch_serialized_fn(fn, x, **args):
    return torch.stack([torch.Tensor(fn(x[_,...], **args)) for _ in range(x.shape[0])])


def get_nparams_nn(model, requires_grad=None):
    """
        get number of parameters in NN
    """
    if requires_grad is True:
        ans = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return format2str(ans)
    elif requires_grad is False:
        ans = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        return format2str(ans)
    elif requires_grad is None:     # UNTESTED
        ans = sum(p.numel() for p in model.parameters())
        return format2str(ans)
    else: raise Exception


get_ntrainableparams_nn = partial(get_nparams_nn, requires_grad=True)
