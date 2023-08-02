import matplotlib.pyplot as plt
import numpy as np

import torch


def imshowNx1(title='', invertYaxis=False, save=False, pngFileName='a', *arrs):
    '''
    USAGE:
        imshowNx1('', True, True, 'temp2', ...)
    '''
    N = len(arrs)
    print(f'{N=}')
    fig, axes = plt.subplots(nrows=N, ncols=1)
    global_min = np.inf
    global_max = -np.inf
    for idx, arr in enumerate(arrs):
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().numpy()
        try:    # for torch
            new_min = arr.double().min().item()
        except:
            new_min = arr.min()
        global_min = min(global_min, new_min)
        try:
            new_max = arr.double().max().item()
        except:
            new_max = arr.max()
        global_max = max(global_max, new_max)
    print(f'{global_min=} {global_max=}')
    idx=0
    if N == 1:
        try:
            plt.imshow(arrs[idx].double(), vmin=global_min, vmax=global_max)
        except:
            plt.imshow(arrs[idx], vmin=global_min, vmax=global_max)
        if invertYaxis:
            ax = plt.gca()
            ax.invert_yaxis()
        plt.colorbar()
        plt.suptitle(title)
    else:
        for ax in axes.flat:
            arr = arrs[idx]
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().numpy()
            try:
                im = ax.imshow(arrs[idx].double(), vmin=global_min, vmax=global_max)
            except:
                im = ax.imshow(arrs[idx], vmin=global_min, vmax=global_max)
            if invertYaxis:
                ax.invert_yaxis()
            idx += 1
        fig.colorbar(im, ax=axes.ravel().tolist())
        fig.suptitle(title)
    if save:
        plt.savefig(f'{pngFileName}.png')
        print(f'SAVED {pngFileName}.png')
        plt.close()
    else:
        mng = plt.get_current_fig_manager()
        try:
            mng.resize(*[mng.window.maximumWidth(), mng.window.maximumHeight()])
        except Exception as e:
            print(e)
        plt.show()
