Code for training time-domain (1-D CNNs) for supervised (conditional GAN/CGAN) [1] and unsupervised GANs (CycleGAN) [2].

Training scripts:
1. local/train_GAN.py - Trains a CGAN
2. local/train_cycleGAN_parallel.py - Trains a CycleGAN
3. local/train_cycleGAN_parallel_jointCGAN.py - Train a joint CGAN and CycleGAN

Additional dependencies are required like Hyperion (https://github.com/hyperion-ml/hyperion).

Notes:
Conv-TasNet [3] implementation is modified from https://github.com/naplab/Conv-TasNet

References:

[1] Mirza, Mehdi, and Simon Osindero. "Conditional generative adversarial nets." arXiv preprint arXiv:1411.1784 (2014).

[2] Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." Proceedings of the IEEE international conference on computer vision. 2017.

[3] Luo, Yi, and Nima Mesgarani. "Conv-tasnet: Surpassing ideal time–frequency magnitude masking for speech separation." IEEE/ACM transactions on audio, speech, and language processing 27.8 (2019): 1256-1266.
