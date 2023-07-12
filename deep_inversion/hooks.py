from typing import Tuple
import torch
import torch.nn as nn
from .regularizations import regularize_feature_map


class FeatureMapStatisticsRegularizerHook:
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss.

    Additional Notes:\n
    This is the easiest way to obtain the feature maps and regularize them.
    Once you learn pytorch hooks, this code will make much more sense.
    '''
    def __init__(self, module: nn.Module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.term: torch.Tensor = torch.tensor(0.0)

    def hook_fn(self, module: nn.BatchNorm2d, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
        # hook co compute deepinversion's feature distribution regularization

        # feature map is the input to the batchnormalization layer
        feature_map = input[0]

        assert module.running_mean is not None
        assert module.running_var is not None

        batchnorm_running_mean = module.running_mean.detach()
        batchnorm_running_var  = module.running_var.detach()

        self.term = regularize_feature_map(
            feature_map=feature_map,
            batchnorm_running_mean=batchnorm_running_mean,
            batchnorm_running_var=batchnorm_running_var
        )

    def close(self):
        self.hook.remove()