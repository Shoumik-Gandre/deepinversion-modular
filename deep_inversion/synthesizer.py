from dataclasses import dataclass
import random
from typing import Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from .regularizations import (
    l1_regularization_total_variance, 
    l2_regularization_total_variance, 
    l2_image_priors
)
# We need a regularizer in the form of a hook for DeepInversion feature map statistics regularizer term
from .hooks import FeatureMapStatisticsRegularizerHook


@dataclass
class DeepInversionHyperparams:
    iterations: int
    batch_size: int
    lr_dataset: float
    bn_first_coeff: float
    coeff_l2: float
    coeff_tv_l2: float
    coeff_tv_l1: float
    bn_coeff: float
    loss_coeff: float


@dataclass
class DeepInversionSynthesizer:
    dimensions: Tuple[int, ...]
    num_labels: int
    model: nn.Module
    criterion: nn.Module
    hyperparams: DeepInversionHyperparams
    device: torch.device

    def __post_init__(self):
        # Set up the deepinversion regularizer
        self.fm_stats_regularizer_terms = [
            FeatureMapStatisticsRegularizerHook(module) 
            for module in self.model.modules()
            if isinstance(module, nn.BatchNorm2d)
        ]
                
    def __del__(self):
        for hooks in self.fm_stats_regularizer_terms:
            hooks.close()

    def synthesize_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:

        # Initalize Synthetic Batch
        synthetic_inputs = torch.randn(size=(self.hyperparams.batch_size, *self.dimensions),
                                        dtype=torch.float, requires_grad=True, device=self.device)
        synthetic_labels = torch.randint(0, self.num_labels, (self.hyperparams.batch_size,), device=self.device)

        optimizer_dataset = torch.optim.Adam([synthetic_inputs], lr=self.hyperparams.lr_dataset)
        optimizer_dataset.zero_grad()
        self.model.eval()

        for iteration in tqdm(range(self.hyperparams.iterations)):
            # The Loss Term from the model
            loss: torch.Tensor = self.criterion(self.model(synthetic_inputs), synthetic_labels)  

            # Regularizer terms:

            # L2 image prior regularization
            regularizer_image_prior = l2_image_priors(synthetic_inputs)

            # L2 total variance regularization
            regularizer_tv_l2 = l2_regularization_total_variance(synthetic_inputs)

            # L1 total variance regularization
            regularizer_tv_l1 = l1_regularization_total_variance(synthetic_inputs)

            # Deep Inversion Novelty regularizer term
            coefficients = [self.hyperparams.bn_first_coeff] + ([1.0] * (len(self.fm_stats_regularizer_terms) - 1))
            regularizer_fm_stats = sum([
                hook.term * coefficient 
                for (coefficient, hook) in zip(coefficients, self.fm_stats_regularizer_terms)
            ])

            regularizer = (
                self.hyperparams.coeff_l2 * regularizer_image_prior
                + self.hyperparams.coeff_tv_l2 * regularizer_tv_l2
                + self.hyperparams.coeff_tv_l1 * regularizer_tv_l1
                + self.hyperparams.bn_coeff * regularizer_fm_stats
            )

            loss = self.hyperparams.loss_coeff * loss + regularizer
            
            optimizer_dataset.zero_grad()
            loss.backward()
            optimizer_dataset.step()

        return synthetic_inputs, synthetic_labels