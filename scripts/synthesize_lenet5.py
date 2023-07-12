import math
import os
from pathlib import Path

import fire
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.abspath('..'))
from deep_inversion import DeepInversionHyperparams, DeepInversionSynthesizer
from models.lenet_bn import LeNet5BN
from utils.save import save_synthesized_images_labelwise


def main(
        save_root: str,
        model_path: str,
        config_file: str,
    ) -> None:

    conf = OmegaConf.load(config_file)
    hyperparams = DeepInversionHyperparams(**conf.hyperparams)
    dimensions = (3, 32, 32)
    num_labels = 10
    device = torch.device('cuda')
    model: LeNet5BN = torch.load(model_path, map_location=device)
    synthesizer = DeepInversionSynthesizer(
        dimensions=dimensions, 
        num_labels=num_labels, 
        model=model, 
        criterion=nn.CrossEntropyLoss(), 
        hyperparams=hyperparams, 
        device=device
    )
    file_count_labelwise = np.zeros(10, dtype=int)
    for batch in range(conf.num_samples // hyperparams.batch_size):
        print(f"batch [{batch+1}/{math.ceil(conf.num_samples / hyperparams.batch_size)}]")
        inputs, labels = synthesizer.synthesize_batch()
        save_synthesized_images_labelwise(inputs, labels, file_count_labelwise, Path(save_root))
    
    # Last batch len(last batch) may not equal batch size
    if conf.num_samples % hyperparams.batch_size:
        print(f"batch [{math.ceil(conf.num_samples / hyperparams.batch_size)}/{math.ceil(conf.num_samples / hyperparams.batch_size)}]")
        synthesizer.hyperparams.batch_size = conf.num_samples % hyperparams.batch_size
        inputs, labels = synthesizer.synthesize_batch()
        save_synthesized_images_labelwise(inputs, labels, file_count_labelwise, Path(save_root))


if __name__ == '__main__':
    fire.Fire(main)