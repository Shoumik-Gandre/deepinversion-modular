"""
Train LeNet to get a pretrained model for experiments
"""
from argparse import ArgumentParser, Namespace
from typing import Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm

from models.lenet_bn import LeNet5BN


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, dataloader: DataLoader, device) -> float:
    """Returns average loss"""
    sum_loss = torch.tensor(0.0, dtype=torch.float, device=device)
    for index, (inputs, labels) in enumerate(tqdm(dataloader), start=1):
        inputs = inputs.to(device)
        labels = labels.to(device)

        loss = F.cross_entropy(model(inputs), labels)
        sum_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return float(sum_loss) / len(dataloader)


def eval_step(model: nn.Module, dataloader: DataLoader, device) -> Tuple[float, float]:
    with torch.no_grad():
        """Returns average loss and average accuracy"""
        sum_loss = torch.tensor(0.0, dtype=torch.float, device=device)
        sum_correct = torch.tensor(0.0, dtype=torch.float, device=device)

        for index, (inputs, labels) in enumerate(tqdm(dataloader), start=1):
            inputs = inputs.to(device)
            labels = labels.to(device)
            probabilities = model(inputs)
            loss = F.cross_entropy(probabilities, labels)
            sum_loss += loss.item()
            sum_correct += (probabilities.argmax(1) == labels).sum()

        return float(sum_loss / len(dataloader)), float(sum_correct / len(dataloader.dataset)) # type: ignore


def train(
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        train_dataloader: DataLoader, 
        eval_dataloader: DataLoader, 
        num_epochs: int, 
        device
    ):
    for epoch in range(1, num_epochs+1):
        print(f"Epoch [{epoch}/{num_epochs}]")
        train_loss = train_step(model, optimizer, train_dataloader, device)
        print(train_loss)
        eval_loss, eval_accuracy = eval_step(model, eval_dataloader, device)
        print(eval_loss, eval_accuracy)


def run(args: Namespace):
    epochs = args.epochs
    batch_size = args.batch_size
    Path(args.dataset_root).mkdir(exist_ok=True, parents=True)
    Path(args.model_save_path).parent.mkdir(exist_ok=True, parents=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = torchvision.datasets.MNIST(
        root=args.dataset_root,
        train=True,
        transform=transforms.Compose([  
            transforms.Grayscale(3),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # KeepChannelsTransform((0,)),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]),
        download=True
    )
    real_images = torch.stack([train_data[0] for train_data in train_dataset])
    real_labels = torch.tensor([train_data[1] for train_data in train_dataset], dtype=torch.long)

    train_dataset = TensorDataset(real_images, real_labels)
    eval_dataset = torchvision.datasets.MNIST(
        root=args.dataset_root,
        train=False,
        transform=transforms.Compose([  
            transforms.Grayscale(3),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # KeepChannelsTransform((0,)),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]),
        download=True
    )
    train_dataloader = DataLoader(train_dataset, batch_size, True)
    eval_dataloader = DataLoader(eval_dataset, batch_size)
    net = LeNet5BN(in_channels=3, num_labels=10).to(device)
    optimizer = torch.optim.Adam(net.parameters())
    train(net, optimizer, train_dataloader, eval_dataloader, epochs, device)
    torch.save(net, args.model_save_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-i', '--dataset-root', type=str)
    parser.add_argument('-o', '--model-save-path', type=str)
    run(parser.parse_args())